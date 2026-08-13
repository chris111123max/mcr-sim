"""Shared episode/epoch bookkeeping, checkpointing, and validation runtime."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import ValidationResult


def _append_csv(path: Path, fieldnames, row) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class EpochExperimentCallback(BaseCallback):
    """Count global episodes and run rank-0 validation at epoch boundaries."""

    def __init__(
        self,
        context,
        algorithm_name: str,
        variant: str,
        epochs: int,
        episodes_per_epoch: int,
        model_dir: Path,
        run_dir: Path,
        validation_interval: int = 2,
        validation_fn: Optional[Callable[[object, int], ValidationResult]] = None,
        resume_progress: bool = True,
        episode_sync_interval_steps: int = 256,
        performance_log_interval_steps: int = 64,
    ):
        super().__init__(verbose=0)
        self.context = context
        self.algorithm_name = str(algorithm_name).lower()
        self.variant = str(variant)
        self.epochs = int(epochs)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.target_episodes = self.epochs * self.episodes_per_epoch
        self.model_dir = Path(model_dir)
        self.run_dir = Path(run_dir)
        self.validation_interval = int(validation_interval)
        self.validation_fn = validation_fn
        self.resume_progress = bool(resume_progress)
        self.episode_sync_interval_steps = max(1, int(episode_sync_interval_steps))
        self.performance_log_interval_steps = max(
            1, int(performance_log_interval_steps)
        )
        self._performance_step_count = 0
        self.global_completed_episodes = 0
        self.next_epoch = 1
        self._epoch_success_count = 0
        self._epoch_reward_sum = 0.0
        self._epoch_episode_steps_sum = 0.0
        self._epoch_out_of_vessel_count = 0
        self._epoch_wrong_branch_count = 0
        self._epoch_non_finite_count = 0
        self._epoch_timeout_count = 0
        self._pending_episode_events = []
        self.best_valid_success_rate = -1.0
        self.best_epoch = 0

    def _on_training_start(self) -> None:
        if self.resume_progress:
            self.best_valid_success_rate = float(
                getattr(self.model, "best_valid_success_rate", -1.0)
            )
            self.best_epoch = int(getattr(self.model, "best_epoch", 0))
            saved_epoch = int(getattr(self.model, "epoch", 0))
            saved_episodes = int(getattr(self.model, "global_completed_episodes", 0))
            if saved_epoch > 0 and saved_episodes == saved_epoch * self.episodes_per_epoch:
                self.next_epoch = saved_epoch + 1
                self.global_completed_episodes = saved_episodes

    def _local_episode_events(self):
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_).reshape(-1)
        infos = list(self.locals.get("infos", []))
        events = np.zeros((len(dones), 8), dtype=np.float32)
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            episode_info = info.get("episode", {})
            events[index] = [
                1.0,
                float(bool(info.get("done_by_target", False))),
                float(episode_info.get("r", 0.0)),
                float(episode_info.get("l", 0.0)),
                float(bool(info.get("done_by_out_of_vessel", False))),
                float(bool(info.get("done_by_wrong_branch", False))),
                float(bool(info.get("done_by_non_finite", False))),
                float(bool(info.get("TimeLimit.truncated", False) or info.get("terminal_reason") == "timeout")),
            ]
        return events

    def _global_pending_episode_events(self):
        """Synchronize one fixed-size block of episode statistics."""

        local_np = np.stack(self._pending_episode_events, axis=0)
        local = th.as_tensor(
            local_np,
            dtype=th.float32,
            device=self.context.device.resolved,
        )
        if not self.context.enabled:
            return local_np.reshape(-1, local_np.shape[-1])
        gathered = [th.zeros_like(local) for _ in range(self.context.world_size)]
        self.context.all_gather(gathered, local)
        # Preserve the old per-step rank/env ordering while synchronizing many
        # steps at once: [rank, step, env, field] -> [step, rank, env, field].
        stacked = th.stack(gathered, dim=0).permute(1, 0, 2, 3).contiguous()
        return stacked.view(-1, stacked.shape[-1]).detach().cpu().numpy()

    def _checkpoint_path(self, epoch: int) -> Path:
        return self.model_dir / (
            f"{self.algorithm_name}_{self.variant}_epoch_{epoch:03d}_"
            f"episodes_{self.global_completed_episodes:05d}"
        )

    def _record_model_metadata(self, epoch: int, train_success_rate: float) -> None:
        self.model.epoch = int(epoch)
        self.model.global_completed_episodes = int(self.global_completed_episodes)
        self.model.global_env_steps = int(self.model.num_timesteps * self.context.world_size)
        self.model.train_success_rate = float(train_success_rate)
        self.model.best_metric = "valid_success_rate"
        self.model.best_valid_success_rate = float(self.best_valid_success_rate)
        self.model.best_epoch = int(self.best_epoch)
        self.model.distributed_world_size_at_save = int(self.context.world_size)

    def _run_validation(self, epoch: int, checkpoint_path: Path) -> None:
        if self.validation_fn is None or epoch % self.validation_interval != 0:
            return
        error_text = ""
        if self.context.is_main:
            try:
                result = self.validation_fn(self.model, epoch)
                valid_success_rate = float(result.valid_success_rate)
                is_new_best = valid_success_rate > self.best_valid_success_rate
                if is_new_best:
                    self.best_valid_success_rate = valid_success_rate
                    self.best_epoch = int(epoch)
                    self.model.best_valid_success_rate = valid_success_rate
                    self.model.best_epoch = int(epoch)
                    self.model.best_metric = "valid_success_rate"
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                    self.model.save(str(self.model_dir / "best_valid"))
                else:
                    self.model.last_valid_success_rate = valid_success_rate
                    self.model.last_valid_epoch = int(epoch)
                # Refresh the just-written epoch checkpoint with validation
                # metadata so resuming from it preserves strict tie behavior.
                self.model.best_valid_success_rate = float(self.best_valid_success_rate)
                self.model.best_epoch = int(self.best_epoch)
                self.model.save(str(checkpoint_path))
                self.logger.record("valid/success_rate", valid_success_rate)
                self.logger.record("valid/episodes", float(result.valid_episodes), exclude="stdout")
                _append_csv(
                    self.run_dir / "valid_summary.csv",
                    [
                        "epoch", "valid_vessels", "valid_episodes", "valid_success_count",
                        "valid_success_rate", "best_valid_success_rate", "is_new_best", "checkpoint",
                    ],
                    {
                        "epoch": epoch,
                        "valid_vessels": result.valid_vessels,
                        "valid_episodes": result.valid_episodes,
                        "valid_success_count": result.valid_success_count,
                        "valid_success_rate": valid_success_rate,
                        "best_valid_success_rate": self.best_valid_success_rate,
                        "is_new_best": is_new_best,
                        "checkpoint": str(checkpoint_path) + ".zip",
                    },
                )
                for episode_result in result.episodes:
                    _append_csv(
                        self.run_dir / "valid_episodes.csv",
                        [
                            "epoch", "vessel_id", "episode_index", "seed", "success",
                            "terminal_reason", "steps", "reward",
                        ],
                        {
                            "epoch": epoch,
                            "vessel_id": episode_result.vessel_id,
                            "episode_index": episode_result.episode_index,
                            "seed": episode_result.seed,
                            "success": episode_result.success,
                            "terminal_reason": episode_result.terminal_reason,
                            "steps": episode_result.steps,
                            "reward": episode_result.reward,
                        },
                    )
                print(
                    f"[VALID][Epoch {epoch:03d}] valid_vessels={result.valid_vessels} "
                    f"valid_episodes={result.valid_episodes} "
                    f"valid_success_count={result.valid_success_count} "
                    f"valid_success_rate={valid_success_rate:.6f} "
                    f"best_valid_success_rate={self.best_valid_success_rate:.6f} "
                    f"is_new_best={is_new_best}"
                )
            except BaseException as exc:
                error_text = f"{type(exc).__name__}: {exc}"
        error_text = self.context.broadcast_text(error_text)
        self.context.barrier()
        if error_text:
            raise RuntimeError(f"Validation failed on rank 0: {error_text}")

    def _finish_epoch(self, epoch: int) -> None:
        train_success_rate = float(self._epoch_success_count / self.episodes_per_epoch)
        train_reward_mean = float(self._epoch_reward_sum / self.episodes_per_epoch)
        train_episode_steps_mean = float(
            self._epoch_episode_steps_sum / self.episodes_per_epoch
        )
        self._record_model_metadata(epoch, train_success_rate)
        checkpoint_path = self._checkpoint_path(epoch)
        if self.context.is_main:
            self.model.save(str(checkpoint_path))
            self.logger.record("train/success_rate", train_success_rate)
            self.logger.record("train/reward_mean", train_reward_mean)
            self.logger.record("train/episode_steps_mean", train_episode_steps_mean)
            self.logger.record("train/episodes", float(self.episodes_per_epoch), exclude="stdout")
            self.logger.record("train/global_completed_episodes", float(self.global_completed_episodes), exclude="stdout")
            self.logger.record("train/global_env_steps", float(self.model.global_env_steps), exclude="stdout")
            _append_csv(
                self.run_dir / "train_summary.csv",
                ["epoch", "train_episodes", "train_success_count", "train_success_rate", "train_reward_mean", "train_episode_steps_mean", "out_of_vessel_count", "wrong_branch_count", "non_finite_count", "timeout_count", "global_completed_episodes", "global_env_steps", "checkpoint"],
                {
                    "epoch": epoch,
                    "train_episodes": self.episodes_per_epoch,
                    "train_success_count": self._epoch_success_count,
                    "train_success_rate": train_success_rate,
                    "train_reward_mean": train_reward_mean,
                    "train_episode_steps_mean": train_episode_steps_mean,
                    "out_of_vessel_count": self._epoch_out_of_vessel_count,
                    "wrong_branch_count": self._epoch_wrong_branch_count,
                    "non_finite_count": self._epoch_non_finite_count,
                    "timeout_count": self._epoch_timeout_count,
                    "global_completed_episodes": self.global_completed_episodes,
                    "global_env_steps": self.model.global_env_steps,
                    "checkpoint": str(checkpoint_path) + ".zip",
                },
            )
            print(
                f"[TRAIN][Epoch {epoch:03d}] train_episodes={self.episodes_per_epoch} "
                f"train_success_count={self._epoch_success_count} "
                f"train_success_rate={train_success_rate:.6f} "
                f"train_reward_mean={train_reward_mean:.3f} "
                f"train_episode_steps_mean={train_episode_steps_mean:.1f} "
                f"global_completed_episodes={self.global_completed_episodes} "
                f"global_env_steps={self.model.global_env_steps} "
                f"checkpoint={checkpoint_path}.zip"
            )
        # No rank may collect more rollout data until the epoch checkpoint is
        # complete.  Even epochs then enter the validation barriers below.
        self.context.barrier()
        self._run_validation(epoch, checkpoint_path)
        self._epoch_success_count = 0
        self._epoch_reward_sum = 0.0
        self._epoch_episode_steps_sum = 0.0
        self._epoch_out_of_vessel_count = 0
        self._epoch_wrong_branch_count = 0
        self._epoch_non_finite_count = 0
        self._epoch_timeout_count = 0

    def _on_step(self) -> bool:
        self._pending_episode_events.append(self._local_episode_events())
        self._performance_step_count += 1
        if self._performance_step_count >= self.performance_log_interval_steps:
            performance = self.context.performance_snapshot(reset=True)
            print(
                f"[PERF][Rank {self.context.rank}] "
                f"vector_steps={self._performance_step_count} "
                f"elapsed_s={performance['elapsed_seconds']:.3f} "
                f"rollout_other_s={performance['rollout_other_seconds']:.3f} "
                f"update_s={performance['update_seconds']:.3f} "
                f"communication_s={performance['communication_seconds']:.3f} "
                f"communication_calls={performance['communication_calls']}"
            )
            self._performance_step_count = 0
        sync_interval = self.episode_sync_interval_steps if self.context.enabled else 1
        if len(self._pending_episode_events) < sync_interval:
            return True

        events = self._global_pending_episode_events()
        self._pending_episode_events.clear()
        completed_events = events[events[:, 0] != 0]
        cursor = 0
        while cursor < len(completed_events) and self.global_completed_episodes < self.target_episodes:
            remaining = self.episodes_per_epoch - (
                self.global_completed_episodes % self.episodes_per_epoch
            )
            take = min(remaining, len(completed_events) - cursor)
            accepted = completed_events[cursor : cursor + take]
            self.global_completed_episodes += len(accepted)
            self._epoch_success_count += int(accepted[:, 1].sum())
            self._epoch_reward_sum += float(accepted[:, 2].sum())
            self._epoch_episode_steps_sum += float(accepted[:, 3].sum())
            self._epoch_out_of_vessel_count += int(accepted[:, 4].sum())
            self._epoch_wrong_branch_count += int(accepted[:, 5].sum())
            self._epoch_non_finite_count += int(accepted[:, 6].sum())
            self._epoch_timeout_count += int(accepted[:, 7].sum())
            cursor += take
            if (
                self.next_epoch <= self.epochs
                and self.global_completed_episodes
                == self.next_epoch * self.episodes_per_epoch
            ):
                self._finish_epoch(self.next_epoch)
                self.next_epoch += 1
        return self.global_completed_episodes < self.target_episodes
