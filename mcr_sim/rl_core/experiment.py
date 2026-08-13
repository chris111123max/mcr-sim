"""Shared episode/epoch bookkeeping, checkpointing, and validation runtime."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch as th
import torch.distributed as dist
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
        self.global_completed_episodes = 0
        self.next_epoch = 1
        self._epoch_success_count = 0
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

    def _global_episode_events(self):
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_).reshape(-1)
        infos = list(self.locals.get("infos", []))
        successes = np.asarray(
            [bool(done and infos[index].get("done_by_target", False)) for index, done in enumerate(dones)],
            dtype=np.bool_,
        )
        local = th.tensor(
            np.stack([dones, successes], axis=1).astype(np.int32),
            dtype=th.int32,
            device=self.context.device.resolved,
        )
        if not self.context.enabled:
            return local.detach().cpu().numpy()
        gathered = [th.zeros_like(local) for _ in range(self.context.world_size)]
        dist.all_gather(gathered, local)
        return np.concatenate([item.detach().cpu().numpy() for item in gathered], axis=0)

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
        self.context.barrier()
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
        self._record_model_metadata(epoch, train_success_rate)
        checkpoint_path = self._checkpoint_path(epoch)
        if self.context.is_main:
            self.model.save(str(checkpoint_path))
            self.logger.record("train/success_rate", train_success_rate)
            self.logger.record("train/episodes", float(self.episodes_per_epoch), exclude="stdout")
            self.logger.record("train/global_completed_episodes", float(self.global_completed_episodes), exclude="stdout")
            self.logger.record("train/global_env_steps", float(self.model.global_env_steps), exclude="stdout")
            _append_csv(
                self.run_dir / "train_summary.csv",
                ["epoch", "train_episodes", "train_success_count", "train_success_rate", "global_completed_episodes", "global_env_steps", "checkpoint"],
                {
                    "epoch": epoch,
                    "train_episodes": self.episodes_per_epoch,
                    "train_success_count": self._epoch_success_count,
                    "train_success_rate": train_success_rate,
                    "global_completed_episodes": self.global_completed_episodes,
                    "global_env_steps": self.model.global_env_steps,
                    "checkpoint": str(checkpoint_path) + ".zip",
                },
            )
            print(
                f"[TRAIN][Epoch {epoch:03d}] train_episodes={self.episodes_per_epoch} "
                f"train_success_count={self._epoch_success_count} "
                f"train_success_rate={train_success_rate:.6f} "
                f"global_completed_episodes={self.global_completed_episodes} "
                f"global_env_steps={self.model.global_env_steps} "
                f"checkpoint={checkpoint_path}.zip"
            )
        # No rank may collect more rollout data until the epoch checkpoint is
        # complete.  Even epochs then enter the validation barriers below.
        self.context.barrier()
        self._run_validation(epoch, checkpoint_path)
        self._epoch_success_count = 0

    def _on_step(self) -> bool:
        events = self._global_episode_events()
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
            cursor += take
            if (
                self.next_epoch <= self.epochs
                and self.global_completed_episodes
                == self.next_epoch * self.episodes_per_epoch
            ):
                self._finish_epoch(self.next_epoch)
                self.next_epoch += 1
        return self.global_completed_episodes < self.target_episodes
