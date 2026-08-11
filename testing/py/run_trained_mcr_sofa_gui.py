#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_trained_mcr_sofa_gui_train_equiv.py

SOFA GUI inference for a trained stable-baselines3 SAC model, modified to
match the training environment step logic as closely as possible while still
running inside the SOFA GUI main loop.

Main differences from the older GUI test file:
1. The raw policy action is clipped and smoothed exactly like MCREnv.step().
2. After each GUI integration step, reward / done / info are updated using the
   same MCREnv logic: _get_observation(), _get_reward(), _get_done(), _get_info().
3. Collision parameters are NOT patched by default. This avoids changing the
   physical test environment relative to training. Use --override-collision-params
   only when you intentionally want collision-debug presets.
4. Metrics are printed from the training environment info dict after reward/state
   update, so gate progress, out_of_vessel, no_progress, terminal_reason, and
   success flags are consistent with the training-side bookkeeping.

Important note:
The SOFA GUI provides the actual animation/integration step. Therefore this file
cannot literally call env.step(action), because env.step() would call SOFA animate
again internally and can fight the GUI event loop. Instead, this file splits the
same MCREnv.step() logic across onAnimateBeginEvent/onAnimateEndEvent:
    onAnimateBeginEvent: action clipping + smoothing + _do_action(smoothed_action)
    onAnimateEndEvent: observation + reward + termination + info update
For closest parity with training, use --frame-skip 1 unless your training also
used a different env frame_skip.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# This executable lives in python/testing/py/.  Add the Python/Git root based
# on this file so direct invocation works from any current working directory.
TEST_PY_DIR = Path(__file__).resolve().parent
TESTING_DIR = TEST_PY_DIR.parent
PYTHON_ROOT = TESTING_DIR.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Gui
from stable_baselines3 import SAC

# Import environment definitions: 非 ROS GUI 版本
from mcr_sim.mcr_rl_env import MCREnv, ObservationType, EnvType
from mcr_sim.paths import PROJECT_ROOT
from mcr_sim.rl_core.base import RenderMode


def _get_component_name(component) -> str:
    """Best-effort SOFA component name extraction."""
    try:
        name = getattr(component, "name", "")
        if hasattr(name, "value"):
            return str(name.value)
        return str(name)
    except Exception:
        return ""


def _find_sofa_object_by_name(node, target_name: str):
    """Recursively find a SOFA object by component name."""
    if node is None:
        return None

    try:
        obj = getattr(node, target_name, None)
        if obj is not None:
            return obj
    except Exception:
        pass

    try:
        for obj in node.getObjects():
            if _get_component_name(obj) == target_name:
                return obj
    except Exception:
        pass

    try:
        for child in node.getChildren():
            found = _find_sofa_object_by_name(child, target_name)
            if found is not None:
                return found
    except Exception:
        pass

    return None


def _set_sofa_component_data(component, data_name: str, value: float) -> bool:
    """Set a SOFA component Data field with robust fallbacks."""
    if component is None:
        return False

    try:
        data = component.findData(data_name)
        if data is not None:
            try:
                data.value = float(value)
            except Exception:
                data.value = str(float(value))
            return True
    except Exception:
        pass

    try:
        data = getattr(component, data_name)
        if hasattr(data, "value"):
            try:
                data.value = float(value)
            except Exception:
                data.value = str(float(value))
            return True
    except Exception:
        pass

    return False


def _get_current_target_position(env: MCREnv) -> Optional[np.ndarray]:
    """Return the current RL target position, preferring env.target_position after reset."""
    try:
        target = getattr(env, "target_position", None)
        if target is not None:
            target = np.asarray(target, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(target)):
                return target
    except Exception:
        pass

    try:
        scr = getattr(env, "scene_creation_result", {}) or {}
        target = scr.get("target_position", None)
        if target is not None:
            target = np.asarray(target, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(target)):
                return target
    except Exception:
        pass

    return None


def _iter_sofa_nodes(node):
    """Yield SOFA node and all children, best-effort."""
    if node is None:
        return
    yield node
    try:
        for child in node.getChildren():
            yield from _iter_sofa_nodes(child)
    except Exception:
        return


def _set_sofa_data_any(component, data_name: str, value) -> bool:
    """Set an arbitrary SOFA Data field; supports list/array values."""
    if component is None:
        return False

    try:
        data = component.findData(data_name)
        if data is not None:
            data.value = value
            return True
    except Exception:
        pass

    try:
        data = getattr(component, data_name)
        if hasattr(data, "value"):
            data.value = value
            return True
    except Exception:
        pass

    try:
        setattr(component, data_name, value)
        return True
    except Exception:
        return False


def update_target_marker_visual(env: MCREnv, force_print: bool = False) -> bool:
    """Move the visible TargetPoint marker to env.target_position.

    The distance/reward uses env.target_position. In forced single-vessel soft
    randomization, env.reset() changes that target after createScene() created
    the original TargetPoint visual marker. Without this sync, the GUI marker is
    stale while curD is computed to the new randomized target.
    """
    target = _get_current_target_position(env)
    if target is None:
        return False

    root = getattr(env, "_sofa_root_node", None)
    if root is None:
        return False

    target_list1 = [float(target[0]), float(target[1]), float(target[2])]
    target_list2 = [target_list1]
    updated = False
    touched = []

    for node in _iter_sofa_nodes(root):
        try:
            node_name = str(getattr(node, "name", ""))
        except Exception:
            node_name = ""

        is_target_node = any(
            key in node_name.lower()
            for key in ("targetpoint", "target_point", "goalpoint", "goal_point")
        )
        if not is_target_node:
            continue

        try:
            objects = list(node.getObjects())
        except Exception:
            objects = []

        for obj in objects:
            obj_name = _get_component_name(obj)
            for data_name, value in (
                ("position", target_list2),
                ("rest_position", target_list2),
                ("translation", target_list1),
                ("center", target_list1),
            ):
                if _set_sofa_data_any(obj, data_name, value):
                    updated = True
                    touched.append(f"{node_name}/{obj_name}.{data_name}")

    try:
        scr = getattr(env, "scene_creation_result", None)
        if isinstance(scr, dict):
            scr["target_position"] = target.copy()
            scr["current_target_position"] = target.copy()
    except Exception:
        pass

    if force_print or (updated and not bool(getattr(env, "_target_marker_sync_printed", False))):
        print(
            "[TARGET_MARKER_SYNC]",
            f"target_sim=({target[0]:+.6f},{target[1]:+.6f},{target[2]:+.6f})",
            f"updated={updated}",
            "objects=" + (";".join(touched[:6]) if touched else "none"),
        )
        env._target_marker_sync_printed = True

    return bool(updated)



def apply_local_min_distance_params(root_node, contact_distance: float, alarm_distance: float) -> bool:
    """Optional debug-only patch for LocalMinDistance parameters."""
    lmd = _find_sofa_object_by_name(root_node, "localmindistance")
    if lmd is None:
        print("[COLLISION_PARAM][WARN] LocalMinDistance object 'localmindistance' not found.")
        return False

    ok_contact = _set_sofa_component_data(lmd, "contactDistance", contact_distance)
    ok_alarm = _set_sofa_component_data(lmd, "alarmDistance", alarm_distance)

    print(
        "[COLLISION_PARAM][OVERRIDE]",
        f"contactDistance={float(contact_distance):.6g}",
        f"alarmDistance={float(alarm_distance):.6g}",
        f"ok_contact={ok_contact}",
        f"ok_alarm={ok_alarm}",
    )
    return bool(ok_contact and ok_alarm)


def _obs_all_finite(observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
    if isinstance(observation, dict):
        for value in observation.values():
            if not np.all(np.isfinite(value)):
                return False
        return True
    return bool(np.all(np.isfinite(observation)))


def _sanitize_observation(observation: Union[np.ndarray, Dict[str, np.ndarray]]):
    if isinstance(observation, dict):
        return {
            k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, v in observation.items()
        }
    return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class InferenceController(Sofa.Core.Controller):
    """SOFA GUI controller with training-equivalent MCREnv step bookkeeping."""

    def __init__(
        self,
        env: MCREnv,
        model: SAC,
        initial_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        sleep_time: float = 0.0,
        print_every: int = 1,
        max_steps: int = 0,
        stop_on_train_done: bool = True,
        stop_on_threshold: bool = True,
        ignore_no_progress_done: bool = True,
        *args,
        **kwargs,
    ):
        kwargs["name"] = "InferenceController"
        super().__init__(*args, **kwargs)
        self.env = env
        self.model = model
        self.sleep_time = float(sleep_time)
        self.print_every = max(1, int(print_every))
        self.stop_on_train_done = bool(stop_on_train_done)
        self.stop_on_threshold = bool(stop_on_threshold)
        self.ignore_no_progress_done = bool(ignore_no_progress_done)

        self.step_counter = 0
        self.stopped = False
        self.obs = initial_obs
        self.last_raw_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.last_smoothed_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}

        max_steps = int(max_steps) if max_steps is not None else 0
        self.max_steps = max_steps if max_steps > 0 else None

        print("[InferenceController] Initialized in TRAIN-EQUIVALENT GUI mode.")
        print("[InferenceController] Per GUI step: predict -> clip/smooth -> _do_action -> observation/reward/done/info update.")
        print("[InferenceController] Metrics use MCREnv info after _get_reward(), not a separate GUI-only projection.")
        if self.ignore_no_progress_done:
            print("[InferenceController] no_progress_failure will be logged but ignored as a GUI-test termination condition.")
        if int(getattr(self.env, "frame_skip", 1)) != 1:
            print(
                "[InferenceController][WARN] env.frame_skip != 1. "
                "For closest parity with the current MCREnv default/training setup, use --frame-skip 1."
            )
        if self.max_steps is not None:
            print("[InferenceController] Max GUI/env steps =", self.max_steps)

        update_target_marker_visual(self.env, force_print=True)

    # ============================================================
    # SOFA GUI callbacks
    # ============================================================
    def onAnimateBeginEvent(self, event):
        """Called before the GUI simulation step; mirrors first half of MCREnv.step()."""
        if self.stopped:
            return

        try:
            raw_action, _ = self.model.predict(self.obs, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
            raw_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)

            expected_dim = int(np.prod(self.env.action_space.shape))
            if raw_action.shape[0] != expected_dim:
                fixed = np.zeros(expected_dim, dtype=np.float32)
                n = min(expected_dim, raw_action.shape[0])
                fixed[:n] = raw_action[:n]
                raw_action = fixed

            previous_smoothed_action = getattr(
                self.env,
                "_last_smoothed_action",
                np.zeros(self.env.action_space.shape, dtype=np.float32),
            )
            previous_smoothed_action = np.asarray(previous_smoothed_action, dtype=np.float32).reshape(raw_action.shape)

            alpha = float(getattr(self.env, "action_smoothing_alpha", 1.0))
            smoothed_action = alpha * raw_action + (1.0 - alpha) * previous_smoothed_action
            smoothed_action = np.asarray(smoothed_action, dtype=np.float32)

            # Same state update as MCREnv.step() before super().step(smoothed_action).
            self.env._prev_smoothed_action = previous_smoothed_action.copy()
            self.env._last_smoothed_action = smoothed_action.copy()
            self.last_raw_action = raw_action.copy()
            self.last_smoothed_action = smoothed_action.copy()

            if self.step_counter % self.print_every == 0:
                print(
                    f"[ACTION] Step={self.step_counter:04d} | "
                    f"raw=({raw_action[0]:+.4f},{raw_action[1]:+.4f},{raw_action[2]:+.4f}) "
                    f"smooth=({smoothed_action[0]:+.4f},{smoothed_action[1]:+.4f},{smoothed_action[2]:+.4f})"
                )

            # Do NOT call env.step() here: env.step() internally animates SOFA.
            # The GUI main loop performs the integration step after this callback.
            self.env._do_action(smoothed_action)

        except Exception as e:
            import traceback
            print(f"[InferenceController] Error during inference/action apply: {e}")
            print(traceback.format_exc())
            self._pause("error during action apply")

    def onAnimateEndEvent(self, event):
        """Called after the GUI simulation step; mirrors second half of MCREnv.step()."""
        if self.stopped:
            return

        try:
            observation, reward, terminated, truncated, info = self._training_equivalent_post_step()
            update_target_marker_visual(self.env, force_print=False)
            self.obs = observation
            self.last_reward = float(reward)
            self.last_info = info
            self.step_counter += 1

            if self.step_counter % self.print_every == 0:
                self._print_metrics(info=info, reward=float(reward), terminated=terminated, truncated=truncated)

            current_dist = float(info.get("current_dist_to_goal", np.inf))
            threshold = float(getattr(self.env, "target_distance_threshold", np.inf))

            if self.stop_on_threshold and np.isfinite(current_dist) and current_dist <= threshold:
                self._pause(
                    f"target threshold reached: curD={current_dist * 1000.0:.3f} mm, "
                    f"threshold={threshold * 1000.0:.3f} mm"
                )
                return

            if self.stop_on_train_done and (bool(terminated) or bool(info.get("done_by_target", False))):
                self._pause(
                    f"training done: success={bool(info.get('done_by_target', False))}, "
                    f"reason={info.get('terminal_reason', 'unknown')}"
                )
                return

            if self.max_steps is not None and self.step_counter >= self.max_steps:
                self._pause(f"max steps reached: {self.max_steps}")
                return

            if self.sleep_time > 0:
                time.sleep(self.sleep_time)

        except Exception as e:
            import traceback
            print(f"[InferenceController] Error during post-step update: {e}")
            print(traceback.format_exc())
            self._pause("error during post-step update")

    # ============================================================
    # Training-equivalent step bookkeeping
    # ============================================================
    def _training_equivalent_post_step(self) -> Tuple[Union[np.ndarray, dict], float, bool, bool, dict]:
        """Mirror MCREnv.step() after SOFA integration has happened in GUI."""
        self.env._elapsed_steps = int(getattr(self.env, "_elapsed_steps", 0)) + 1

        # Same order as MCREnv.step(): observation first, then reward.
        reward = self.env._get_reward()
        observation = self.env._get_observation(None)
        terminated = bool(self.env._get_done())
        non_finite_failure = False

        if getattr(self.env, "observation_type", ObservationType.STATE) == ObservationType.STATE:
            if not _obs_all_finite(observation):
                non_finite_failure = True
                observation = _sanitize_observation(observation)

        if not np.isfinite(float(reward)):
            non_finite_failure = True
            reward = -100.0

        if non_finite_failure:
            self.env.non_finite_failure = True
            terminated = True

        # In training, MCREnv can terminate after a long period without passing
        # the next ordered gate. For GUI diagnosis we usually want to keep watching
        # whether the catheter eventually recovers, so ignore that termination by
        # default while still logging noProg.
        ignored_no_progress_failure = bool(getattr(self.env, "no_progress_failure", False)) and bool(
            getattr(self, "ignore_no_progress_done", True)
        )
        if ignored_no_progress_failure:
            self.env.no_progress_failure = False

        if (
            bool(getattr(self.env, "is_out_of_bounds", False))
            or bool(getattr(self.env, "terminal_escape_failed", False))
            or (
                bool(getattr(self.env, "no_progress_failure", False))
                and not bool(getattr(self, "ignore_no_progress_done", True))
            )
            or bool(getattr(self.env, "non_finite_failure", False))
        ):
            terminated = True

        truncated = (int(getattr(self.env, "_elapsed_steps", 0)) >= int(getattr(self.env, "max_episode_steps", 2000))) and (not terminated)
        info = self.env._get_info(terminated=terminated, truncated=truncated)
        if ignored_no_progress_failure:
            info["no_progress_failure_ignored"] = True
            info["done_by_no_progress"] = False
            if str(info.get("terminal_reason", "")) == "no_progress":
                info["terminal_reason"] = "not_done"
        else:
            info["no_progress_failure_ignored"] = False
        if truncated:
            info["TimeLimit.truncated"] = True

        return observation, float(reward), bool(terminated), bool(truncated), info

    # ============================================================
    # Metrics and stopping helpers
    # ============================================================
    def _pause(self, reason: str):
        print(f"\n[InferenceController] {reason}")
        print("=> SOFA animate paused. You can rotate/inspect the final scene in the GUI.")
        self.stopped = True
        try:
            self.env._sofa_root_node.animate.value = False
        except Exception:
            pass

    @staticmethod
    def _yesno(v) -> str:
        return "YES" if bool(v) else "NO"

    def _print_metrics(self, info: Dict[str, Any], reward: float, terminated: bool, truncated: bool):
        cur_d = float(info.get("current_dist_to_goal", np.nan))
        min_d = float(info.get("min_dist_to_goal", np.nan))
        gate_r = float(info.get("gate_progress_ratio", info.get("centerline_progress_ratio", np.nan)))
        gate_idx = int(info.get("gate_idx", -1))
        gate_next = int(info.get("gate_next_idx", -1))
        gate_num = int(info.get("gate_num", 0))
        cl_dist = float(info.get("centerline_distance", np.nan))
        r_local = float(info.get("centerline_local_radius", np.nan))
        safety_ratio = float(info.get("centerline_safety_ratio", np.nan))
        no_progress_counter = int(info.get("no_progress_counter", 0))

        # Current ordered waypoint diagnostics.
        waypoint_idx = int(info.get("waypoint_idx", -1))
        waypoint_num = int(info.get("waypoint_num", 0))
        waypoint_distance = float(info.get("waypoint_distance", np.nan))
        waypoint_is_final = bool(info.get("waypoint_is_final", False))
        waypoint_handoff_counter = int(
            info.get("waypoint_handoff_counter", 0)
        )
        waypoint_handoff = bool(
            info.get("waypoint_handoff_this_step", False)
        )
        waypoint_handoff_count = int(
            info.get("waypoint_handoff_count_episode", 0)
        )

        waypoint_position = np.full(3, np.nan, dtype=np.float32)
        try:
            waypoint_points = getattr(self.env, "waypoint_points", None)
            if waypoint_points is not None and len(waypoint_points) > 0:
                last_idx = int(len(waypoint_points) - 1)
                safe_idx = int(np.clip(waypoint_idx, 0, last_idx))
                if safe_idx >= last_idx:
                    waypoint_position = np.asarray(
                        getattr(self.env, "target_position", waypoint_points[safe_idx]),
                        dtype=np.float32,
                    ).reshape(3)
                else:
                    waypoint_position = np.asarray(
                        waypoint_points[safe_idx],
                        dtype=np.float32,
                    ).reshape(3)
        except Exception:
            pass

        print(
            f"[TRAIN_EQUIV_METRIC] Step={self.step_counter:04d} | "
            f"R={reward:+.3f} term={self._yesno(terminated)} trunc={self._yesno(truncated)} "
            f"reason={info.get('terminal_reason', 'unknown')} | "
            f"curD={cur_d * 1000.0:.2f}mm minD={min_d * 1000.0:.2f}mm | "
            f"wp={waypoint_idx}/{max(waypoint_num - 1, 0)} "
            f"wpFinal={self._yesno(waypoint_is_final)} "
            f"wpD={waypoint_distance * 1000.0:.2f}mm "
            f"wpHandoffCnt={waypoint_handoff_counter} "
            f"wpHandoff={self._yesno(waypoint_handoff)} "
            f"wpHandoffN={waypoint_handoff_count} "
            f"wpPos=({waypoint_position[0]:+.6f},"
            f"{waypoint_position[1]:+.6f},"
            f"{waypoint_position[2]:+.6f}) | "
            f"gate={gate_idx}/{max(gate_num - 1, 0)} next={gate_next} gate_r={gate_r:.3f} "
            f"passed={self._yesno(info.get('gate_passed_this_step', False))} "
            f"noProg={no_progress_counter} "
            f"noProgIgnored={self._yesno(info.get('no_progress_failure_ignored', False))} | "
            f"CL_dist={cl_dist * 1000.0:.2f}mm R_local={r_local * 1000.0:.2f}mm "
            f"ratio={safety_ratio:.3f} outV={self._yesno(info.get('out_of_vessel', False))} | "
            f"rawIns={float(info.get('raw_insert', np.nan)):+.3f} "
            f"effIns={float(info.get('effective_insert', np.nan)):+.3f} "
            f"align={float(info.get('forward_alignment', np.nan)):+.3f} | "
            f"10mm={self._yesno(info.get('success_10mm', False))} "
            f"6mm={self._yesno(info.get('success_6mm', False))} "
            f"3mm={self._yesno(info.get('success_3mm', False))} "
            f"2mm={self._yesno(info.get('success_2mm', False))}"
        )



def _print_centerline_debug(env: MCREnv):
    """Print loaded/resampled centerline endpoints so file replacement can be verified."""
    try:
        pts = getattr(env, "centerline_points", None)
        if pts is None:
            print("[CENTERLINE_DEBUG] env.centerline_points=None")
            return
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            print("[CENTERLINE_DEBUG] invalid centerline_points:", getattr(pts, "shape", None))
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_mm = float(np.sum(seg) * 1000.0)
        y_min_idx = int(np.argmin(pts[:, 1]))
        y_max_idx = int(np.argmax(pts[:, 1]))
        print(
            "[CENTERLINE_DEBUG]",
            f"count={len(pts)}",
            f"length_mm={total_mm:.3f}",
            f"first=({pts[0,0]:+.6f},{pts[0,1]:+.6f},{pts[0,2]:+.6f})",
            f"last=({pts[-1,0]:+.6f},{pts[-1,1]:+.6f},{pts[-1,2]:+.6f})",
            f"ymin_idx={y_min_idx}",
            f"ymin=({pts[y_min_idx,0]:+.6f},{pts[y_min_idx,1]:+.6f},{pts[y_min_idx,2]:+.6f})",
            f"ymax_idx={y_max_idx}",
        )
        try:
            target = _get_current_target_position(env)
            if target is not None:
                print(
                    "[TARGET_DEBUG]",
                    f"target=({target[0]:+.6f},{target[1]:+.6f},{target[2]:+.6f})",
                    f"dist_to_first_mm={float(np.linalg.norm(target-pts[0])*1000.0):.3f}",
                    f"dist_to_last_mm={float(np.linalg.norm(target-pts[-1])*1000.0):.3f}",
                )
        except Exception as e:
            print("[TARGET_DEBUG] failed:", e)
    except Exception as e:
        print("[CENTERLINE_DEBUG] failed:", e)


def _print_scene_debug(env: MCREnv):
    try:
        scr = getattr(env, "scene_creation_result", None)
        print("\n[DEBUG] scene_creation_result keys:", list(scr.keys()) if isinstance(scr, dict) else scr)
        if isinstance(scr, dict):
            print("[DEBUG] chosen_model    =", scr.get("chosen_model"))
            print("[DEBUG] task_id         =", scr.get("task_id"))
            print("[DEBUG] environment_stl =", scr.get("environment_stl"))
            print("[DEBUG] centerline_vtk  =", scr.get("centerline_vtk"))
            print("[DEBUG] target_position =", scr.get("target_position"))
    except Exception as e:
        print("[DEBUG] Could not print scene_creation_result:", e)

    try:
        root = getattr(env, "_sofa_root_node", None)
        if root is not None:
            children = [c.name for c in root.getChildren()]
            print("[DEBUG] root node children:", children)
            has_instrument = any("mcr" in (c.name or "").lower() or "instrument" in (c.name or "").lower() for c in root.getChildren())
            print(f"[DEBUG] instrument present: {has_instrument}")
    except Exception as e:
        print("[DEBUG] Could not inspect root node children:", e)


def _auto_center_camera(env: MCREnv):
    try:
        scr = getattr(env, "scene_creation_result", {}) or {}
        target = _get_current_target_position(env)
        cam = scr.get("camera")
        if target is not None and cam is not None:
            cam_pos = [float(target[0]), float(target[1]) - 0.5, float(target[2]) + 0.5]
            cam_look = [float(target[0]), float(target[1]), float(target[2])]
            try:
                cam.position = cam_pos
                cam.lookAt = cam_look
            except Exception:
                try:
                    cam.findData("position").value = cam_pos
                    cam.findData("lookAt").value = cam_look
                except Exception:
                    pass
            print(f"[DEBUG] Camera repositioned to {cam_pos}, lookAt {cam_look}")
    except Exception as e:
        print("[DEBUG] Could not auto-center camera:", e)


def main():
    parser = argparse.ArgumentParser(description="Test trained SAC model in SOFA GUI with training-equivalent step bookkeeping.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained SAC model zip.")
    parser.add_argument("--env-type", choices=["aortic", "flat"], default="aortic")
    parser.add_argument("--force-model", type=str, default="", help="Specific model to force, e.g. 0207, V1, 0210, 0021. Leave empty for random.")
    parser.add_argument("--target-threshold", type=float, default=0.010, help="Success distance threshold in meters. Default 0.010 for 10 mm testing.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Wait time after each GUI/env step to slow visualization.")
    parser.add_argument("--vessel-alpha", type=float, default=0.35, help="Vessel opacity (0.0~1.0).")
    parser.add_argument("--positioning-camera", action="store_true", help="Force camera setup even if debug rendering is off.")
    parser.add_argument("--print-every", type=int, default=1, help="Print metrics every N GUI/env steps.")
    parser.add_argument("--time-step", type=float, default=0.1, help="SOFA time step in seconds. Use the value used in training when known.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Env frame skip. Default 1, matching current MCREnv default.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max GUI/env steps before pausing. Default 2000 for longer GUI testing.")
    parser.add_argument("--max-episode-steps", type=int, default=2000, help="MCREnv max_episode_steps. Default 2000 for longer GUI testing.")
    parser.add_argument("--continue-after-done", action="store_true", help="Do not pause when training termination condition is reached.")
    parser.add_argument("--continue-after-threshold", action="store_true", help="Do not pause immediately when current distance reaches target threshold.")
    parser.add_argument("--enable-no-progress-termination", action="store_true", help="Use training no-progress termination. Default: disabled for GUI testing.")

    # Training-style start/target/orientation randomization. In forced single-vessel
    # mode, soft_randomize_single_vessel=True keeps the scene loaded and randomizes
    # the target/instrument pose inside MCREnv.reset(), instead of double-randomizing
    # in createScene().
    parser.add_argument("--no-randomize-start-target", action="store_true", help="Disable start/target randomization. Default: enabled.")
    parser.add_argument("--start-target-random-radius", type=float, default=0.002, help="Start/target randomization radius in meters. Default 0.002 = 2 mm.")
    parser.add_argument("--no-randomize-initial-orientation", action="store_true", help="Disable initial orientation cone randomization. Default: enabled.")
    parser.add_argument("--initial-orientation-max-angle-deg", type=float, default=30.0, help="Initial orientation random cone half-angle in degrees. Default 30.")
    parser.add_argument("--entry-tangent-points", type=int, default=5, help="Number of centerline points used to estimate entry tangent. Default 5.")
    parser.add_argument("--disable-soft-randomize-single-vessel", action="store_true", help="Disable soft single-vessel random reset; normally keep off.")

    # Collision arguments are retained for compatibility with older shell scripts,
    # but they are ignored unless --override-collision-params is explicitly passed.
    parser.add_argument("--override-collision-params", action="store_true", help="Patch LocalMinDistance after reset. Debug only; off by default to match training.")
    parser.add_argument("--contact-distance", type=float, default=0.0008, help="Debug-only LocalMinDistance contactDistance in meters.")
    parser.add_argument("--alarm-distance", type=float, default=0.0016, help="Debug-only LocalMinDistance alarmDistance in meters.")
    parser.add_argument("--vessel-collision-proximity", type=float, default=None, help="Optional vessel collision proximity passed to scene creation. If omitted, use scene/training default.")
    parser.add_argument("--use-vessel-line-point-collision", action="store_true", help="Enable vessel LineCollisionModel/PointCollisionModel if the scene supports it.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading Model: {model_path}")
    model = SAC.load(str(model_path), device="auto")

    env_type = EnvType.AORTIC if args.env_type == "aortic" else EnvType.FLAT

    create_scene_kwargs = {
        "debug_rendering": True,
        "vessel_alpha": float(args.vessel_alpha),
        "positioning_camera": bool(args.positioning_camera),
        "use_vessel_line_point_collision": bool(args.use_vessel_line_point_collision),

        "randomize_start_target": not bool(args.no_randomize_start_target),
        "start_target_random_radius": float(args.start_target_random_radius),
        "randomize_initial_orientation": not bool(args.no_randomize_initial_orientation),
        "initial_orientation_max_angle_deg": float(args.initial_orientation_max_angle_deg),
        "entry_tangent_points": int(args.entry_tangent_points),
        "soft_randomize_single_vessel": not bool(args.disable_soft_randomize_single_vessel),
    }
    if args.vessel_collision_proximity is not None:
        create_scene_kwargs["vessel_collision_proximity"] = float(args.vessel_collision_proximity)
    if args.force_model:
        create_scene_kwargs["force_model"] = args.force_model
        print(f"Forcing model to: {args.force_model}")

    print(
        "[RANDOMIZATION]",
        f"randomize_start_target={create_scene_kwargs['randomize_start_target']}",
        f"radius_mm={create_scene_kwargs['start_target_random_radius'] * 1000.0:.3f}",
        f"randomize_initial_orientation={create_scene_kwargs['randomize_initial_orientation']}",
        f"max_angle_deg={create_scene_kwargs['initial_orientation_max_angle_deg']:.3f}",
        f"soft_randomize_single_vessel={create_scene_kwargs['soft_randomize_single_vessel']}",
    )
    print(
        "[NO_PROGRESS_TERMINATION]",
        "enabled=" + str(bool(args.enable_no_progress_termination)),
        "(default is disabled in GUI test)"
    )

    print("Initializing SOFA Environment...")
    env = MCREnv(
        create_scene_kwargs=create_scene_kwargs,
        env_type=env_type,
        time_step=float(args.time_step),
        frame_skip=int(args.frame_skip),
        target_distance_threshold=float(args.target_threshold),
        settle_steps=8,
        max_episode_steps=int(args.max_episode_steps),
        # RenderMode.NONE prevents pyglet/pygame conflicts with Sofa.Gui.
        render_mode=RenderMode.NONE,
    )
    initial_obs, _ = env.reset()
    update_target_marker_visual(env, force_print=True)

    print("[COLLISION_PARAM] No LocalMinDistance patch by default; using scene/training parameters.")
    if args.override_collision_params:
        print(
            "[COLLISION_PARAM] override requested:",
            "contact=", float(args.contact_distance),
            "alarm=", float(args.alarm_distance),
        )
        apply_local_min_distance_params(
            root_node=getattr(env, "_sofa_root_node", None),
            contact_distance=float(args.contact_distance),
            alarm_distance=float(args.alarm_distance),
        )

    _print_scene_debug(env)
    _print_centerline_debug(env)
    _auto_center_camera(env)

    controller = InferenceController(
        env=env,
        model=model,
        initial_obs=initial_obs,
        sleep_time=float(args.sleep),
        print_every=int(args.print_every),
        max_steps=int(args.max_steps),
        stop_on_train_done=not bool(args.continue_after_done),
        stop_on_threshold=not bool(args.continue_after_threshold),
        ignore_no_progress_done=not bool(args.enable_no_progress_termination),
    )
    env._sofa_root_node.addObject(controller)
    controller.init()

    print("Starting SOFA GUI...")
    Sofa.Gui.GUIManager.Init("mcr_test_gui_train_equiv", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(env._sofa_root_node, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(env._sofa_root_node)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == "__main__":
    main()
