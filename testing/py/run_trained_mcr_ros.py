#!/usr/bin/env python3
"""
run_trained_mcr_ros.py

统一推理入口：加载训练好的 SAC 模型，创建 ROS 版 MCREnv，循环运行闭环控制。

这里不重新实现 PyBullet / MoveIt / 磁场执行逻辑，只保持标准 Gym 推理方式：
    obs -> model.predict(obs, deterministic=True) -> env.step(action)

严格同步闭环由 mcr_sim_ros/mcr_rl_env_ros.py 内部完成：
    action -> B_target -> /magnetic/execute_request
    PyBullet executor -> /magnetic/execute_result -> B_actual
    B_actual -> SOFA animate -> obs_{t+1}
"""

import argparse
import sys
import time
from pathlib import Path

# This executable lives in python/testing/py/.  Resolve local packages from
# the script location rather than relying on cwd or a fixed PYTHONPATH.
TEST_PY_DIR = Path(__file__).resolve().parent
TESTING_DIR = TEST_PY_DIR.parent
PYTHON_ROOT = TESTING_DIR.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np
import rospy
from stable_baselines3 import SAC

from mcr_sim_ros.mcr_rl_env_ros import MCREnv, ObservationType, ActionType, EnvType
from mcr_sim_ros.rl_core_ros.base_ros import RenderMode, RenderFramework


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained SAC policy with the ROS strict magnetic closed loop."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained SAC .zip model.",
    )
    parser.add_argument(
        "--env-type",
        choices=["aortic", "flat"],
        default="aortic",
        help="Environment type. Default: aortic.",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.002,
        help="Success threshold in meters. Default: 0.002 for 2 mm.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2048,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.1,
        help="SOFA time step.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="SOFA frame skip.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=8,
        help="SOFA settle steps after reset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="SB3 device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--render",
        choices=["headless", "human"],
        default="headless",
        help="SOFA render mode.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy output. Default is deterministic.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds after each step.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Maximum episodes to run. 0 means run forever.",
    )
    parser.add_argument(
        "--force-model",
        type=str,
        default="",
        help="Force one vessel model if mcr scene supports this option. Empty means default/random.",
    )

    return parser.parse_args()


def build_env(args):
    env_type = EnvType.AORTIC if args.env_type == "aortic" else EnvType.FLAT
    render_mode = RenderMode.REMOTE if args.render == "human" else RenderMode.NONE

    create_scene_kwargs = {}
    if args.force_model:
        create_scene_kwargs["force_model"] = args.force_model

    if args.render == "human":
        create_scene_kwargs["debug_rendering"] = True
        create_scene_kwargs["positioning_camera"] = True
        create_scene_kwargs["vessel_alpha"] = 0.25
    env = MCREnv(
        env_type=env_type,
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        settle_steps=args.settle_steps,
        render_mode=render_mode,
        render_framework=RenderFramework.PYGLET,
        target_distance_threshold=args.target_threshold,
        max_episode_steps=args.max_episode_steps,
        create_scene_kwargs=create_scene_kwargs,
    )
    return env


def fmt_bool(value):
    return "YES" if bool(value) else "NO"


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def main():
    args = parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PYTHON_ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"SAC model not found: {model_path}")

    # mcr_rl_env_ros.py 内部也会初始化 ROS。这里先初始化，避免匿名节点重复和日志混乱。
    if not rospy.core.is_initialized():
        rospy.init_node("run_trained_mcr_ros", anonymous=True, disable_signals=True)

    deterministic = not args.stochastic

    print("=" * 100)
    print("Run trained SAC model with ROS strict magnetic closed loop")
    print(f"Model:             {model_path}")
    print(f"Env type:          {args.env_type}")
    print(f"Target threshold:  {args.target_threshold * 1000.0:.1f} mm")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Device:            {args.device}")
    print(f"Policy mode:       {'deterministic' if deterministic else 'stochastic'}")
    print(f"Force model:       {args.force_model if args.force_model else 'default/random scene setting'}")
    print("=" * 100)

    env = build_env(args)
    model = SAC.load(str(model_path), env=None, device=args.device)

    obs, _ = env.reset()
    episode_idx = 1
    step_in_episode = 0
    global_step = 0
    episode_reward = 0.0

    try:
        while not rospy.is_shutdown():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            step_in_episode += 1
            global_step += 1
            episode_reward += float(reward)

            min_dist = safe_float(info.get("min_dist_to_goal", np.nan))
            current_dist = safe_float(info.get("current_dist_to_goal", np.nan))
            rel_error = safe_float(info.get("magnetic_executor_rel_error", np.nan))
            moved = bool(info.get("magnetic_executor_moved", False))
            executor_success = bool(info.get("magnetic_executor_success", False))
            success_10mm = bool(info.get("success_10mm", False))
            success_6mm = bool(info.get("success_6mm", False))
            success_3mm = bool(info.get("success_3mm", False))
            success_2mm = bool(info.get("success_2mm", False))
            i_l = safe_float(info.get("magnetic_executor_I_L", np.nan))
            i_r = safe_float(info.get("magnetic_executor_I_R", np.nan))
            centerline_delta = safe_float(info.get("centerline_delta_progress", np.nan))
            centerline_ratio = safe_float(info.get("centerline_progress_ratio", np.nan))
            centerline_dist = safe_float(info.get("centerline_distance", np.nan))
            using_cl = bool(info.get("using_centerline_reward", False))
            using_terminal = bool(info.get("using_euclidean_terminal_reward", False))

            print(
                f"Ep={episode_idx} Step={step_in_episode} Global={global_step} | "
                f"R={float(reward):.3f} SumR={episode_reward:.3f} | "
                f"curD={current_dist * 1000.0:.2f}mm "
                f"minD={min_dist * 1000.0:.2f}mm | "
                f"CL_d={centerline_delta * 1000.0:.3f}mm "
                f"CL_r={centerline_ratio:.3f} "
                f"CL_dist={centerline_dist * 1000.0:.2f}mm "
                f"UseCL={fmt_bool(using_cl)} "
                f"UseTerm={fmt_bool(using_terminal)} | "
                f"10mm={fmt_bool(success_10mm)} 6mm={fmt_bool(success_6mm)} "
                f"3mm={fmt_bool(success_3mm)} 2mm={fmt_bool(success_2mm)} | "
                f"Exec={fmt_bool(executor_success)} RelErr={rel_error * 100.0:.2f}% "
                f"Moved={fmt_bool(moved)} I=[{i_l:.3f},{i_r:.3f}]"
            )

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print("-" * 100)
                print(
                    f"Episode {episode_idx} finished by {reason}: "
                    f"steps={step_in_episode}, return={episode_reward:.3f}, "
                    f"min_dist={min_dist * 1000.0:.2f}mm, "
                    f"success_3mm={fmt_bool(success_3mm)}, success_2mm={fmt_bool(success_2mm)}"
                )
                print("-" * 100)

                if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                    break

                obs, _ = env.reset()
                episode_idx += 1
                step_in_episode = 0
                episode_reward = 0.0

            if args.sleep > 0.0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        print("Closed MCREnv.")


if __name__ == "__main__":
    main()
