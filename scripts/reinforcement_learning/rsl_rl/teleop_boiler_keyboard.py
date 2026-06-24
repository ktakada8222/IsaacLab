# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard tele-operation of a trained RSL-RL locomotion policy (no ROS required).

A trained velocity-tracking policy walks the robot; the operator supplies the velocity
command with the keyboard via Isaac Lab's built-in ``Se2Keyboard`` device. The command is
written into the environment's ``base_velocity`` buffer every step.

Run (GUI must be visible -- do NOT use --headless; click the viewport to give it focus):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/teleop_boiler_keyboard.py \
        --task Isaac-Velocity-BoilerSolo-G1-Teleop-v0 \
        --checkpoint logs/rsl_rl/g1_flat/<run>/model_<N>.pt --real-time

Keys:  UP/DOWN = forward/back,  LEFT/RIGHT = strafe,  Z/X = turn left/right.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Keyboard teleop of an RSL-RL locomotion policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# keyboard sensitivities (per-key velocity magnitude); keep within the trained command range
parser.add_argument("--v_x", type=float, default=0.8, help="Forward velocity per key press [m/s].")
parser.add_argument("--v_y", type=float, default=0.4, help="Lateral velocity per key press [m/s].")
parser.add_argument("--w_z", type=float, default=1.0, help="Yaw rate per key press [rad/s].")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Teleoperate an RSL-RL locomotion policy with the keyboard."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # create environment and wrap for rsl-rl
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load policy
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # velocity-command term we overwrite from the keyboard
    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    # keyboard device (SE(2): [v_x, v_y, omega_z])
    keyboard = Se2Keyboard(
        Se2KeyboardCfg(v_x_sensitivity=args_cli.v_x, v_y_sensitivity=args_cli.v_y, omega_z_sensitivity=args_cli.w_z)
    )
    keyboard.reset()
    print("[INFO]: Keyboard teleop ready. Click the viewport, then use:")
    print("        UP/DOWN = forward/back,  LEFT/RIGHT = strafe,  Z/X = turn")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()

        # read keyboard -> [v_x, v_y, omega_z] (already scaled by the sensitivities)
        cmd = keyboard.advance().to(env.unwrapped.device)
        command_term.vel_command_b[:, :3] = cmd

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
