# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard tele-operation of a trained RSL-RL locomotion policy via ROS 2 (humble).

A trained velocity-tracking policy walks the robot; the operator supplies the velocity
command through a ROS 2 ``/cmd_vel`` (geometry_msgs/Twist) topic. The Twist is written
into the environment's ``base_velocity`` command buffer every step, replacing the random
command generator.

Run (terminal 1, this script):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/teleop_boiler_ros.py \
        --task Isaac-Velocity-BoilerSolo-G1-Teleop-v0 \
        --checkpoint logs/rsl_rl/g1_flat/<run>/model_<N>.pt --real-time

Drive (terminal 2, after ``source /opt/ros/humble/setup.bash``):
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="ROS 2 keyboard teleop of an RSL-RL locomotion policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS 2 topic to read the velocity command.")
# command clamps (keep within the policy's trained command range)
parser.add_argument("--max_vx", type=float, default=1.0, help="Max forward velocity command [m/s].")
parser.add_argument("--min_vx", type=float, default=0.0, help="Min forward velocity command [m/s] (0 = no backward).")
parser.add_argument("--max_vy", type=float, default=0.5, help="Max |lateral| velocity command [m/s].")
parser.add_argument("--max_wz", type=float, default=1.0, help="Max |yaw-rate| command [rad/s].")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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

# ROS 2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class CmdVelSubscriber(Node):
    """Stores the most recent /cmd_vel Twist message."""

    def __init__(self, topic: str):
        super().__init__("isaaclab_teleop_cmd_vel")
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._sub = self.create_subscription(Twist, topic, self._callback, 10)
        self.get_logger().info(f"Listening for velocity commands on '{topic}'.")

    def _callback(self, msg: Twist):
        self.vx = float(msg.linear.x)
        self.vy = float(msg.linear.y)
        self.wz = float(msg.angular.z)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Teleoperate an RSL-RL locomotion policy with ROS 2 /cmd_vel."""
    # override configurations with non-hydra CLI arguments
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

    # grab the velocity-command term so we can overwrite it from ROS
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    device = env.unwrapped.device

    # ROS 2 setup
    rclpy.init()
    cmd_node = CmdVelSubscriber(args_cli.cmd_vel_topic)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    try:
        while simulation_app.is_running():
            start_time = time.time()

            # pull the latest /cmd_vel (non-blocking)
            rclpy.spin_once(cmd_node, timeout_sec=0.0)

            # clamp to the policy's trained command range
            vx = max(args_cli.min_vx, min(args_cli.max_vx, cmd_node.vx))
            vy = max(-args_cli.max_vy, min(args_cli.max_vy, cmd_node.vy))
            wz = max(-args_cli.max_wz, min(args_cli.max_wz, cmd_node.wz))

            # write into the command buffer (body frame: [lin_x, lin_y, ang_z])
            command_term.vel_command_b[:, 0] = vx
            command_term.vel_command_b[:, 1] = vy
            command_term.vel_command_b[:, 2] = wz

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cmd_node.destroy_node()
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
