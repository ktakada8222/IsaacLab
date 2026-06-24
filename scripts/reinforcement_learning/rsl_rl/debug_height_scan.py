# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the height-scan sensor values in an environment (sanity check for USD terrain).

It creates the env, steps a few times with zero actions, and prints statistics of the
height-scan readings the policy would observe. Useful to confirm the scanner returns sane
values on a scanned/USD mesh (vs NaN/inf from missed rays, or values saturating the clip).

Run:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/debug_height_scan.py \
        --task Isaac-Velocity-Boiler-G1-Play-v0 --num_envs 4 --steps 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug the height-scan sensor values.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (must have a height_scanner).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=20, help="Number of steps to log.")
parser.add_argument("--offset", type=float, default=0.5, help="Height-scan offset (matches mdp.height_scan).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from pxr import Usd

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def dump_meshes(root_path: str):
    """Report mesh prims under `root_path`, with and without descending into instances."""
    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[dump] root prim '{root_path}' is INVALID (wrong path).")
        return

    def count_meshes(traverse_instances: bool):
        if traverse_instances:
            rng = Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        else:
            rng = Usd.PrimRange(root)
        paths = [str(p.GetPath()) for p in rng if p.GetTypeName() == "Mesh"]
        return paths

    normal = count_meshes(False)
    with_instances = count_meshes(True)
    print(f"[dump] Mesh prims under '{root_path}':")
    print(f"       normal traversal          : {len(normal)}")
    print(f"       with instance proxies      : {len(with_instances)}")
    instanceable = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsInstanceable() or p.IsInstance()]
    print(f"       instanceable/instance prims: {len(instanceable)}")
    for p in instanceable[:5]:
        print(f"         - {p}")
    for p in with_instances[:5]:
        print(f"       example mesh: {p}")


def stats(name: str, t: torch.Tensor):
    finite = torch.isfinite(t)
    n_total = t.numel()
    n_bad = int((~finite).sum().item())
    vals = t[finite]
    if vals.numel() == 0:
        print(f"  {name}: ALL non-finite ({n_bad}/{n_total})  <-- broken")
        return
    print(
        f"  {name}: min={vals.min().item():+.3f} max={vals.max().item():+.3f} "
        f"mean={vals.mean().item():+.3f} std={vals.std().item():.3f} | "
        f"non-finite={n_bad}/{n_total} | |v|>1 (clipped)={int((vals.abs() > 1.0).sum().item())}/{vals.numel()}"
    )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # report the mesh structure the ray-caster has to work with
    dump_meshes("/World/ground")

    scene = env.unwrapped.scene
    if "height_scanner" not in scene.sensors:
        print("[ERROR] This task has no 'height_scanner' sensor. Use a task that keeps height_scan.")
        env.close()
        return

    sensor = scene.sensors["height_scanner"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)

    env.reset()
    for i in range(args_cli.steps):
        env.step(zero_actions)
        # height = sensor_z - hit_z - offset  (same formula as mdp.height_scan)
        hit_z = sensor.data.ray_hits_w[..., 2]
        sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
        height = sensor_z - hit_z - args_cli.offset

        print(f"[step {i:3d}]")
        stats("ray_hit_z   ", hit_z)
        stats("height_scan ", height)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
