# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standardize the transform ops of a USD so the ray-caster reads them correctly.

The boiler scan stores its placement on an Xform using a matrix / unit-resolve xformOp,
which MultiMeshRayCaster's `resolve_prim_pose` cannot read -> the ray-cast collision mesh
ends up at the raw (wrong) location while visual/physics are correct. This rewrites every
Xformable prim's transform stack into the canonical [translate, orient, scale] form
(world pose preserved, units baked), and writes a new USD. Physics/material APIs are kept.

Run:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/standardize_boiler_usd.py \
        --input  robot_assets/tron_showroom/environments/boiler_pose_fixed.usd \
        --output robot_assets/tron_showroom/environments/boiler_std.usd
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standardize a USD's xform ops for ray-casting.")
parser.add_argument("--input", type=str, required=True, help="Path to the source USD.")
parser.add_argument("--output", type=str, required=True, help="Path to write the standardized USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# run headless by default
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils


def main():
    stage = Usd.Stage.Open(args_cli.input)
    if stage is None:
        raise FileNotFoundError(f"Could not open USD: {args_cli.input}")

    n_ok, n_skip = 0, 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            ok = sim_utils.standardize_xform_ops(prim)
            if ok:
                n_ok += 1
            else:
                n_skip += 1

    print(f"[standardize] standardized {n_ok} xformable prims (skipped {n_skip}).")

    # export the (composed) stage to the new file -- keeps geometry + physics/material APIs
    stage.Export(args_cli.output)
    print(f"[standardize] wrote: {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
