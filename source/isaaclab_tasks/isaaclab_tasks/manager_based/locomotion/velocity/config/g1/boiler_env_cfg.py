# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg

_ROBOT_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../..", "robot_assets"))

BOILER_USD_PATH = os.path.join(_ROBOT_ASSETS_DIR, "tron_showroom/environments/boiler_pose_fixed.usd")


@configclass
class G1BoilerEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Replace generated rough terrain with the Boiler USD environment
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = BOILER_USD_PATH
        self.scene.terrain.terrain_generator = None

        # No terrain difficulty curriculum for a fixed USD environment
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # The env grid is generated centered on (0, 0). Shift every env's spawn so the
        # grid is centered on the boiler room floor instead, with the robot starting
        # slightly above the floor.
        # NOTE: tune `num_envs` / `env_spacing` to match the boiler room's floor size.
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.5
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.85)
        # match the rigid-body physics material used during rough-terrain training
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class G1BoilerEnvCfg_PLAY(G1BoilerEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1BoilerSoloEnvCfg(G1RoughEnvCfg):
    """One boiler room per environment, replicated N times (one robot per room).

    Unlike :class:`G1BoilerEnvCfg` (which packs many robots into a single shared room
    imported as the terrain), this places one copy of the boiler room under each
    environment's namespace so InteractiveScene clones it per env. Each robot then has
    its own private room and spawns at the same spot inside it.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Global ground: a flat plane. Each room sits above it; the robot stands on
        #     the room's own floor. The plane is just a safe base / fall-catcher. ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # --- One boiler room per environment ---
        # prim_path under {ENV_REGEX_NS} => cloned once per env by InteractiveScene.
        # The room's collision meshes + baked-in physics material travel with the USD.
        # boiler_pose_fixed.usd is authored upright/Z-up with an identity default prim,
        # so referencing it with the default identity init_state already places it
        # correctly (the orientation correction lives on a child Xform inside the USD).
        self.scene.boiler = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Boiler",
            spawn=sim_utils.UsdFileCfg(usd_path=BOILER_USD_PATH),
        )

        # --- Blind policy: the single-mesh height scanner cannot see per-env rooms,
        #     so drop the height scan (proprioception-only, like the flat env). ---
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- One robot per room ---
        # Start modest; each room is a full detailed mesh, so VRAM scales with num_envs.
        self.scene.num_envs = 32
        # IMPORTANT: env_spacing must exceed the boiler room's footprint (x/y extent) so
        # neighbouring room copies do not overlap. Measure the room bbox and tune this.
        self.scene.env_spacing = 40.0
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.00)

        # physics material for the plane (matches rough-training friction)
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class G1BoilerSoloEnvCfg_PLAY(G1BoilerSoloEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 4
        self.episode_length_s = 40.0

        # fixed, slow, straight-line command
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # deterministic spawn (no random yaw/xy) for debugging
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
