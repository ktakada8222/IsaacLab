# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

from .rough_env_cfg import BoosterK1RoughEnvCfg

_ROBOT_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../..", "robot_assets"))

BOILER_USD_PATH = os.path.join(_ROBOT_ASSETS_DIR, "tron_showroom/environments/Boiler.usd")


@configclass
class BoosterK1BoilerEnvCfg(BoosterK1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Replace generated rough terrain with the Boiler USD environment
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = BOILER_USD_PATH
        self.scene.terrain.terrain_generator = None

        # No terrain difficulty curriculum for a fixed USD environment
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None


@configclass
class BoosterK1BoilerEnvCfg_PLAY(BoosterK1BoilerEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
