# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Booster Robotics robots.

The following configurations are available:

* :obj:`BOOSTER_K1_CFG`: Booster K1 humanoid robot (22 DOF)

Reference: https://github.com/BoosterRobotics/booster_gym
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the robot_assets directory (relative to this file: ../../../../robot_assets)
_ROBOT_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../robot_assets"))

##
# Configuration
##

BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            _ROBOT_ASSETS_DIR, "tron_showroom/booster_k1/robots/K1/usd/K1_22dof.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*_Hip_Pitch": -0.20,
            ".*_Knee_Pitch": 0.40,
            ".*_Ankle_Pitch": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_Hip_Yaw": 60.0,
                ".*_Hip_Roll": 80.0,
                ".*_Hip_Pitch": 80.0,
                ".*_Knee_Pitch": 100.0,
            },
            damping={
                ".*_Hip_Yaw": 2.0,
                ".*_Hip_Roll": 3.0,
                ".*_Hip_Pitch": 3.0,
                ".*_Knee_Pitch": 3.0,
            },
            armature={
                ".*_Hip_.*": 0.01,
                ".*_Knee_Pitch": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim=50,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim=40,
            stiffness=20.0,
            damping=3.0,
        ),
    },
)
"""Configuration for the Booster Robotics K1 humanoid robot (22 DOF).

Joint layout (revolute joints only):
- Head:  AAHead_yaw, Head_pitch  (merged as fixed in locomotion URDF)
- Arms:  ALeft/Right_Shoulder_Pitch, Left/Right_Shoulder_Roll, Left/Right_Elbow_Pitch, Left/Right_Elbow_Yaw
- Legs:  Left/Right_Hip_Pitch, Left/Right_Hip_Roll, Left/Right_Hip_Yaw,
         Left/Right_Knee_Pitch, Left/Right_Ankle_Pitch, Left/Right_Ankle_Roll

Effort limits (Nm): Hip_Pitch=30, Hip_Roll=35, Hip_Yaw=20, Knee=40, Ankle=20
USD asset: robot_assets/tron_showroom/booster_k1/robots/K1/usd/K1_22dof.usd
"""
