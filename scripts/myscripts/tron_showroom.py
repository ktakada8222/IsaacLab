# Copyright (c) 2024, Tron KK
# SPDX-License-Identifier: BSD-3-Clause

"""
Tron Showroom: 4 つのロボット (G1, Oli, TRON1, BOOSTER K1) を 1 つのシーンに並べて表示するデモ。

使い方:
    ./isaaclab.sh -p scripts/demo/tron_showroom.py
"""

import argparse
from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. 引数の処理
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Tron Showroom: display all robot models in one scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ----------------------------------------------------------------------
# 2. シミュレーションアプリを起動
# ----------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 3. アプリ起動後に Isaac Lab モジュールを import
# ----------------------------------------------------------------------
import os
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaacsim.core.utils.prims as prim_utils


# ----------------------------------------------------------------------
# 4. ロボットのアセットパスを定義
# ----------------------------------------------------------------------
# ローカルのアセットフォルダ (ホームディレクトリ基準)
ASSETS_ROOT = os.path.expanduser("~/takada_ws/IsaacLab/assets/robot_assets/tron_showroom")

ROBOT_USD_PATHS = {
    #"G1":     f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
    #"G1":    f"{ASSETS_ROOT}/unitree_model/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd",
    "Oli":    f"{ASSETS_ROOT}/oli/HU_D04_description/usd/HU_D04_01_colored.usd",
    #"TRON1":  f"{ASSETS_ROOT}/tron1/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/PF_TRON1A/PF_TRON1A.usd",
    #"K1":   f"{ASSETS_ROOT}/booster_k1/robots/K1/usd/K1_colored.usd",
}

# 各ロボットを並べる位置 (x 軸方向に 1m 間隔)
ROBOT_POSITIONS = {
    "G1":    (-1.5, 0.0, 0.0),
    "Oli":   (-0.5, 0.0, 0.0),
    "TRON1": ( 0.5, 0.0, 0.0),
    "K1":    ( 1.5, 0.0, 0.0),
}


# ----------------------------------------------------------------------
# 5. シーン設計
# ----------------------------------------------------------------------
def design_scene():
    """ショールーム用のシーンを構築する。"""

    # 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 平行光源
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light.func("/World/Light", cfg_light, translation=(0.0, 0.0, 10.0))

    # ロボットをまとめる親 Xform
    prim_utils.create_prim("/World/Robots", "Xform")

    # 各ロボットを USD ファイルから読み込んで配置する
    for name, usd_path in ROBOT_USD_PATHS.items():
        position = ROBOT_POSITIONS[name]
        prim_path = f"/World/Robots/{name}"

        cfg_robot = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg_robot.func(prim_path, cfg_robot, translation=position)

        print(f"[INFO] Spawned {name:6s} at {position}  (path={usd_path})")


# ----------------------------------------------------------------------
# 6. メイン処理
# ----------------------------------------------------------------------
def main():
    # 物理設定
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # カメラ位置 (ロボット列を斜め前から見下ろす)
    sim.set_camera_view(eye=[3.0, -4.0, 2.5], target=[0.0, 0.0, 0.8])

    # シーンを構築
    design_scene()

    # 物理エンジンに登録
    sim.reset()
    print("[INFO] Tron Showroom ready. Press Ctrl+C to exit.")

    # シミュレーションループ
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()