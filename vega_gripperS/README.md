# Vega + DexGripper S — custom URDF → USD

## Summary
- **No official Gripper S USD exists.** The dexmate-urdf releases (latest v0.8.3) only ship
  `vega_1p_gripper` / `vega_1u_gripper` (DexGripper **D**). There is no `gripperS` asset.
- These two URDFs were created by taking the official Vega gripper URDFs and replacing the
  DexGripper **D** portion with DexGripper **S** (`robots/hands/dexs_gripper/dm_gripper.urdf`).

## Key finding: D and S share the same attachment frame
DexGripper D (`dexd_gripper.urdf`) and DexGripper S (`dm_gripper.urdf`) have **identical kinematics**:
same link names (`gripper_base`/`l1`/`l2`), joints (`gripper_j1`/`j2`), axes, mimic relation, and the
same `gripper_base` root frame. So **the wrist mount did not need to change.**

What was changed vs. the official Vega gripper URDFs:
| Item | D (original) | S (new) |
|------|-------------|---------|
| mesh folder | `hands/dexd_gripper/` | `hands/dexs_gripper/` |
| link inertials (base/l1/l2) | D values (base 0.393 kg, finger 0.0997 kg) | S values (base 0.373 kg, finger 0.0535 kg) |
| finger joint origins | `0.0258 / 0.02875` | `0.02581 / 0.0288` (sub-0.1 mm) |

Preserved unchanged: wrist fixed mount `L/R_hand_mount` (`origin xyz="0 0 0.078" rpy="0 0 0"`,
parent `L/R_arm_l8`), the f5d6 wrist `connector.glb` visual, and the Vega finger limit `upper=0.7854`
(the standalone gripper allows 0.96; 0.7854 is the Vega-platform value — change it if you want full range).

## Validation (passed, see report)
Single root (`base` / `base_link`), no duplicate or disconnected links, mimic joints intact,
all mesh references resolve (vega_1p: 75 meshes, vega_1u: 58 meshes, 0 missing).

## How to convert to USD (run on the Ubuntu machine with Isaac Sim + IsaacLab)
The URDFs use **relative** mesh paths (`../../hands/dexs_gripper/...`, `../vega_1/meshes/...`), so they
**must sit inside a dexmate-urdf clone** at their original locations:

```bash
# 1. on the Ubuntu box, in your dexmate-urdf clone:
cp vega_1p_gripperS.urdf robots/humanoid/vega_1p/
cp vega_1u_gripperS.urdf robots/humanoid/vega_1u/

# 2a. Easiest — the official workflow auto-discovers every *.urdf in the target dirs
#     (it convex-decomposes collisions, converts with --fix-base --headless, and zips each output):
python scripts/workflows/usd_workflow.py --output_dir output --python_exe <isaaclab python>

# 2b. Or convert a single file directly with IsaacLab's converter:
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripperS.urdf \
    /path/to/output/vega_1p_gripperS/vega_1p_gripperS.usd \
    --merge-joints --fix-base --headless
# repeat for vega_1u_gripperS.urdf
```

Both converters set `convert_mimic_joints_to_normal_joints=True`, so the gripper mimic becomes a
normal driven joint in USD (drive S finger joints together in your controller).

Outputs land in `output/<name>/<name>.usd` (+ a zipped folder from workflow 2a).
