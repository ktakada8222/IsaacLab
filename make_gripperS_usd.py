#!/usr/bin/env python3
"""
Create Vega-1P / Vega-1U URDF variants with DexGripper S, then optionally convert them to USD.

Run from the root of https://github.com/dexmate-ai/dexmate-urdf

Typical Isaac Sim / Isaac Lab usage:
  /workspace/isaaclab/_isaac_sim/python.sh make_gripperS_usd.py --convert-usd

URDF-only usage:
  python make_gripperS_usd.py --urdf-only
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

VARIANTS = ("vega_1p", "vega_1u")
SIDES = ("L", "R")

REPO_HINT = "Run this script from the root of the dexmate-urdf repository."


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists():
        die(f"Required file not found: {path}\n{REPO_HINT}")


def find_child(parent: ET.Element, tag: str, **attrs: str) -> ET.Element | None:
    for child in list(parent):
        if child.tag != tag:
            continue
        if all(child.get(k) == v for k, v in attrs.items()):
            return child
    return None


def remove_named(root: ET.Element, tag: str, name: str) -> None:
    for child in list(root):
        if child.tag == tag and child.get("name") == name:
            root.remove(child)


def mesh_filename_to_dexs_path(filename: str) -> str:
    # dm_gripper.urdf stores meshes as "meshes/..." relative to robots/hands/dexs_gripper/.
    # The generated Vega URDFs live in robots/humanoid/vega_1p or robots/humanoid/vega_1u,
    # so DexGripper S paths must go up two levels and then into hands/dexs_gripper.
    if filename.startswith("meshes/"):
        return "../../hands/dexs_gripper/" + filename
    return filename


def rewrite_mesh_paths(element: ET.Element) -> None:
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", mesh_filename_to_dexs_path(filename))


def prefix_gripper_element(element: ET.Element, side: str) -> ET.Element:
    """Return a deep-copied DexGripper S element with L_/R_ prefixes applied."""
    new = copy.deepcopy(element)

    if "name" in new.attrib:
        new.set("name", f"{side}_{new.get('name')}")

    for node in new.iter():
        if node.tag in {"parent", "child"} and node.get("link"):
            node.set("link", f"{side}_{node.get('link')}")
        if node.tag == "mimic" and node.get("joint"):
            node.set("joint", f"{side}_{node.get('joint')}")

    rewrite_mesh_paths(new)
    return new


def get_connector_visual(original_root: ET.Element, side: str) -> ET.Element | None:
    """Preserve the connector.glb visual from the original DexGripper D base link."""
    base = find_child(original_root, "link", name=f"{side}_gripper_base")
    if base is None:
        return None
    for visual in base.findall("visual"):
        for mesh in visual.iter("mesh"):
            filename = mesh.get("filename", "")
            if "connector.glb" in filename:
                return copy.deepcopy(visual)
    return None


def build_side_s_gripper(dm_root: ET.Element, original_root: ET.Element, side: str) -> list[ET.Element]:
    elements: list[ET.Element] = []

    # Link order in the source dm_gripper.urdf is base, l1, j1, l2, j2.
    # We preserve that order, and add the Vega connector visual to the base link.
    for src in list(dm_root):
        if src.tag not in {"link", "joint"}:
            continue
        new = prefix_gripper_element(src, side)
        if src.tag == "link" and src.get("name") == "gripper_base":
            connector_visual = get_connector_visual(original_root, side)
            if connector_visual is not None:
                # Add connector as the first visual so the arm-gripper interface remains visible.
                insert_index = 0
                # Keep inertial first if present.
                if len(new) and new[0].tag == "inertial":
                    insert_index = 1
                new.insert(insert_index, connector_visual)
        elements.append(new)
    return elements


def replace_dexd_with_dexs(repo_root: Path, variant: str) -> Path:
    humanoid_dir = repo_root / "robots" / "humanoid" / variant
    src_urdf = humanoid_dir / f"{variant}_gripper.urdf"
    out_urdf = humanoid_dir / f"{variant}_gripperS.urdf"
    dm_urdf = repo_root / "robots" / "hands" / "dexs_gripper" / "dm_gripper.urdf"

    require_file(src_urdf)
    require_file(dm_urdf)

    src_tree = ET.parse(src_urdf)
    src_root = src_tree.getroot()
    dm_root = ET.parse(dm_urdf).getroot()

    src_root.set("name", f"{variant}_gripperS")

    # Remove the original embedded DexGripper D links/joints, but keep L_hand_mount/R_hand_mount.
    for side in SIDES:
        for suffix in ("gripper_base", "gripper_l1", "gripper_l2"):
            remove_named(src_root, "link", f"{side}_{suffix}")
        for suffix in ("gripper_j1", "gripper_j2"):
            remove_named(src_root, "joint", f"{side}_{suffix}")

    # Append DexGripper S blocks for both hands.
    for side in SIDES:
        for element in build_side_s_gripper(dm_root, src_root, side):
            src_root.append(element)

    ET.indent(src_tree, space="  ")
    src_tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf


def validate_urdf_paths(repo_root: Path, urdf_path: Path) -> list[str]:
    missing: list[str] = []
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        # Skip URI-style paths; the current repo uses relative paths, but this keeps the validator generic.
        if "://" in filename or filename.startswith("package://"):
            continue
        mesh_path = (urdf_dir / filename).resolve()
        if not mesh_path.exists():
            missing.append(str(mesh_path.relative_to(repo_root) if repo_root in mesh_path.parents else mesh_path))
    return missing


def convert_to_usd(repo_root: Path, urdf_path: Path, output_dir: Path, python_exe: str) -> Path:
    # Reuse the official Dexmate workflow pieces. This requires Isaac Sim / Isaac Lab Python.
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.workflows.decomp_urdf_collision_meshes import decompose_urdf
    except Exception as exc:
        die(f"Could not import Dexmate decomposition workflow: {exc}")

    convert_script = repo_root / "scripts" / "workflows" / "convert_urdf.py"
    require_file(convert_script)

    decomposed_urdf_path = Path(decompose_urdf(urdf_path))
    robot_name = urdf_path.stem
    usd_dir = output_dir / robot_name
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd_path = usd_dir / f"{robot_name}.usd"

    cmd = [
        python_exe,
        str(convert_script),
        str(decomposed_urdf_path),
        str(usd_path),
        "--fix-base",
        "--headless",
    ]
    print("Converting:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root))

    zip_path = output_dir / f"{robot_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in usd_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(usd_dir))
    print(f"Created {zip_path}")
    return usd_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="dexmate-urdf repository root")
    parser.add_argument("--output-dir", type=Path, default=Path("output_gripperS"), help="USD output directory")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for convert_urdf.py")
    parser.add_argument("--urdf-only", action="store_true", help="Only create URDFs; do not convert to USD")
    parser.add_argument("--convert-usd", action="store_true", help="Convert generated URDFs to USD")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    require_file(repo_root / "scripts" / "workflows" / "convert_urdf.py")
    require_file(repo_root / "robots" / "hands" / "dexs_gripper" / "dm_gripper.urdf")

    generated: list[Path] = []
    for variant in VARIANTS:
        out_urdf = replace_dexd_with_dexs(repo_root, variant)
        generated.append(out_urdf)
        missing = validate_urdf_paths(repo_root, out_urdf)
        if missing:
            print(f"WARNING: {out_urdf.relative_to(repo_root)} has missing mesh references:")
            for item in missing:
                print(f"  - {item}")
        else:
            print(f"Created and validated: {out_urdf.relative_to(repo_root)}")

    should_convert = args.convert_usd and not args.urdf_only
    if should_convert:
        output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for urdf_path in generated:
            usd_path = convert_to_usd(repo_root, urdf_path.resolve(), output_dir, args.python_exe)
            print(f"USD: {usd_path}")
    else:
        print("URDF generation complete. Re-run with --convert-usd using Isaac Sim / Isaac Lab Python to create USDs.")


if __name__ == "__main__":
    main()
