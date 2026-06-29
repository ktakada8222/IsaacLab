#!/usr/bin/env python3
"""
Safer Vega + DexGripper S generator.

Compared with the first helper, this version does NOT remove and append gripper links.
It keeps the original Vega URDF element order and only patches the existing
L_/R_ DexGripper D blocks in-place using values from DexGripper S.

Run from dexmate-urdf root:
  python make_gripperS_usd_v2.py --urdf-only

Convert with Isaac Sim / Isaac Lab Python:
  /path/to/isaac/python.sh make_gripperS_usd_v2.py --convert-usd --python-exe /path/to/isaac/python.sh

Debug-friendly conversion, usually recommended first:
  /path/to/isaac/python.sh make_gripperS_usd_v2.py --convert-usd --skip-decompose --merge-joints --python-exe /path/to/isaac/python.sh
"""

from __future__ import annotations

import argparse
import copy
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


def find_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    for child in root:
        if child.tag == tag and child.get("name") == name:
            return child
    return None


def mesh_filename_to_dexs_path(filename: str) -> str:
    # dm_gripper.urdf paths are relative to robots/hands/dexs_gripper/.
    # The generated Vega URDFs are in robots/humanoid/vega_1p or vega_1u.
    if filename.startswith("meshes/"):
        return "../../hands/dexs_gripper/" + filename
    return filename


def rewrite_mesh_paths(element: ET.Element) -> None:
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", mesh_filename_to_dexs_path(filename))


def prefix_refs(element: ET.Element, side: str) -> None:
    # Only references inside joints/mimic need side prefixes. The target element
    # name itself is handled separately so we can keep the original Vega names.
    for node in element.iter():
        if node.tag in {"parent", "child"} and node.get("link"):
            node.set("link", f"{side}_{node.get('link')}")
        if node.tag == "mimic" and node.get("joint"):
            node.set("joint", f"{side}_{node.get('joint')}")


def clone_children(src: ET.Element) -> list[ET.Element]:
    return [copy.deepcopy(child) for child in list(src)]


def clear_children(dst: ET.Element) -> None:
    for child in list(dst):
        dst.remove(child)


def get_connector_visual(link: ET.Element) -> ET.Element | None:
    for visual in link.findall("visual"):
        for mesh in visual.iter("mesh"):
            if "connector.glb" in mesh.get("filename", ""):
                return copy.deepcopy(visual)
    return None


def replace_link_contents(dst_link: ET.Element, src_link: ET.Element) -> None:
    original_name = dst_link.get("name")
    connector_visual = get_connector_visual(dst_link) if original_name and original_name.endswith("gripper_base") else None

    clear_children(dst_link)
    for child in clone_children(src_link):
        rewrite_mesh_paths(child)
        dst_link.append(child)

    # Preserve Vega connector visual on gripper_base links.
    if connector_visual is not None:
        insert_index = 0
        if len(dst_link) and dst_link[0].tag == "inertial":
            insert_index = 1
        dst_link.insert(insert_index, connector_visual)


def replace_joint_contents(dst_joint: ET.Element, src_joint: ET.Element, side: str) -> None:
    original_name = dst_joint.get("name")
    original_type = dst_joint.get("type")

    clear_children(dst_joint)
    for child in clone_children(src_joint):
        prefix_refs(child, side)
        rewrite_mesh_paths(child)
        dst_joint.append(child)

    if original_name:
        dst_joint.set("name", original_name)
    if original_type:
        dst_joint.set("type", original_type)


def patch_variant(repo_root: Path, variant: str) -> Path:
    humanoid_dir = repo_root / "robots" / "humanoid" / variant
    src_urdf = humanoid_dir / f"{variant}_gripper.urdf"
    out_urdf = humanoid_dir / f"{variant}_gripperS.urdf"
    dm_urdf = repo_root / "robots" / "hands" / "dexs_gripper" / "dm_gripper.urdf"

    require_file(src_urdf)
    require_file(dm_urdf)

    tree = ET.parse(src_urdf)
    root = tree.getroot()
    root.set("name", f"{variant}_gripperS")

    dm_root = ET.parse(dm_urdf).getroot()
    dm_links = {el.get("name"): el for el in dm_root.findall("link")}
    dm_joints = {el.get("name"): el for el in dm_root.findall("joint")}

    for side in SIDES:
        for suffix in ("gripper_base", "gripper_l1", "gripper_l2"):
            dst = find_named(root, "link", f"{side}_{suffix}")
            src = dm_links.get(suffix)
            if dst is None or src is None:
                die(f"Could not patch {variant}: missing link {side}_{suffix} or source {suffix}")
            replace_link_contents(dst, src)

        for suffix in ("gripper_j1", "gripper_j2"):
            dst = find_named(root, "joint", f"{side}_{suffix}")
            src = dm_joints.get(suffix)
            if dst is None or src is None:
                die(f"Could not patch {variant}: missing joint {side}_{suffix} or source {suffix}")
            replace_joint_contents(dst, src, side)

    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf


def validate_mesh_paths(repo_root: Path, urdf_path: Path) -> list[str]:
    missing: list[str] = []
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or "://" in filename or filename.startswith("package://"):
            continue
        mesh_path = (urdf_dir / filename).resolve()
        if not mesh_path.exists():
            try:
                missing.append(str(mesh_path.relative_to(repo_root)))
            except ValueError:
                missing.append(str(mesh_path))
    return missing


def validate_connectivity(urdf_path: Path) -> tuple[list[str], int, int]:
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    child_links = set()
    joint_count = 0
    for joint in root.findall("joint"):
        joint_count += 1
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_links.add(child.get("link"))
    roots = sorted(links - child_links)
    return roots, len(links), joint_count


def get_decomposed_urdf(repo_root: Path, urdf_path: Path, do_decompose: bool) -> Path:
    if not do_decompose:
        return urdf_path
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.workflows.decomp_urdf_collision_meshes import decompose_urdf
    except Exception as exc:
        die(f"Could not import decomposition workflow: {exc}")
    return Path(decompose_urdf(urdf_path))


def convert_to_usd(
    repo_root: Path,
    urdf_path: Path,
    output_dir: Path,
    python_exe: str,
    *,
    merge_joints: bool,
    decompose_collision: bool,
) -> Path:
    convert_script = repo_root / "scripts" / "workflows" / "convert_urdf.py"
    require_file(convert_script)

    input_urdf = get_decomposed_urdf(repo_root, urdf_path, decompose_collision)
    robot_name = urdf_path.stem
    usd_dir = output_dir / robot_name
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd_path = usd_dir / f"{robot_name}.usd"

    cmd = [
        python_exe,
        str(convert_script),
        str(input_urdf.resolve()),
        str(usd_path.resolve()),
        "--fix-base",
        "--headless",
    ]
    if merge_joints:
        cmd.append("--merge-joints")

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
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("output_gripperS_v2"))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--urdf-only", action="store_true")
    parser.add_argument("--convert-usd", action="store_true")
    parser.add_argument("--merge-joints", action="store_true", help="Pass --merge-joints to Dexmate/Isaac converter")
    parser.add_argument("--skip-decompose", action="store_true", help="Do not run collision mesh decomposition before USD conversion")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    require_file(repo_root / "robots" / "hands" / "dexs_gripper" / "dm_gripper.urdf")
    require_file(repo_root / "scripts" / "workflows" / "convert_urdf.py")

    generated: list[Path] = []
    for variant in VARIANTS:
        out_urdf = patch_variant(repo_root, variant)
        generated.append(out_urdf)
        missing = validate_mesh_paths(repo_root, out_urdf)
        roots, link_count, joint_count = validate_connectivity(out_urdf)
        print(f"Created: {out_urdf.relative_to(repo_root)}")
        print(f"  connectivity: links={link_count}, joints={joint_count}, roots={roots}")
        if len(roots) != 1:
            print("  WARNING: URDF has multiple roots. This can cause floating/exploded links.")
        if missing:
            print("  WARNING: missing mesh files:")
            for item in missing:
                print(f"    - {item}")
        else:
            print("  mesh paths: OK")

    if args.convert_usd and not args.urdf_only:
        output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for urdf_path in generated:
            usd_path = convert_to_usd(
                repo_root,
                urdf_path.resolve(),
                output_dir.resolve(),
                args.python_exe,
                merge_joints=args.merge_joints,
                decompose_collision=not args.skip_decompose,
            )
            print(f"USD: {usd_path}")
    else:
        print("URDF generation complete. Use --convert-usd with Isaac Sim / Isaac Lab Python to create USDs.")


if __name__ == "__main__":
    main()
