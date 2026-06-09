#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


CANONICAL_ODB_RE = re.compile(r"^\d+_[A-Za-z][A-Za-z0-9_]*\.odb$")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def human_size(num_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0

    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass

    return total


def is_canonical_odb(path: Path) -> bool:
    """
    Keep canonical ORFS stage checkpoints like:
      2_floorplan.odb
      3_place.odb
      4_cts.odb
      5_route.odb
      6_final.odb

    Delete substep/intermediate checkpoints like:
      2_1_floorplan.odb
      3_4_place_resized.odb
      5_2_fillcell.odb
    """
    return CANONICAL_ODB_RE.match(path.name) is not None


def prune_intermediate_odbs(flow_dir: Path, platform: str, design: str, variant: str, dry_run: bool) -> None:
    results_dir = flow_dir / "results" / platform / design / variant

    if not results_dir.exists():
        print(f"[orfs-checkpoint] Results directory does not exist: {results_dir}")
        return

    before = dir_size(results_dir)
    print(f"[orfs-checkpoint] Results directory: {results_dir}")
    print(f"[orfs-checkpoint] Size before pruning: {human_size(before)}")

    odbs = sorted(results_dir.glob("*.odb"))

    keep = []
    delete = []

    for odb in odbs:
        if is_canonical_odb(odb):
            keep.append(odb)
        else:
            delete.append(odb)

    print("[orfs-checkpoint] Keeping canonical ODBs:")
    for p in keep:
        print(f"  keep   {p.name} ({human_size(file_size(p))})")

    print("[orfs-checkpoint] Deleting intermediate ODBs:")
    deleted_bytes = 0
    for p in delete:
        sz = file_size(p)
        deleted_bytes += sz
        print(f"  delete {p.name} ({human_size(sz)})")
        if not dry_run:
            p.unlink(missing_ok=True)

    after = dir_size(results_dir)
    print(f"[orfs-checkpoint] Deleted approximately: {human_size(deleted_bytes)}")
    print(f"[orfs-checkpoint] Size after pruning: {human_size(after)}")

def prune_static_orfs_dirs(flow_dir: Path, platform: str, design: str, dry_run: bool) -> None:
    """
    Remove bulky static ORFS collateral from the checkpoint while keeping the
    current design's config and RTL. This assumes later nodes restore by
    overlaying onto a fresh Docker image's /OpenROAD-flow-scripts/flow.
    """
    keep_design_platform = flow_dir / "designs" / platform / design
    keep_design_src = flow_dir / "designs" / "src" / design

    tmp_keep = flow_dir / ".checkpoint_keep"
    tmp_platform_design = tmp_keep / "designs" / platform / design
    tmp_src_design = tmp_keep / "designs" / "src" / design

    print("[orfs-checkpoint] Pruning static ORFS designs/platforms")
    print(f"[orfs-checkpoint] Keeping design config: {keep_design_platform}")
    print(f"[orfs-checkpoint] Keeping design RTL:    {keep_design_src}")

    if dry_run:
        print("[orfs-checkpoint] Dry run: not pruning static ORFS directories")
        return

    if tmp_keep.exists():
        shutil.rmtree(tmp_keep)

    if keep_design_platform.exists():
        tmp_platform_design.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(keep_design_platform, tmp_platform_design, symlinks=True)
    else:
        print(f"[orfs-checkpoint] WARNING: missing expected design config dir: {keep_design_platform}")

    if keep_design_src.exists():
        tmp_src_design.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(keep_design_src, tmp_src_design, symlinks=True)
    else:
        print(f"[orfs-checkpoint] WARNING: missing expected design RTL dir: {keep_design_src}")

    for dirname in ["designs", "platforms"]:
        path = flow_dir / dirname
        if path.exists():
            print(f"[orfs-checkpoint] Removing {path} ({human_size(dir_size(path))})")
            shutil.rmtree(path)

    if (tmp_keep / "designs").exists():
        shutil.copytree(tmp_keep / "designs", flow_dir / "designs", symlinks=True)

    shutil.rmtree(tmp_keep, ignore_errors=True)

def create_checkpoint(flow_dir: Path, checkpoint: Path, dry_run: bool) -> None:
    if not flow_dir.exists():
        raise RuntimeError(f"Cannot checkpoint missing flow directory: {flow_dir}")

    tmp_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")

    print(f"[orfs-checkpoint] Creating checkpoint: {checkpoint}")
    print(f"[orfs-checkpoint] Flow directory size: {human_size(dir_size(flow_dir))}")

    if dry_run:
        print("[orfs-checkpoint] Dry run: not creating checkpoint")
        return

    if tmp_checkpoint.exists():
        tmp_checkpoint.unlink()

    # Use tar instead of Python tarfile for speed and behavior closer to shell flow.
    subprocess.run(
        ["tar", "czf", str(tmp_checkpoint), str(flow_dir)],
        check=True,
    )

    if not tmp_checkpoint.exists() or tmp_checkpoint.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint creation failed or produced empty file: {tmp_checkpoint}")

    tmp_checkpoint.replace(checkpoint)

    print(f"[orfs-checkpoint] Checkpoint size: {human_size(checkpoint.stat().st_size)}")


def delete_flow_dir(flow_dir: Path, dry_run: bool) -> None:
    if not flow_dir.exists():
        print(f"[orfs-checkpoint] Flow directory already absent: {flow_dir}")
        return

    print(f"[orfs-checkpoint] Deleting expanded flow directory: {flow_dir}")
    print(f"[orfs-checkpoint] Reclaiming approximately: {human_size(dir_size(flow_dir))}")

    if dry_run:
        print("[orfs-checkpoint] Dry run: not deleting flow directory")
        return

    shutil.rmtree(flow_dir)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--flow", default="flow")
    parser.add_argument("--checkpoint", default="flow-checkpoint.tar.gz")

    parser.add_argument("--platform", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--variant", default="base")

    parser.add_argument("--prune", default="0")
    parser.add_argument("--delete-flow", default="0")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-static", default="0")

    args = parser.parse_args()

    flow_dir = Path(args.flow)
    checkpoint = Path(args.checkpoint)

    if truthy(args.prune):
        prune_intermediate_odbs(
            flow_dir=flow_dir,
            platform=args.platform,
            design=args.design,
            variant=args.variant,
            dry_run=args.dry_run,
        )
    else:
        print(f"[orfs-checkpoint] Skipping ODB pruning because --prune={args.prune}")

    if truthy(args.prune_static):
        prune_static_orfs_dirs(
            flow_dir=flow_dir,
            platform=args.platform,
            design=args.design,
            dry_run=args.dry_run,
        )
    else:
        print(f"[orfs-checkpoint] Skipping static ORFS pruning because --prune-static={args.prune_static}")
    
    create_checkpoint(
        flow_dir=flow_dir,
        checkpoint=checkpoint,
        dry_run=args.dry_run,
    )

    if truthy(args.delete_flow):
        delete_flow_dir(flow_dir=flow_dir, dry_run=args.dry_run)
    else:
        print(f"[orfs-checkpoint] Keeping expanded flow directory because --delete-flow={args.delete_flow}")

    return 0


if __name__ == "__main__":
    sys.exit(main())