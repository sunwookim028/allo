#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path
import yaml

REQUIRED_GLOBS = ["*.v", "*.lib", "*.db", "*.lef"]

OPTIONAL_GLOBS = ["*.gds", "*.gds.gz", "*.sp", "*.cdl"]

def has_any(path, patterns):
    return any(path.glob(p) for p in patterns)

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--cache-path", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

manifest = yaml.safe_load(Path(args.manifest).read_text())
cache = Path(args.cache_path).resolve()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

for sram in manifest.get("srams", []):
    name = sram["name"]
    src = cache / name
    dst = out / name

    if not src.is_dir():
        raise SystemExit(f"ERROR: Missing cached SRAM directory: {src}")

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    for pattern in REQUIRED_GLOBS:
        if not list(dst.glob(pattern)):
            raise SystemExit(f"ERROR: Cached SRAM {name} missing required {pattern}")

    if not has_any(dst, ["*.sp", "*.cdl"]):
        print(f"WARNING: Cached SRAM {name} has no SPICE/CDL view")

print(f"Copied cached SRAMs from {cache} to {out}")