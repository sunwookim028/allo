#!/usr/bin/env bash
set -euo pipefail

: "${construct_path:?Missing parameter: construct_path}"
: "${allo_design_file:?Missing parameter: allo_design_file}"
: "${allo_entrypoint:=build}"
: "${backend:=vitis}"
: "${build_mode:=csyn}"
: "${clock_period:=10.0}"
: "${backend_options:={\"device\":\"u280\"}}"
: "${python_bin:=python}"
: "${allo_setup_script:=/work/shared/common/allo/setup-llvm-main.sh}"
: "${backend_module:=}"

if [ ! -f "$allo_setup_script" ]; then
  echo "ERROR: Allo LLVM setup script does not exist: $allo_setup_script" >&2
  exit 2
fi
# The setup modifies this shell's LLVM library/tool environment, so it must be
# sourced rather than executed as a child process.
# Remove inherited commercial-tool compiler/runtime paths first. In particular,
# Synopsys ships a libstdc++.so.6 that is too old for Allo's MLIR extension.
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset GCC_EXEC_PREFIX
unset LD_PRELOAD
set +u

if [ -n "$backend_module" ]; then
  if ! type module >/dev/null 2>&1; then
    source /usr/share/Modules/init/bash
  fi
  case "$backend" in
    vitis)
      module load "$backend_module"
      ;;
    *)
      echo "ERROR: unsupported backend '$backend'; currently supported: vitis" >&2
      exit 2
      ;;
  esac
fi

source "$allo_setup_script"
set -u

"$python_bin" preflight.py \
  --python-bin "$python_bin" \
  --design "$allo_design_file" \
  --construct-path "$construct_path" \
  --backend "$backend" \
  --mode "$build_mode" \
  --setup-script "$allo_setup_script"

if [[ "$allo_design_file" = /* ]]; then
  design_path="$allo_design_file"
else
  construct_dir="$(cd "$(dirname "$construct_path")" && pwd)"
  design_path="$construct_dir/$allo_design_file"
fi

if [ ! -f "$design_path" ]; then
  echo "ERROR: Allo design does not exist: $design_path" >&2
  exit 2
fi

command -v "$python_bin" >/dev/null
"$python_bin" -c 'import allo; print(allo.__file__)'

if [ "$backend" = "vitis" ]; then
  command -v vitis_hls >/dev/null
else
  echo "ERROR: unsupported backend '$backend'; currently supported: vitis" >&2
  exit 2
fi

rm -rf work
mkdir -p work/project outputs

set +e
"$python_bin" run_design.py \
  --design "$design_path" \
  --entrypoint "$allo_entrypoint" \
  --project work/project \
  --backend "$backend" \
  --mode "$build_mode" \
  --clock-period "$clock_period" \
  --backend-options "$backend_options" \
  > >(tee outputs/allo-build.log) \
  2> >(tee -a outputs/allo-build.log >&2)
build_status=$?
set -e

if [ "$build_status" -ne 0 ]; then
  echo "ERROR: Allo backend build exited with status $build_status" \
    | tee -a outputs/allo-build.log
  exit "$build_status"
fi

"$python_bin" validate_build.py \
  --project work/project \
  --backend "$backend" \
  --rtl-output outputs/backend-rtl

cp -a work/project outputs/allo-build
cp -a work/project/asic-debug outputs/asic-debug
for name in asic-manifest.json asic-manifest.tcl \
            asic-manifest-final.json asic-manifest-final.tcl; do
  cp "work/project/$name" "outputs/$name"
done

export ALLO_NODE_DESIGN_PATH="$design_path"
export ALLO_NODE_BACKEND="$backend"
export ALLO_NODE_BUILD_MODE="$build_mode"
export ALLO_NODE_CLOCK_PERIOD="$clock_period"
export ALLO_NODE_BACKEND_OPTIONS="$backend_options"
"$python_bin" -c '
import os
from pathlib import Path
from backend import parse_backend_options
metadata = {
    "design": str(Path(os.environ["ALLO_NODE_DESIGN_PATH"]).resolve()),
    "backend": os.environ["ALLO_NODE_BACKEND"],
    "mode": os.environ["ALLO_NODE_BUILD_MODE"],
    "clock_period_ns": float(os.environ["ALLO_NODE_CLOCK_PERIOD"]),
    "backend_options": parse_backend_options(
        os.environ["ALLO_NODE_BACKEND_OPTIONS"]
    ),
}
Path("outputs/build-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
'
touch outputs/build-success
