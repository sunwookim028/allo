#! /usr/bin/env bash
#=========================================================================
# run.sh
#=========================================================================
# Author : Julian Bushlow
# Date   : June 17, 2026
#

set -euo pipefail

mkdir -p outputs/srams work/logs

: "${construct_path:?Missing parameter: construct_path}"
: "${sram_manifest:?Missing parameter: sram_manifest}"
: "${openram_bin:=openram}"

if [[ "$sram_manifest" = /* ]]; then
  manifest_path="$sram_manifest"
else
  construct_dir="$(cd "$(dirname "$construct_path")" && pwd)"
  manifest_path="$construct_dir/$sram_manifest"
fi

if [ ! -f "$manifest_path" ]; then
  echo "ERROR: SRAM manifest does not exist: $manifest_path"
  exit 1
fi

python scripts/gen_openram_cfgs.py \
  --manifest "$manifest_path" \
  --template templates/openram_cfg.py.in \
  --out-dir work/cfgs \
  --tech-name "$tech_name" \
  --process-corner "$process_corner" \
  --supply-voltage "$supply_voltage" \
  --temperature "$temperature" \
  --check-lvsdrc "$check_lvsdrc" \
  --route-supplies "$route_supplies"

cp "$manifest_path" outputs/srams/sram_manifest.yml

shopt -s nullglob
cfgs=(work/cfgs/*_cfg.py)

if [ "${#cfgs[@]}" -eq 0 ]; then
  echo "ERROR: No OpenRAM cfg files generated in work/cfgs"
  exit 1
fi

for cfg in work/cfgs/*_cfg.py; do
  name="$(basename "$cfg" _cfg.py)"
  outdir="outputs/srams/$name"
  mkdir -p "$outdir"

  echo "--- Generating SRAM: $name ---"
  rm -rf "work/$name"
  mkdir -p "work/$name"

  (
    cd work
    "$openram_bin" -v -v "cfgs/${name}_cfg.py"
  ) 2>&1 | tee "$outdir/$name.openram.log"

  cp "$cfg" "$outdir/"
  cp "work/$name"/*.v "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.lef "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.gds "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.lib "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.sp  "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.cdl "$outdir/" 2>/dev/null || true

  lib_file="$(ls "$outdir"/*.lib | head -n 1)"
  if [ -z "$lib_file" ]; then
    echo "ERROR: OpenRAM did not produce a .lib for $name"
    exit 1
  fi

  lc_shell -x "read_lib $lib_file; write_lib ${name}_lib -format db -output $outdir/$name.db; exit" \
    2>&1 | tee "$outdir/$name.lc_shell.log"

  test -f "$outdir/$name.db"
done

