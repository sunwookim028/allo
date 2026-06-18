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
: "${python_bin:=python}"
: "${openram_script:=}"
: "${use_sram_cache:=False}"
: "${sram_cache_path:=}"

is_true() {
  case "$1" in
    True|true|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "$use_sram_cache"; then
  "$python_bin" scripts/copy_sram_cache.py \
    --manifest "$manifest_path" \
    --cache-path "$cache_path" \
    --out-dir outputs/srams

  cp "$manifest_path" outputs/srams/sram_manifest.yml
  exit 0
fi

if [ -z "$openram_script" ]; then
  openram_script="$("$python_bin" - <<'PY'
import openram, pathlib
print(pathlib.Path(openram.__file__).resolve().parent / "sram_compiler.py")
PY
)"
fi

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

"$python_bin" scripts/gen_openram_cfgs.py \
  --manifest "$manifest_path" \
  --template templates/openram_cfg.py.in \
  --out-dir work/cfgs \
  --tech-name "$tech_name" \
  --process-corner "$process_corner" \
  --supply-voltage "$supply_voltage" \
  --temperature "$temperature" \
  --check-lvsdrc "$check_lvsdrc" \
  --route-supplies "$route_supplies" \
  --analytical-delay "$analytical_delay"

cp "$manifest_path" outputs/srams/sram_manifest.yml

shopt -s nullglob
cfgs=(work/cfgs/*_cfg.py)

if [ "${#cfgs[@]}" -eq 0 ]; then
  echo "ERROR: No OpenRAM cfg files generated in work/cfgs"
  exit 1
fi

for cfg in "${cfgs[@]}"; do
  name="$(basename "$cfg" _cfg.py)"
  outdir="outputs/srams/$name"
  mkdir -p "$outdir"

  echo "--- Generating SRAM: $name ---"
  rm -rf "work/$name"
  mkdir -p "work/$name"

  openram_args=(-v -v)

  if [ "$analytical_delay" != "True" ] && [ "$analytical_delay" != "true" ]; then
    openram_args+=("-c")
  fi

  (
    cd work

    touch "$name/$name.log"
    tail -n +1 -f "$name/$name.log" &
    tail_pid=$!

    "$python_bin" "$openram_script" "${openram_args[@]}" "cfgs/${name}_cfg.py"
    status=$?

    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    exit "$status"
  ) 2>&1 | tee "$outdir/$name.openram.log"

  cp "$cfg" "$outdir/"
  cp "work/$name"/*.v "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.lef "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.gds "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.lib "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.sp  "$outdir/" 2>/dev/null || true
  cp "work/$name"/*.cdl "$outdir/" 2>/dev/null || true

  lib_files=("$outdir"/*.lib)
  if [ "${#lib_files[@]}" -eq 0 ]; then
    echo "ERROR: OpenRAM did not produce a .lib for $name"
    exit 1
  fi

  lib_file="${lib_files[0]}"

  lc_shell -x "read_lib $lib_file; write_lib ${name}_lib -format db -output $outdir/$name.db; exit" \
    2>&1 | tee "$outdir/$name.lc_shell.log"

  test -f "$outdir/$name.db"
done

