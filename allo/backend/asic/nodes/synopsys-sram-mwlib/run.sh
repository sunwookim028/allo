#! /usr/bin/env bash
#=========================================================================
# run.sh
#=========================================================================
# Create Synopsys Milkyway physical-reference libraries for generated SRAMs.
#

set -euo pipefail

mkdir -p outputs
rm -rf outputs/srams
cp -R inputs/srams outputs/srams

: "${mwlib_tool:=auto}"
: "${mwlib_tech_file:=rtk-tech.tf}"

tech_file="inputs/adk/$mwlib_tech_file"

if [ ! -f "$tech_file" ]; then
  echo "ERROR: Milkyway technology file does not exist: $tech_file"
  exit 1
fi

shopt -s nullglob
lef_files=(outputs/srams/*/*.lef)

if [ "${#lef_files[@]}" -eq 0 ]; then
  echo "ERROR: No SRAM LEF files found under outputs/srams"
  exit 1
fi

run_with_lef2mw() {
  local lef="$1"
  local mwlib="$2"

  lef2mw \
    -tech "$tech_file" \
    -lef "$lef" \
    -mwlib "$mwlib"
}

run_with_dc_shell() {
  local lef="$1"
  local mwlib="$2"

  dc_shell-xg-t -topographical_mode \
    -x "set sram_lef \"$lef\"; set sram_mwlib \"$mwlib\"; set sram_tech_file \"$tech_file\"; source scripts/create-sram-mwlib.tcl"
}

for lef in "${lef_files[@]}"; do
  sram_dir="$(dirname "$lef")"
  name="$(basename "$lef" .lef)"
  mwlib="$sram_dir/$name.mwlib"

  echo "--- Creating SRAM Milkyway library for $name ---"
  echo "LEF:   $lef"
  echo "MWLIB: $mwlib"

  rm -rf "$mwlib"

  case "$mwlib_tool" in
    auto)
      if command -v lef2mw >/dev/null 2>&1; then
        run_with_lef2mw "$lef" "$mwlib"
      else
        run_with_dc_shell "$lef" "$mwlib"
      fi
      ;;
    lef2mw)
      run_with_lef2mw "$lef" "$mwlib"
      ;;
    dc_shell|dc_shell-xg-t)
      run_with_dc_shell "$lef" "$mwlib"
      ;;
    *)
      echo "ERROR: Unknown mwlib_tool '$mwlib_tool'. Use auto, lef2mw, or dc_shell."
      exit 1
      ;;
  esac

  if [ ! -d "$mwlib" ]; then
    echo "ERROR: Expected Milkyway library was not created: $mwlib"
    exit 1
  fi
done

missing=0

for db in outputs/srams/*/*.db; do
  sram_dir="$(dirname "$db")"
  name="$(basename "$db" .db)"

  if [ ! -d "$sram_dir/$name.mwlib" ]; then
    echo "ERROR: Found $db but missing $sram_dir/$name.mwlib"
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  exit 1
fi
