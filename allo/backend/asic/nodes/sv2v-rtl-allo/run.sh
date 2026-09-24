#!/usr/bin/env bash
set -euo pipefail

: "${top_module:=top}"
: "${backend:=vitis}"
: "${sv2v_bin:=sv2v}"
: "${sv2v_defines:=}"
: "${sv2v_include_dirs:=}"

mkdir -p outputs/normalized-rtl
cp -a inputs/backend-rtl outputs/source-rtl

files=()
case "$backend" in
  vitis)
    while IFS= read -r file; do
      files+=("$file")
    done < <(find -H inputs/backend-rtl -type f \( -name '*.v' -o -name '*.sv' \) | sort)
    ;;
  catapult|systemc)
    files+=(inputs/backend-rtl/concat_rtl.v)
    ;;
  *)
    echo "ERROR: unsupported RTL normalization backend: $backend" \
      | tee outputs/sv2v.log
    exit 2
    ;;
esac

if [ "${#files[@]}" -eq 0 ]; then
  echo "ERROR: inputs/backend-rtl contains no Verilog" | tee outputs/sv2v.log
  exit 1
fi

printf '%s\n' "${files[@]}" > outputs/source-manifest.f

if [[ "$backend" =~ ^(catapult|systemc)$ && "${#files[@]}" -ne 1 ]]; then
  echo "ERROR: Catapult/SystemC normalization must consume only concat_rtl.v" \
    | tee outputs/sv2v.log
  exit 1
fi

args=()
for define in $sv2v_defines; do
  args+=("-D" "$define")
done
IFS=':' read -r -a include_dirs <<< "$sv2v_include_dirs"
for incdir in "${include_dirs[@]}"; do
  [ -z "$incdir" ] || args+=("-I" "$incdir")
done

{
  echo "Running: $sv2v_bin ${args[*]} -w outputs/design.v"
  printf '  %s\n' "${files[@]}"
  "$sv2v_bin" "${args[@]}" -w outputs/design.v "${files[@]}"
} 2>&1 | tee outputs/sv2v.log

if [ ! -s outputs/design.v ]; then
  echo "ERROR: sv2v produced an empty design.v" | tee -a outputs/sv2v.log
  exit 1
fi

perl -0pi -e 's/\b(parameter|localparam)\s+string\s+/$1 /g' outputs/design.v
grep -Eq "(^|[[:space:]])module[[:space:]]+$top_module([[:space:]#(]|$)" \
  outputs/design.v || {
    echo "ERROR: normalized RTL does not contain top module $top_module" \
      | tee -a outputs/sv2v.log
    exit 1
  }

cp outputs/design.v outputs/normalized-rtl/design.v
for name in asic-manifest.json asic-manifest.tcl \
            asic-manifest-final.json asic-manifest-final.tcl \
            build-metadata.json; do
  cp "inputs/$name" "outputs/$name"
done
