#! /usr/bin/env bash
#=========================================================================
# run.sh
#=========================================================================
# Author : Julian Bushlow
# Date   : June 16, 2026
#

set -euo pipefail

mkdir -p outputs

: "${design_path:?Missing parameter: design_path}"
: "${manifest:?Missing parameter: manifest}"
: "${top_module:=${design_name:?Missing parameter: top_module or design_name}}"
: "${sv2v_bin:=sv2v}"
: "${sv2v_defines:=TARGET_ASIC=1}"
: "${sv2v_include_dirs:=.}"
: "${construct_path:?Missing parameter: construct_path}"

if [ ! -d "$design_path" ]; then
  echo "ERROR: design_path does not exist: $design_path" | tee outputs/sv2v.log
  exit 1
fi

if [[ "$manifest" = /* ]]; then
  manifest_path="$manifest"
else
  construct_dir="$(cd "$(dirname "$construct_path")" && pwd)"
  manifest_path="$construct_dir/$manifest"
fi

if [ ! -f "$manifest_path" ]; then
  echo "ERROR: manifest does not exist: $manifest_path" | tee outputs/sv2v.log
  exit 1
fi

files=()

while IFS= read -r line || [ -n "$line" ]; do
  # Strip comments and surrounding whitespace.
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"

  [ -z "$line" ] && continue

  if [[ "$line" = /* ]]; then
    file="$line"
  else
    file="$design_path/$line"
  fi

  if [ ! -f "$file" ]; then
    echo "ERROR: manifest entry does not exist: $file" | tee outputs/sv2v.log
    exit 1
  fi

  files+=("$file")
done < "$manifest_path"

if [ "${#files[@]}" -eq 0 ]; then
  echo "ERROR: manifest has no RTL files: $manifest_path" | tee outputs/sv2v.log
  exit 1
fi

{
  echo "# design_path: $design_path"
  echo "# manifest: $manifest_path"
  echo "# top_module: $top_module"
  printf "%s\n" "${files[@]}"
} > outputs/source-manifest.f

args=()

for define in $sv2v_defines; do
  args+=("-D" "$define")
done

IFS=':' read -r -a include_dirs <<< "$sv2v_include_dirs"

for incdir in "${include_dirs[@]}"; do
  [ -z "$incdir" ] && continue

  if [[ "$incdir" = /* ]]; then
    args+=("-I" "$incdir")
  else
    args+=("-I" "$design_path/$incdir")
  fi
done

{
  echo "Running: $sv2v_bin ${args[*]} -w outputs/design.v <files>"
  "$sv2v_bin" \
    "${args[@]}" \
    -w outputs/design.v \
    "${files[@]}"
} > outputs/sv2v.log 2>&1

if [ ! -s outputs/design.v ]; then
  echo "ERROR: sv2v did not produce a non-empty outputs/design.v" | tee -a outputs/sv2v.log
  exit 1
fi

grep -q "module $top_module" outputs/design.v || {
  echo "ERROR: outputs/design.v does not contain module $top_module" | tee -a outputs/sv2v.log
  exit 1
}

# remove not allowed 'string' specifier on parameter/localparams in verilog

perl -0pi -e 's/\b(parameter|localparam)\s+string\s+/$1 /g' outputs/design.v

