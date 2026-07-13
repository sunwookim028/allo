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
: "${top_module:?Missing parameter: top_module}"
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
excludes=()

resolve_manifest_path() {
  local entry="$1"

  if [[ "$entry" = /* ]]; then
    printf "%s\n" "$entry"
  else
    printf "%s\n" "$design_path/$entry"
  fi
}

add_entry() {
  local entry="$1"
  local path
  path="$(resolve_manifest_path "$entry")"

  if [ -f "$path" ]; then
    files+=("$path")
  elif [ -d "$path" ]; then
    while IFS= read -r file; do
      files+=("$file")
    done < <(find "$path" -maxdepth 1 -type f \( -name '*.sv' -o -name '*.v' \) | sort)
  else
    echo "ERROR: manifest entry does not exist: $path" | tee outputs/sv2v.log
    exit 1
  fi
}

add_exclude() {
  local entry="$1"
  local path
  path="$(resolve_manifest_path "$entry")"

  if [ -f "$path" ]; then
    excludes+=("$path")
  elif [ -d "$path" ]; then
    while IFS= read -r file; do
      excludes+=("$file")
    done < <(find "$path" -maxdepth 1 -type f \( -name '*.sv' -o -name '*.v' \) | sort)
  else
    echo "ERROR: manifest exclude does not exist: $path" | tee outputs/sv2v.log
    exit 1
  fi
}

while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"

  [ -z "$line" ] && continue

  if [[ "$line" == !* ]]; then
    add_exclude "${line#!}"
  else
    add_entry "$line"
  fi
done < "$manifest_path"

# Remove excluded files and deduplicate while preserving order.
filtered=()
seen=" "

for file in "${files[@]}"; do
  skip=0

  for ex in "${excludes[@]}"; do
    if [ "$file" = "$ex" ]; then
      skip=1
      break
    fi
  done

  if [ "$skip" -eq 0 ] && [[ "$seen" != *" $file "* ]]; then
    filtered+=("$file")
    seen="${seen}${file} "
  fi
done

files=("${filtered[@]}")

if [ "${#files[@]}" -eq 0 ]; then
  echo "ERROR: manifest has no RTL files: $manifest_path" | tee outputs/sv2v.log
  exit 1
fi

{
  echo "# design_path: $design_path"
  echo "# manifest: $manifest_path"
  echo "# top_module: $top_module"
  echo "# excludes:"
  printf "#   %s\n" "${excludes[@]}"
  echo "# files:"
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

# make flow print more detailed logging info

args+=("-v")

{
  echo "Running: $sv2v_bin ${args[*]} -w outputs/design.v"
  printf "  %s\n" "${files[@]}"
  "$sv2v_bin" \
    "${args[@]}" \
    -w outputs/design.v \
    "${files[@]}"
} 2>&1 | tee outputs/sv2v.log

if [ ! -s outputs/design.v ]; then
  echo "ERROR: sv2v did not produce a non-empty outputs/design.v" | tee -a outputs/sv2v.log
  exit 1
fi

# remove not allowed 'string' specifier on parameter/localparams in verilog

perl -0pi -e 's/\b(parameter|localparam)\s+string\s+/$1 /g' outputs/design.v

# check top module is present

grep -Eq "^[[:space:]]*module[[:space:]]+$top_module([[:space:]#(;]|$)" outputs/design.v || {
  echo "ERROR: outputs/design.v does not contain module $top_module" | tee -a outputs/sv2v.log
  exit 1
}

