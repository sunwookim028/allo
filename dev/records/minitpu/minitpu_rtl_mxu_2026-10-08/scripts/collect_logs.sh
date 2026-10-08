#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: collect_logs.sh <scratch> <record-dir>: keep per build its latency.json, run.tcl, rtl.rpt.gz, cycle.rpt.gz,
# the BUILD/[latency] lines, the generated mxu_allo.sv, the tb outputs, and SHA256SUMS of kernel.cpp / concat_rtl.v.
set -euo pipefail
S=$1; R=$2
mkdir -p "$R/logs/catapult" "$R/rtl"
: > "$R/SHA256SUMS.txt"
for prj in "$S"/*.prj; do
  n=$(basename "$prj" .prj)
  top=$(grep -o "mxu_wide_dim[0-9]*" "$prj/kernel.cpp" | head -1)
  sol="$prj/build/Catapult/$top.v1"
  d="$R/logs/catapult/$n"; mkdir -p "$d"
  cp "$prj/run.tcl" "$d/"; [ -f "$prj/latency.json" ] && cp "$prj/latency.json" "$d/"
  for f in rtl.rpt cycle.rpt; do [ -f "$sol/$f" ] && gzip -c "$sol/$f" > "$d/$f.gz"; done
  [ -f "$S/$n.log" ] && grep -h "^BUILD\|^EMITTED\|^\[latency\]\|SCHD-30\|could not schedule\|^real" "$S/$n.log" | cut -c1-2000 > "$d/build.txt"
  grep -h "Error" "$S/$n.log" | head -20 >> "$d/build.txt" || true
  (cd "$prj" && sha256sum kernel.cpp) | sed "s#^\(\S*\)  #\1  $n/#" >> "$R/SHA256SUMS.txt"
  [ -f "$sol/concat_rtl.v" ] && (cd "$sol" && sha256sum concat_rtl.v) | sed "s#^\(\S*\)  #\1  $n/#" >> "$R/SHA256SUMS.txt"
done
for t in "$S"/tb*/mxu_allo.sv; do cp "$t" "$R/rtl/mxu_allo_$(basename "$(dirname "$t")").sv"; done
cp "$S"/*.out "$R/logs/" 2>/dev/null || true
cp "$S"/measured_*.json "$S"/versions_*.json "$R/logs/" 2>/dev/null || true
echo "collected into $R"
