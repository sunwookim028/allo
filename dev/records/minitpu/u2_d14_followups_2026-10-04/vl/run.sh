#!/bin/bash
# run.sh <concat_sim_rtl.v> <outdir>
set -e
src=$1; out=$2; rm -rf $out; mkdir -p $out
export PATH=/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH
scl enable gcc-toolset-13 -- bash -c "verilator --cc --exe --build -Wno-fatal -Wno-lint -Wno-style --top-module rf_0 -Mdir $out $src $(dirname $0)/tb.cpp -o sim > $out/build.log 2>&1"
$out/sim
