#!/bin/bash
# usage: collect_logs.sh <scratch> <record dir>: BUILD/cmp lines, per-project latency.json + cycle.rpt + rtl.rpt (gzipped) + run.tcl,
# DC area/qor reports, and SHA256SUMS of every emitted kernel.cpp, concat_rtl.v, DC netlist and MiniTPU source used.
S=$1; R=$2; L=$R/logs; mkdir -p $L/catapult $L/dc
cat $S/batch*.out > $L/catapult_builds.txt 2>/dev/null
for f in $S/cmp_*.out; do cp $f $L/; done
for p in $S/*.prj; do n=$(basename $p .prj); case $n in probe_*) continue;; esac
  v1=$(ls -d $p/build/Catapult/*.v1 2>/dev/null | head -1); mkdir -p $L/catapult/$n
  cp $p/latency.json $L/catapult/$n/ 2>/dev/null; cp $p/run.tcl $L/catapult/$n/ 2>/dev/null
  [ -n "$v1" ] && { gzip -c $v1/cycle.rpt > $L/catapult/$n/cycle.rpt.gz; gzip -c $v1/rtl.rpt > $L/catapult/$n/rtl.rpt.gz; }
  grep -h "SCHD-30\|SCHD-6\|CNS-4\|MEM-8\|^# Error" $p/build/catapult.log 2>/dev/null | head -40 > $L/catapult/$n/errors.txt; [ -s $L/catapult/$n/errors.txt ] || rm -f $L/catapult/$n/errors.txt
done
for o in $S/dc/out/*; do n=$(basename $o); mkdir -p $L/dc/$n; cp $o/*.area.rpt $o/*.qor.rpt $L/dc/$n/ 2>/dev/null; gzip -c $o/*.timing.rpt > $L/dc/$n/timing.rpt.gz 2>/dev/null; done
python3 dev/records/minitpu/u1_catapult_units_2026-10-02/scripts/dc/summarize_dc.py $S/dc > $L/dc_summary.txt
python3 $R/scripts/u3c_summary.py $(ls -d $S/*.prj | grep -v probe_) > $L/catapult_summary.txt
( for p in $S/*.prj; do n=$(basename $p .prj); case $n in probe_*) continue;; esac; sha256sum $p/kernel.cpp 2>/dev/null | sed "s#$S/##"; for v in $p/build/Catapult/*.v1/concat_rtl.v; do [ -f $v ] && sha256sum $v | sed "s#$S/##"; done; done
  for o in $S/dc/out/*; do for f in $o/*.mapped.v; do [ -f $f ] && sha256sum $f | sed "s#$S/dc/out/##"; done; done
  M=/work/shared/users/phd/sk3463/minitpu; for f in src/core/vpu/vpu_pkg.sv src/core/sfu/sfu.sv src/core/sfu/gelu_bf16.mem src/core/sfu/exp_bf16.mem src/core/vpu/vpu_bf16_add_pipe.sv src/core/xlu/xlu_reduction_tree.sv src/core/xlu/xlu_transpose.sv src/core/mxu/mxu_bf16_mul_acc24.sv src/core/mxu/mxu_acc24_add_pipe.sv src/core/mxu/mxu_pe.sv src/core/mxu/mxu_systolic_array.sv src/core/mxu/mxu.sv; do sha256sum $M/$f | sed "s#$M/#minitpu@b3ba0a4d/#"; done
  sha256sum $S/sfu_dc.sv | sed "s#$S/#dc-standin/#" ) > $R/SHA256SUMS.txt
echo "collected: $(ls $L/catapult | wc -l) projects, $(ls $L/dc | wc -l) DC runs, $(wc -l < $R/SHA256SUMS.txt) hashes"
