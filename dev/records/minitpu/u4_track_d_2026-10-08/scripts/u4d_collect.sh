#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_collect.sh <scratch> <record dir> <name>...: per Catapult project <name>.prj: the run's
# BUILD/[latency]/wall/peak lines, the u4d_check output (<name>.check), latency.json, memory.json,
# run.tcl, cycle.rpt and rtl.rpt (gzipped), Catapult errors; SHA256 of kernel.cpp and concat_rtl.v
# appended to SHA256SUMS.txt (one line per file, latest wins on re-collect).
S=$1; R=$2; shift 2; L=$R/logs/catapult
for n in "$@"; do p=$S/$n.prj; mkdir -p $L/$n
  grep -h "^BUILD\|^\[latency\]\|Elapsed (wall\|Maximum resident\|^RUN_EXIT" $S/$n.log 2>/dev/null | cut -c1-2000 > $L/$n/build.txt
  [ -f $S/$n.check ] && cp $S/$n.check $L/$n/check.txt
  [ -f $S/$n.d23 ] && cp $S/$n.d23 $L/$n/d23.txt
  for f in latency.json memory.json; do [ -f $p/$f ] && cp $p/$f $L/$n/; done
  [ -f $p/run.tcl ] && gzip -c $p/run.tcl > $L/$n/run.tcl.gz
  v1=$(ls -d $p/build/Catapult/*.v1 2>/dev/null | head -1)
  [ -n "$v1" ] && { gzip -c $v1/cycle.rpt > $L/$n/cycle.rpt.gz; gzip -c $v1/rtl.rpt > $L/$n/rtl.rpt.gz; }
  grep -h "^# Error\|SCHD-30\|SCHD-6" $p/build/catapult.log 2>/dev/null | head -40 > $L/$n/errors.txt
  [ -s $L/$n/errors.txt ] || rm -f $L/$n/errors.txt
  touch $R/SHA256SUMS.txt
  for f in $p/kernel.cpp $v1/concat_rtl.v; do [ -f $f ] || continue
    k=$(echo $f | sed "s#$S/##"); grep -v " $k\$" $R/SHA256SUMS.txt > $R/SHA256SUMS.tmp; mv $R/SHA256SUMS.tmp $R/SHA256SUMS.txt
    sha256sum $f | sed "s#$S/##" >> $R/SHA256SUMS.txt; done
done
