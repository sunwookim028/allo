#!/bin/bash
# usage: run.sh <variant> [extra -d defines...]
set -e
S="$(cd "$(dirname "$0")" && pwd)"
V=$1; shift
W="$S/w_${V}_$(echo "$*" | tr -cd 'A-Za-z0-9')"
rm -rf "$W"; mkdir -p "$W"; cd "$W"
source /opt/xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1 || true
xvlog -d TOPMOD=$V -d "TOPNAME=\"$V\"" "$@" \
      "$S/../noc/rtl/$V/rtl.v" "$S/mgc_shim.v" "$S/tb_top.v" > xvlog.log 2>&1
xelab --timescale 1ns/1ps tb -s sim > xelab.log 2>&1 || { tail -20 xelab.log; exit 1; }
xsim sim -R 2>&1 | grep -E "TOP=|C\[|RESULT:"
