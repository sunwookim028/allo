#!/bin/bash
# usage: run_mulacc.sh <pe_wire|pe_stream|pe_channel> [extra xvlog -d flags...]
S="$(cd "$(dirname "$0")" && pwd)"
V=$1; shift
case $V in
  pe_wire)    B=BOUNDARY_WIRE ;;
  pe_stream)  B=BOUNDARY_FIFO ;;
  pe_channel) B=BOUNDARY_HS ;;
esac
TAGEXTRA=$(echo "$*" | tr -s ' ' '_' | tr -cd 'A-Za-z0-9_=')
W="$S/ma_${V}${TAGEXTRA}"
rm -rf "$W"; mkdir -p "$W"; cd "$W"
source /opt/xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1 || true
xvlog -d $B -d "TAG=\"$V$TAGEXTRA\"" "$@" \
      "$S/../noc/rtl/$V/rtl.v" "$S/mgc_shim.v" "$S/tb_mulacc.v" > xvlog.log 2>&1 \
  || { echo "XVLOG FAILED"; tail -20 xvlog.log; exit 1; }
xelab --timescale 1ns/1ps tb -s sim > xelab.log 2>&1 \
  || { echo "XELAB FAILED"; tail -20 xelab.log; exit 1; }
xsim sim -R 2>&1 | grep -E "^==|C\[|RESULT:"
