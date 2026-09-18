#!/bin/bash
# Xcelium counterpart of run_mulacc.sh, with the same arguments:
#   run_mulacc_xrun.sh <pe_wire|pe_stream|pe_channel> [-d MACRO[=VAL] ...]
# RTLDIR selects the netlist root (default ../noc/rtl); it must hold <design>/rtl.v.
S="$(cd "$(dirname "$0")" && pwd)"
V=$1; shift
case $V in
  pe_wire)    B=BOUNDARY_WIRE ;;
  pe_stream)  B=BOUNDARY_FIFO ;;
  pe_channel) B=BOUNDARY_HS ;;
esac
RTLDIR="${RTLDIR:-$S/../noc/rtl}"
DEFS=()
while [ $# -gt 0 ]; do
  [ "$1" = "-d" ] && { DEFS+=("+define+$2"); shift 2; } || { DEFS+=("$1"); shift; }
done
TAGEXTRA=$(printf '%s' "${DEFS[*]}" | sed 's/+define+//g' | tr -s ' ' '_' | tr -cd 'A-Za-z0-9_=')
W="${XRUN_WORK:-$S}/xr_${V}${TAGEXTRA}"
rm -rf "$W"; mkdir -p "$W"; cd "$W"
unset LD_PRELOAD
export CDS_LIC_FILE="${CDS_LIC_FILE:-5280@en-license-05.coecis.cornell.edu}"
XRUN="${XRUN:-/opt/cadence/XCELIUM2403/tools.lnx86/bin/xrun}"
"$XRUN" -q -timescale 1ns/1ps -top tb +define+$B "+define+TAG=\"$V$TAGEXTRA\"" "${DEFS[@]}" \
      "$RTLDIR/$V/rtl.v" "$S/mgc_shim.v" "$S/tb_mulacc.v" > xrun.log 2>&1 \
  || { echo "XRUN FAILED"; grep -E '\*[EF],' xrun.log | head -20; exit 1; }
grep -E "^==|C\[|RESULT:" xrun.log
