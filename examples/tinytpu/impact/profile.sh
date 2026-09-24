#!/bin/bash
# Per-process timeline of the LAST shape cosim'd in a run dir (tb.cpp is left
# at the last shape by cosim.py). Re-runs cosim with dataflow profiling, then
# re-runs the xsim snapshot so the monitor's CSVs survive, then summarizes.
#   ./profile.sh runs/<variant>
set -e
P=$(readlink -f $1)/isa_sweep.prj
cat > $P/run_prof.tcl <<'T'
open_project out.prj
open_solution solution1
set_top tinytpu_isa
cosim_design -trace_level port -enable_dataflow_profiling -rtl verilog -ldflags "-B/usr/bin"
exit
T
bash -lc "source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh && cd $P && vitis_hls -f run_prof.tcl" > $P/cosim_prof.log 2>&1
grep -h "mismatches" $P/cosim_prof.log
V=$P/out.prj/solution1/sim/verilog
printf 'run all\nquit\n' > $V/mine.tcl
bash -lc "source /opt/xilinx/Vivado/2023.2/settings64.sh >/dev/null; cd $V && xsim --noieeewarnings tinytpu_isa -tclbatch mine.tcl" > $V/mine.log 2>&1
D=$(dirname $0)
python3 $D/analyze_df.py $V > $1/timeline.txt
python3 $D/rle_df.py $V $(grep -oP "(?<=AESL_inst_tinytpu_isa\.)(dma_ld|spm|vru|accu|dma_st|pe_0_0|pe_3_3|wld_0_0)\w*?(?=_U0\.ap_start)" $V/dataflow_monitor.sv | sort -u) > $1/rle.txt
cat $1/timeline.txt
