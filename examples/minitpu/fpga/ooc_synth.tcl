# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# vivado -mode batch -source ooc_synth.tcl -tclargs <verilog-dir> <top> <out-dir> [period_ns 5.0] [part]
# Out-of-context synthesis of a Vitis HLS core's RTL (syn/verilog/*.v): utilization (flat and per instance)
# and post-synthesis timing at the given clock on ap_clk. A per-PE area/timing probe, not the board build.
set vdir [lindex $argv 0]; set top [lindex $argv 1]; set out [lindex $argv 2]
set period [expr {[llength $argv] > 3 ? [lindex $argv 3] : 5.0}]
set part [expr {[llength $argv] > 4 ? [lindex $argv 4] : "xczu7ev-ffvc1156-2-e"}]
set_param general.maxThreads 8
file mkdir $out
read_verilog [glob $vdir/*.v]
synth_design -top $top -part $part -mode out_of_context
create_clock -period $period -name ap_clk [get_ports ap_clk]
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 2 -file $out/util_hier.txt
report_timing_summary -file $out/timing.txt
report_timing -max_paths 5 -file $out/paths.txt
