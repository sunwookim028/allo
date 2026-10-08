# U3 track C: dc_u1.tcl (U1, byte-identical flow) plus env DEFINES ("A=1 B") -> analyze -define, and PARAMS ("N=16") -> elaborate -parameters, for MiniTPU's small tree geometry. Nothing else changed.
# One DC flow for every U1 RTL: FreePDK45 view-standard stdcells.db (the TinyTPU
# ASIC flow's ADK), compile_ultra, flatten, clock gating, as mflowgen's
# synopsys-dc-synthesis node; non-topographical; inputs/outputs at 0 delay.
# env: DESIGN, SRCS (space-separated), CLK (port name, or "none" -> virtual clock),
#      PERIOD (ns), ADK (view-standard dir), OUT (report dir)
set design  $::env(DESIGN)
set period  $::env(PERIOD)
set adk     $::env(ADK)
set out     $::env(OUT)
file mkdir $out
source $adk/adk.tcl
set_host_options -max_cores 8
set_app_var search_path ". $adk $search_path"
set_app_var target_library    stdcells.db
set_app_var synthetic_library dw_foundation.sldb
set_app_var link_library      "* stdcells.db dw_foundation.sldb"
set hdlin_ff_always_sync_set_reset      true
set compile_seqmap_honor_sync_set_reset true
if {[info exists ADK_DONT_USE_CELL_LIST]} { set_dont_use [get_lib_cells $ADK_DONT_USE_CELL_LIST] }
define_design_lib WORK -path $out/WORK
set defs ""
if { [info exists ::env(DEFINES)] && $::env(DEFINES) ne "" } { set defs [list -define [split $::env(DEFINES)]] }
if { ![eval analyze -format sverilog $defs [list [split $::env(SRCS)]]] } { exit 1 }
if { [info exists ::env(PARAMS)] && $::env(PARAMS) ne "" } { elaborate $design -parameters $::env(PARAMS) } else { elaborate $design }
current_design $design
if { ![link] } { echo "Error: failed to link $design"; exit 1 }
if { $::env(CLK) eq "none" } {
  create_clock -name ideal_clock -period $period
  set ins [all_inputs]
} else {
  create_clock -name ideal_clock -period $period [get_ports $::env(CLK)]
  set ins [remove_from_collection [all_inputs] [get_ports $::env(CLK)]]
}
set_input_delay  -clock ideal_clock 0 $ins
set_driving_cell -no_design_rule -lib_cell $ADK_DRIVING_CELL $ins
set_output_delay -clock ideal_clock 0 [all_outputs]
set_load -pin_load $ADK_TYPICAL_ON_CHIP_LOAD [all_outputs]
set_max_fanout 20 [current_design]
set_max_transition [expr $period * 0.25] [current_design]
ungroup -start_level 2 -all -flatten
set_app_var compile_ultra_ungroup_dw true
set_fix_multiple_port_nets -all -buffer_constants
check_design -summary
compile_ultra -gate_clock
check_design -summary
report_area            > $out/$design.area.rpt
report_area -hierarchy > $out/$design.area.hier.rpt
report_timing -max_paths 1 -nets -transition_time -input_pins > $out/$design.timing.rpt
report_qor             > $out/$design.qor.rpt
report_reference       > $out/$design.reference.rpt
report_power           > $out/$design.power.rpt
report_clock_gating    > $out/$design.clock_gating.rpt
write -format verilog -hierarchy -output $out/$design.mapped.v
exit
