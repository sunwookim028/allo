# The U2 flow (u2_regfile_2026-10-02/scripts/dc/dc_u1.tcl, unchanged in every
# setting) plus two optional inputs for the SRAM-macro path
# (asic_memories_2026-10-04.rst): MACRO_DB (a .db from lc_shell, linked beside
# stdcells.db so the macro is a black box whose area is the Liberty's) and
# DEFINES (space-separated `-define` names for MiniTPU's vpu_pkg instances).
# env: DESIGN, SRCS (space-separated), CLK (port name, or "none" -> virtual clock),
#      PERIOD (ns), ADK (view-standard dir), OUT (report dir), [MACRO_DB], [DEFINES]
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
set macro_db [expr {[info exists ::env(MACRO_DB)] ? $::env(MACRO_DB) : ""}]
set_app_var link_library      "* stdcells.db dw_foundation.sldb $macro_db"
set hdlin_ff_always_sync_set_reset      true
set compile_seqmap_honor_sync_set_reset true
if {[info exists ADK_DONT_USE_CELL_LIST]} { set_dont_use [get_lib_cells $ADK_DONT_USE_CELL_LIST] }
define_design_lib WORK -path $out/WORK
set defs [expr {[info exists ::env(DEFINES)] ? [split $::env(DEFINES)] : [list]}]
if { [llength $defs] } {
  if { ![analyze -format sverilog -define $defs [split $::env(SRCS)]] } { exit 1 }
} else {
  if { ![analyze -format sverilog [split $::env(SRCS)]] } { exit 1 }
}
elaborate $design
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
report_cell -nosplit [get_cells -hierarchical -filter "is_black_box==true"] > $out/$design.blackbox.rpt
report_power           > $out/$design.power.rpt
report_clock_gating    > $out/$design.clock_gating.rpt
write -format verilog -hierarchy -output $out/$design.mapped.v
exit
