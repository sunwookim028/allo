#=========================================================================
# read-design.tcl
#=========================================================================
# Author : Christopher Torng
# Date   : May 14, 2018
#

# Check libraries

check_library > $dc_reports_dir/${dc_design_name}.check_library.rpt

# The first "WORK" is a reserved word for Design Compiler. The value for
# the -path option is customizable.

define_design_lib WORK -path ${dc_results_dir}/WORK

# import SRAMs

#set dc_sram_verilog_files [glob -nocomplain inputs/srams/*/*.v]
#
#foreach sram_v $dc_sram_verilog_files {
#  puts "Info: Reading SRAM Verilog model: $sram_v"
#  if { ![analyze -format verilog $sram_v] } { exit 1 }
#}

# Analyze the RTL source file

if { ![analyze -format sverilog $dc_rtl_handoff] } { exit 1 }

# Elaborate the design with design parameters from a file, or else just
# elaborate normally

if {[file exists [which setup-design-params.txt]]} {
  elaborate $dc_design_name -file_parameters setup-design-params.txt
  rename_design $dc_design_name* $dc_design_name
} else {
  elaborate $dc_design_name
}

current_design $dc_design_name
if {![link]} {
  echo "Error: failed to link full-chip design"
  exit 1
}

# Prove that each canonical hardened macro linked as an instantiated library
# cell, then preserve those instances across full-chip optimization.
set macro_report [open $dc_reports_dir/macro-link.rpt w]
set linked_macro_instances 0
foreach macro_name $allo_asic_macro_modules {
  set macro_cells [get_cells -hierarchical -filter "ref_name == $macro_name"]
  set count [sizeof_collection $macro_cells]
  puts $macro_report "$macro_name $count"
  if {$count == 0} {
    close $macro_report
    echo "Error: no linked instances found for hardened macro $macro_name"
    exit 1
  }
  set_dont_touch $macro_cells true
  incr linked_macro_instances $count
}
puts $macro_report "TOTAL $linked_macro_instances"
close $macro_report

# Load UPF if it exists
if {[file exists $dc_upf]} {
  load_upf $dc_upf
}

#-------------------------------------------------------------------------
# Write out useful files
#-------------------------------------------------------------------------

# This ddc can be used as a checkpoint to load up to the current state

write -hierarchy -format ddc \
      -output ${dc_results_dir}/${dc_design_name}.elab.ddc

# This Verilog is useful to double-check the netlist that dc will use for
# mapping

write -hierarchy -format verilog \
      -output ${dc_results_dir}/${dc_design_name}.elab.v

