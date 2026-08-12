#=========================================================================
# designer-interface.tcl
#=========================================================================
# The designer_interface.tcl file is the first script run by PT 
# and sets up ASIC design kit variables and inputs.
#
# Author : Christopher Torng
# Date   : May 20, 2019


#-------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------

set ptpx_design_name        		$::env(design_name)

# The strip path must be defined!
#
#   export strip_path = th/dut
#
# There must _not_ be any quotes, or read_saif will fail. This fails:
#
#   export strip_path = "th/dut"
#

set ptpx_strip_path         		$::env(saif_instance)

set ptpx_analysis_mode				$::env(analysis_mode)
set ptpx_zero_delay_simulation		$::env(zero_delay_simulation)
set ptpx_op_condition				$::env(lib_op_condition)

#-------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------

set adk_dir                       inputs/adk

set ptpx_additional_search_path   $adk_dir
set ptpx_target_libraries         stdcells.db

set ptpx_extra_link_libraries     [join "
                                      [lsort [glob -nocomplain inputs/*.db]]
                                      [lsort [glob -nocomplain inputs/adk/*.db]]
                                  "]

set ptpx_sram_db_files [lsort [glob -nocomplain inputs/srams/*/*.db]]

# Hardened Allo macro timing/power models are published one directory per
# macro class. An absent or empty registry contributes no extra libraries.
set ptpx_macro_db_files [lsort [glob -nocomplain inputs/macro-registry/*/*.db]]

set ptpx_extra_link_libraries [join "
  $ptpx_extra_link_libraries
  $ptpx_sram_db_files
  $ptpx_macro_db_files
"]

#-------------------------------------------------------------------------
# Inputs
#-------------------------------------------------------------------------

set ptpx_gl_netlist         inputs/design.vcs.v
set ptpx_sdc                prepared-design.pt.sdc
set ptpx_spef               inputs/design.spef.gz
set ptpx_saif               inputs/run.saif
set ptpx_vcd				inputs/run.vcd
set ptpx_namemap			inputs/design.namemap

#-------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------

set ptpx_reports_dir	   	reports
set ptpx_logs_dir	   		logs
set ptpx_outputs_dir		outputs
