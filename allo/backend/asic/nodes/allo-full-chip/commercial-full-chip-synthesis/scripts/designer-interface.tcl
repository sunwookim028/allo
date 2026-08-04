#=========================================================================
# designer-interface.tcl
#=========================================================================
# The designer-interface.tcl file is the first script run by Design
# Compiler. It is the interface that connects the synthesis scripts with
# the following:
#
# - Build system parameters
# - Build system inputs
# - ASIC design kit
#
# Author : Christopher Torng
# Date   : April 8, 2018

#-------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------

set design_name                   $::env(top_module)
set clock_period                  $::env(clock_period)
# dc_design_name and dc_clock_period kept for backwards comptability,
# prefer to use design_name and clock_period for tool independence
set dc_design_name                $::env(top_module)
set dc_clock_period               $::env(clock_period)
set dc_saif_instance              $::env(saif_instance)
set dc_flatten_effort             $::env(flatten_effort)
set dc_topographical              $::env(topographical)
set dc_num_cores                  $::env(nthreads)
set dc_high_effort_area_opt       $::env(high_effort_area_opt)
set dc_gate_clock                 $::env(gate_clock)
set dc_uniquify_with_design_name  $::env(uniquify_with_design_name)
set dc_suppress_msg 			  $::env(suppress_msg)
set dc_suppressed_msg  			  [split $::env(suppressed_msg) ","]
set write_svsim_wrapper $::env(write_svsim_wrapper)

#-------------------------------------------------------------------------
# Inputs
#-------------------------------------------------------------------------

set dc_rtl_handoff              inputs/assembled-design.v
set adk_dir                     inputs/adk
set dc_upf                      inputs/design.upf

# Extra libraries
#
# The glob below will capture any libraries collected by the build system
# (e.g., SRAM libraries) generated from steps that synthesis depends on.
#
# To add more link libraries (e.g., IO cells, hierarchical blocks), append
# to the "dc_extra_link_libraries" variable in the pre-synthesis plugin
# like this:
#
#   set dc_extra_link_libraries  [join "
#                                  $dc_extra_link_libraries
#                                  extra1.db
#                                  extra2.db
#                                  extra3.db
#                                "]

set dc_extra_link_libraries     [join "
                                    [lsort [glob -nocomplain inputs/*.db]]
                                    [lsort [glob -nocomplain inputs/adk/*.db]]
                                "]

# Hardened Allo macros are link-only library cells. Their DBs must never be
# added to target_library, or DC could map arbitrary logic into macro cells.
if {![file exists inputs/macro-collateral.tcl]} {
  echo "Error: missing inputs/macro-collateral.tcl"
  exit 1
}
source inputs/macro-collateral.tcl
if {![info exists allo_asic_macro_db_files] ||
    ![info exists allo_asic_macro_modules] ||
    ![info exists allo_asic_bypass_macro_generation]} {
  echo "Error: macro collateral is missing required declarations"
  exit 1
}
set macro_db_count [llength $allo_asic_macro_db_files]
set macro_module_count [llength $allo_asic_macro_modules]
if {$macro_db_count != $macro_module_count} {
  echo "Error: macro collateral defines $macro_db_count DBs but $macro_module_count modules"
  exit 1
}
if {$allo_asic_bypass_macro_generation} {
  if {$macro_db_count != 0} {
    echo "Error: flat bypass collateral unexpectedly defines hardened macros"
    exit 1
  }
  echo "Info: flat macro-bypass mode selected; no hardened macro DBs will be linked"
} elseif {$macro_db_count == 0} {
  echo "Error: hierarchical macro mode does not define any macro DBs or modules"
  exit 1
}
foreach macro_db $allo_asic_macro_db_files {
  if {![file exists $macro_db]} {
    echo "Error: missing hardened macro DB $macro_db"
    exit 1
  }
}
set dc_extra_link_libraries [concat $dc_extra_link_libraries $allo_asic_macro_db_files]

#-------------------------------------------------------------------------
# Interface to the ASIC design kit
#-------------------------------------------------------------------------

set dc_milkyway_ref_libraries   $adk_dir/stdcells.mwlib
set dc_milkyway_tf              $adk_dir/rtk-tech.tf
set dc_tluplus_map              $adk_dir/rtk-tluplus.map
set dc_tluplus_max              $adk_dir/rtk-max.tluplus
set dc_tluplus_min              $adk_dir/rtk-min.tluplus
set dc_adk_tcl                  $adk_dir/adk.tcl
set dc_target_libraries         stdcells.db

# add srams

set dc_sram_db_files [glob -nocomplain inputs/srams/*/*.db]

if {[llength $dc_sram_db_files] > 0} {
  puts "Info: Found SRAM db files: $dc_sram_db_files"
  set dc_extra_link_libraries [concat $dc_extra_link_libraries $dc_sram_db_files]
  set dc_target_libraries [concat $dc_target_libraries $dc_sram_db_files]
}

# Extra libraries

set dc_additional_search_path   $adk_dir

#-------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------

set dc_reports_dir              reports
set dc_results_dir              results
set dc_alib_dir                 alib
