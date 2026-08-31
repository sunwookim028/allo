#=========================================================================
# mmmc.tcl
#=========================================================================
# MMMC setup for the single-node Innovus PNR flow. Innovus requires
# set_analysis_view to run as part of init_design's MMMC processing.

set sram_lib_files [lsort [glob -nocomplain inputs/srams/*/*.lib]]

if {![info exists env(adk_cap_table)]} {
  set env(adk_cap_table) inputs/adk/rtk-typical.captable
}
if {![info exists env(adk_typical_lib)]} {
  set env(adk_typical_lib) inputs/adk/stdcells.lib
}
if {![info exists env(adk_bc_lib)]} {
  set env(adk_bc_lib) inputs/adk/stdcells-bc.lib
}
if {![info exists env(adk_wc_lib)]} {
  set env(adk_wc_lib) inputs/adk/stdcells-wc.lib
}

create_rc_corner -name typical -cap_table $env(adk_cap_table) -T 25

create_library_set -name libs_typical \
  -timing [concat \
    [list $env(adk_typical_lib)] \
    $sram_lib_files \
  ]

create_library_set -name libs_bc \
  -timing [concat [list $env(adk_bc_lib)] $sram_lib_files]

create_library_set -name libs_wc \
  -timing [concat \
    [list $env(adk_wc_lib)] \
    $sram_lib_files \
  ]

create_delay_corner -name delay_typical \
  -early_library_set libs_typical \
  -late_library_set libs_typical \
  -rc_corner typical

create_constraint_mode -name constraints_default \
  -sdc_files [list ./inputs/design.sdc]

create_analysis_view -name analysis_default \
  -constraint_mode constraints_default \
  -delay_corner delay_typical

set_analysis_view -setup [list analysis_default] -hold [list analysis_default]
