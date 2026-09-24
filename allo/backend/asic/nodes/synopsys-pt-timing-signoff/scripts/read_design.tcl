#-------------------------------------------------------------------------
# Designer interface
#-------------------------------------------------------------------------
# Source the designer interface script, which sets up variables from the
# build system, sets up ASIC design kit variables, etc.

source -echo -verbose designer_interface.tcl

#-------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------

# Set up paths and libraries

set_app_var search_path      ". $ptpx_additional_search_path $search_path"
set_app_var target_library   $ptpx_target_libraries
set_app_var link_library     "* $ptpx_target_libraries $ptpx_extra_link_libraries"

#-------------------------------------------------------------------------
# Read design
#-------------------------------------------------------------------------

# Read and link the design

read_verilog   $ptpx_gl_netlist
current_design $ptpx_design_name

if {[catch {link_design} link_error]} {
  echo "Error: failed to link $ptpx_design_name: $link_error"
  exit 1
}

# Read in the SDC and parasitics

# read_sdc -echo $ptpx_sdc
source $ptpx_sdc

if {[catch {read_parasitics -format spef $ptpx_spef} spef_error]} {
  echo "Error: failed to read parasitics $ptpx_spef: $spef_error"
  exit 1
}
