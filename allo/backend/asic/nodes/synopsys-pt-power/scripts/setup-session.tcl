#=========================================================================
# setup-session.tcl
#=========================================================================
# The setup session script configures the PrimteTime session to use
# PT PX or PrimePower to do power analysis
#
# Author : Maximilian Koschay
# Date   : 05.03.2021


# Set up paths and libraries

set_app_var search_path      ". $ptpx_additional_search_path $search_path"
set_app_var target_library   $ptpx_target_libraries
set_app_var link_library     [join "
                               *
                               $ptpx_target_libraries
                               $ptpx_extra_link_libraries
                             "]

# Set up power analysis

set_app_var power_enable_analysis true
set_app_var power_analysis_mode   $ptpx_analysis_mode

set_app_var report_default_significant_digits 3

if {$ptpx_analysis_mode != "averaged" && $ptpx_analysis_mode != "time_based"} {
	echo "Error: analysis_mode must be set either averaged or time_based!"
	exit 1
}
