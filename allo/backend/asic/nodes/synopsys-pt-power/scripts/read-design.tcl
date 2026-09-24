#=========================================================================
# read-design.tcl
#=========================================================================
# This script reads in all the input files
#
# Author : Maximilian Koschay
# Date   : 05.03.2021

# Read and link the design

read_verilog   $ptpx_gl_netlist
current_design $ptpx_design_name

link_design > ${ptpx_logs_dir}/${ptpx_design_name}.link.rpt

# Namemapping for RTL switching activities

if { $ptpx_activity_is_rtl == True } {
	echo "Sourcing RTL name mapping file for RTL switching activity annotation."
	source $ptpx_namemap -echo > ${ptpx_logs_dir}/${ptpx_design_name}.namemap.rpt
}

# Read in switching activity

if {$ptpx_activity_format == "vcd"} {
	report_activity_file_check $ptpx_vcd -strip_path $ptpx_strip_path \
		> $ptpx_reports_dir/$ptpx_design_name.activity.pre.rpt

	if { $ptpx_activity_is_rtl == True } {
		echo "Read VCD activity annotation file from RTL simulation."
		read_vcd -strip_path $ptpx_strip_path -rtl $ptpx_vcd 
	} else {
		if {$ptpx_activity_is_zero_delay == True} {
			echo "Read VCD activity annotation file from zero-delay simulation."
			read_vcd -strip_path $ptpx_strip_path -zero_delay $ptpx_vcd 
		} else {
			echo "Read VCD activity annotation file from SDF annotated simulation."
			read_vcd -strip_path $ptpx_strip_path $ptpx_vcd 
		}
		
	}
	
} else {
	report_activity_file_check $ptpx_saif -strip_path $ptpx_strip_path \
		> $ptpx_reports_dir/$ptpx_design_name.activity.pre.rpt

	echo "Read SAIF activity annotation file."
	read_saif -strip_path $ptpx_strip_path $ptpx_saif
}

# Read SDC file

# read_sdc -echo $ptpx_sdc > ${ptpx_logs_dir}/${ptpx_design_name}.read_sdc.rpt
source -echo -verbose $ptpx_sdc > ${ptpx_logs_dir}/${ptpx_design_name}.read_sdc.rpt

# Read parsitics file

read_parasitics -format spef $ptpx_spef
report_annotated_parasitics -check \
  > $ptpx_reports_dir/$ptpx_design_name.parasitics.rpt

# Set operating condition in case mutliple libraries are loaded

if {$ptpx_op_condition != "undefined"} {
	set_operating_condition $ptpx_op_condition
} else {
	echo "No operating condition is defined. PrimeTime will choose one."
}


# This is the entry point for the debug view
if {[info exists ::env(SYN_EXIT_AFTER_SETUP)]} { set SYN_SETUP_DONE true }
