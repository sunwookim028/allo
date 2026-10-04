# Vivado out-of-context synthesis of an AMC dump (kernel + FSM; the BRAMs are
# the kernel's ports; mem*_bram are thin adapters). AMC's own
# AMCModule.get_resource_estimates() is broken at fe60c121 (it calls a
# missing get_verilog() and an undefined calyxDir), hence this script.
#   vivado -mode batch -source synth.tcl -tclargs <dump_dir> <out_dir> [period_ns] [part]
set d [lindex $argv 0]
set o [lindex $argv 1]
set period [expr {[llength $argv] > 2 ? [lindex $argv 2] : 10.0}]
set part [expr {[llength $argv] > 3 ? [lindex $argv 3] : "xcu55c-fsvh2892-2L-e"}]
set top [expr {[llength $argv] > 4 ? [lindex $argv 4] : "sfu"}]
file mkdir $o
read_verilog -sv [concat [list $d/fsm_enum_typedefs.sv $d/${top}_fsm.sv $d/${top}.sv] [glob -nocomplain $d/mem*_bram.sv $d/*_impl.sv $d/int_arith_pipe.v $d/ieeefpmult_l5.sv]]
synth_design -mode out_of_context -top $top -part $part
create_clock -name clk -period $period [get_ports clk]
# Time the port-to-port paths too: the AMC datapath runs combinationally from
# a BRAM's dout to another's din, which OOC leaves untimed without I/O delays.
set_input_delay 0 -clock clk [get_ports -filter {NAME != clk}]
set_output_delay 0 -clock clk [all_outputs]
opt_design
report_utilization -file $o/util.rpt
report_timing_summary -file $o/timing.rpt
