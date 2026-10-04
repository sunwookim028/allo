# Project root directory
set sfd [file dir [info script]]

# Create new solution
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11

# Add source files
solution file add "$sfd/kernel.cpp" -type C++

# Set top-level design function
directive set -DESIGN_HIERARCHY rf_0

# Set clock constraints
directive set -CLOCKS {clk {-CLOCK_PERIOD 3.330}}

# Set output language
solution options set /Output/OutputVerilog true
solution options set /Output/OutputVHDL false

directive set -IO_MODE super
directive set -SPECULATE true
solution library add nangate-45nm_beh

# Flow
go analyze
go compile
foreach p {/rf_0/wr/__stateful_rf_0_mem_1:rsc /rf_0/__stateful_rf_0_mem_1:rsc /rf_0/__stateful_rf_0_mem_1 /rf_0/wr/__stateful_rf_0_mem_1 /rf_0/mem /rf_0/run} {
  if {[catch {directive set $p -RESET_CLEARS_ALL_REGS no} msg]} { puts "PROBE FAIL $p : $msg" } else { puts "PROBE OK $p : $msg" }
}

solution library add ccs_sample_mem
go assembly
go extract

exit
