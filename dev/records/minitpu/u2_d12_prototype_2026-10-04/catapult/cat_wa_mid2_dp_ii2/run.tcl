# Project root directory
set sfd [file dir [info script]]

# Create new solution
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11

# Add source files
solution file add "$sfd/kernel.cpp" -type C++

# Set top-level design function
directive set -DESIGN_HIERARCHY wa_d12g

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

solution library add ccs_sample_mem
directive set /wa_d12g/vmem_mem_0/run/mem:rsc -MAP_TO_MODULE ccs_sample_mem.ccs_ram_sync_dualport  ;# [hand-patch]
go assembly
go extract

exit
