#=============================================================================
# TCL Script for Example 1: Basic Static Variable
#=============================================================================

set hls_prj ex01.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function
set_top top

# Add design and testbench files
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=c++11"

open_solution "solution1"

# Use a common device (Zynq-7000)
set_part {xc7z020clg484-1}

# Target clock period is 10ns (100 MHz)
create_clock -period 10

# Run C simulation (fast iteration)
csim_design -O

# Optionally synthesize to see RTL
# csynth_design

exit

