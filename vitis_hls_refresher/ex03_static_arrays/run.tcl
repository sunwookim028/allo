#=============================================================================
# TCL Script for Example 3: Static Arrays
#=============================================================================

set hls_prj ex03.prj

open_project ${hls_prj} -reset
open_solution -reset solution1 -flow_target vivado

set_top top_c
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=c++11"

open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design
csynth_design
csynth_design 

exit

