set hls_prj ex08_test5.prj
open_project ${hls_prj} -reset
open_solution -reset solution1 -flow_target vivado
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=c++11"
set_top matmul_shared
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10
csim_design -O
exit
