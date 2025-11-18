# MiniTPU_HLS Vitis HLS batch project script.

open_project miniTPU_HLS
set_top top_module

add_files ../src/top_module.cpp -cflags "-I../include"
add_files -tb ../host/host_test.cpp -cflags "-I../include"

open_solution -reset solution1
set_part {xcvu9p-flga2104-2-i}
create_clock -period 5.0 -name default

csim_design -clean
csynth_design
# cosim_design ;# Uncomment after providing cycle-accurate RTL test bench.

export_design -rtl verilog -format ip_catalog

exit


