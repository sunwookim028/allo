set sfd [file dir [info script]]
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11
solution file add "$sfd/kernel.cpp" -type C++
directive set -DESIGN_HIERARCHY rf_comb
directive set -CLOCKS {clk {-CLOCK_PERIOD 2.000}}
solution options set /Output/OutputVerilog true
solution options set /Output/OutputVHDL false
directive set -IO_MODE super
directive set -SPECULATE true
solution library add nangate-45nm_beh
go analyze
go compile
solution library add ccs_sample_mem
go assembly
go extract
exit
