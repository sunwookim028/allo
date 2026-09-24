options set Input/CppStandard c++11
solution new -state initial
solution file add stream_boundary.cpp -type C++
directive set -DESIGN_HIERARCHY {source_0}
go analyze
go compile
solution library add nangate-45nm_beh
directive set -CLOCKS {clk {-CLOCK_PERIOD 5.0}}
go assembly
go extract
project save
exit
