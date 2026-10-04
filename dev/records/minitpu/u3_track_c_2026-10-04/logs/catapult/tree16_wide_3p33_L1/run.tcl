# Project root directory
set sfd [file dir [info script]]

# Create new solution
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11

# Add source files
solution file add "$sfd/kernel.cpp" -type C++

# Set top-level design function
directive set -DESIGN_HIERARCHY top

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
go architect
cycle set {v298.Push()} -from {v294.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v298.Push()} -from {v295.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v298.Push()} -from {v296.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v298.Push()} -from {v297.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v299.Push()} -from {v294.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v299.Push()} -from {v295.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v299.Push()} -from {v296.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v299.Push()} -from {v297.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v300.Push()} -from {v294.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v300.Push()} -from {v295.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v300.Push()} -from {v296.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v300.Push()} -from {v297.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v301.Push()} -from {v294.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v301.Push()} -from {v295.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v301.Push()} -from {v296.Pop()} -equal 1  ;# latency=1 on tree_0
cycle set {v301.Push()} -from {v297.Pop()} -equal 1  ;# latency=1 on tree_0
go assembly
go extract

exit
