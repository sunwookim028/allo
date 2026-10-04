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
directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0}}

# Set output language
solution options set /Output/OutputVerilog true
solution options set /Output/OutputVHDL false

directive set -IO_MODE super
directive set -SPECULATE true
solution library add nangate-45nm_beh

# Flow
directive set -REGISTER_THRESHOLD 4096  ;# [u3c --tcl]
go analyze
go compile

solution library add ccs_sample_mem
go architect
cycle set {v6.Push()} -from {v0.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v6.Push()} -from {v1.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v6.Push()} -from {v2.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v6.Push()} -from {v3.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v6.Push()} -from {v4.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v6.Push()} -from {v5.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v0.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v1.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v2.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v3.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v4.Pop()} -equal 1  ;# latency=1 on tx_0
cycle set {v7.Push()} -from {v5.Pop()} -equal 1  ;# latency=1 on tx_0
go assembly
go extract

exit
