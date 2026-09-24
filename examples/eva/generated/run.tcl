# Project root directory
set sfd [file dir [info script]]

# Create new solution
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11
solution options set /Input/CompilerFlags {{-D_GLIBCXX_USE_CXX11_ABI=0}}

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
go analyze
go compile

exit
