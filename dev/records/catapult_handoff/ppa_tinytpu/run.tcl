# Project root directory
set sfd [file dir [info script]]

# Create new solution
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11
flow package require /SCVerify
flow package option set /SCVerify/USE_QUESTASIM false
flow package option set /SCVerify/USE_NCSIM true
flow package require /NCSim
flow package option set /NCSim/NC_ROOT /opt/cadence/XCELIUM2403
flow package require /LowPower
flow package option set /LowPower/SWITCHING_ACTIVITY_TYPE saif

# Add source files
solution file add "$sfd/kernel.cpp" -type C++
solution file add "$sfd/tinytpu_tb.cpp" -type C++ -exclude true

# Set top-level design function
directive set -DESIGN_HIERARCHY tinytpu_isa

# Set clock constraints
directive set -CLOCKS {clk {-CLOCK_PERIOD 5.0}}

# Set output language
solution options set /Output/OutputVerilog true
solution options set /Output/OutputVHDL false
solution library add nangate-45nm_beh

# Flow
go analyze
go compile

solution library add ccs_sample_mem
go assembly
go extract

# Power: annotate simulated activity, then report
directive set USE_MODES {test}
directive set /USE_MODES/test/PWR_CLOCK_MODE {{clk default}}
directive set /USE_MODES/test/PWR_OPT_WEIGHT 1.0
go switching
flow run /PowerAnalysis/report_pre_pwropt_Verilog

exit
