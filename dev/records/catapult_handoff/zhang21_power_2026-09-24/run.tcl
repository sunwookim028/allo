options set Input/CppStandard c++11
project new
flow package require /SCVerify
flow package option set /SCVerify/USE_QUESTASIM false
flow package option set /SCVerify/USE_NCSIM true
flow package require /NCSim
flow package option set /NCSim/NC_ROOT /opt/cadence/XCELIUM2403
flow package require /LowPower
flow package option set /LowPower/SWITCHING_ACTIVITY_TYPE saif
flow package option set /SCVerify/USE_CCS_BLOCK true
solution file add ./mac.cpp
solution file add ./mac_tb.cpp -exclude true
go compile
solution library add nangate-45nm_beh
directive set -CLOCKS {clk {-CLOCK_PERIOD 5.0}}
go assembly
go extract
directive set USE_MODES {test}
directive set /USE_MODES/test/PWR_CLOCK_MODE {{clk default}}
directive set /USE_MODES/test/PWR_OPT_WEIGHT 1.0
go switching
flow run /PowerAnalysis/report_pre_pwropt_Verilog
exit
