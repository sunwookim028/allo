#=========================================================================
# SRAM Probe Constraints
#=========================================================================

set clock_net  clk
set clock_name ideal_clock

create_clock -name ${clock_name} \
             -period ${clock_period} \
             [get_ports ${clock_net}]

set_load -pin_load $ADK_TYPICAL_ON_CHIP_LOAD [all_outputs]

set_driving_cell -no_design_rule \
  -lib_cell $ADK_DRIVING_CELL [all_inputs]

set_input_delay -clock ${clock_name} [expr ${clock_period}/2.0] [all_inputs]
set_output_delay -clock ${clock_name} 0 [all_outputs]

set_max_fanout 20 $design_name
set_max_transition [expr 0.25*${clock_period}] $design_name

