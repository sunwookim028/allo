set ipdir [file join [file dirname [file normalize [info script]]] ip]
file mkdir $ipdir
create_project -in_memory -part xcu55c-fsvh2892-2L-e
set_property target_language Verilog [current_project]
if {![file exists $ipdir/add_f32_f32_f32_l2_core/add_f32_f32_f32_l2_core.xci]} {
  create_ip -name floating_point -vendor xilinx.com -library ip -module_name add_f32_f32_f32_l2_core -dir $ipdir
  set_property -dict [list CONFIG.Flow_Control NonBlocking CONFIG.Has_ACLKEN true CONFIG.Has_RESULT_TREADY false CONFIG.Operation_Type Add_Subtract CONFIG.C_Mult_Usage No_Usage CONFIG.Maximum_Latency false CONFIG.C_Latency 2] [get_ips add_f32_f32_f32_l2_core]
} else {
  read_ip $ipdir/add_f32_f32_f32_l2_core/add_f32_f32_f32_l2_core.xci
}
set_property generate_synth_checkpoint false [get_files $ipdir/add_f32_f32_f32_l2_core/add_f32_f32_f32_l2_core.xci]
generate_target synthesis [get_ips add_f32_f32_f32_l2_core]
if {![file exists $ipdir/cmp_f32_f32_u1_l1_core/cmp_f32_f32_u1_l1_core.xci]} {
  create_ip -name floating_point -vendor xilinx.com -library ip -module_name cmp_f32_f32_u1_l1_core -dir $ipdir
  set_property -dict [list CONFIG.Flow_Control NonBlocking CONFIG.Has_ACLKEN true CONFIG.Has_RESULT_TREADY false CONFIG.Operation_Type Compare CONFIG.C_Compare_Operation Programmable CONFIG.Maximum_Latency false CONFIG.C_Latency 1] [get_ips cmp_f32_f32_u1_l1_core]
} else {
  read_ip $ipdir/cmp_f32_f32_u1_l1_core/cmp_f32_f32_u1_l1_core.xci
}
set_property generate_synth_checkpoint false [get_files $ipdir/cmp_f32_f32_u1_l1_core/cmp_f32_f32_u1_l1_core.xci]
generate_target synthesis [get_ips cmp_f32_f32_u1_l1_core]
if {![file exists $ipdir/mul_f32_f32_f32_l2_core/mul_f32_f32_f32_l2_core.xci]} {
  create_ip -name floating_point -vendor xilinx.com -library ip -module_name mul_f32_f32_f32_l2_core -dir $ipdir
  set_property -dict [list CONFIG.Flow_Control NonBlocking CONFIG.Has_ACLKEN true CONFIG.Has_RESULT_TREADY false CONFIG.Operation_Type Multiply CONFIG.Maximum_Latency false CONFIG.C_Latency 2] [get_ips mul_f32_f32_f32_l2_core]
} else {
  read_ip $ipdir/mul_f32_f32_f32_l2_core/mul_f32_f32_f32_l2_core.xci
}
set_property generate_synth_checkpoint false [get_files $ipdir/mul_f32_f32_f32_l2_core/mul_f32_f32_f32_l2_core.xci]
generate_target synthesis [get_ips mul_f32_f32_f32_l2_core]
if {![file exists $ipdir/sub_f32_f32_f32_l2_core/sub_f32_f32_f32_l2_core.xci]} {
  create_ip -name floating_point -vendor xilinx.com -library ip -module_name sub_f32_f32_f32_l2_core -dir $ipdir
  set_property -dict [list CONFIG.Flow_Control NonBlocking CONFIG.Has_ACLKEN true CONFIG.Has_RESULT_TREADY false CONFIG.Operation_Type Add_Subtract CONFIG.C_Mult_Usage No_Usage CONFIG.Maximum_Latency false CONFIG.C_Latency 2] [get_ips sub_f32_f32_f32_l2_core]
} else {
  read_ip $ipdir/sub_f32_f32_f32_l2_core/sub_f32_f32_f32_l2_core.xci
}
set_property generate_synth_checkpoint false [get_files $ipdir/sub_f32_f32_f32_l2_core/sub_f32_f32_f32_l2_core.xci]
generate_target synthesis [get_ips sub_f32_f32_f32_l2_core]
