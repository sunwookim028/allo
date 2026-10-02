# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set sfd [file dir [info script]]
solution new -state initial
solution options defaults
solution options set /Input/CppStandard c++11
solution file add "$sfd/kernel.cpp" -type C++
directive set -DESIGN_HIERARCHY bf16_add_comb
directive set -CLOCKS {clk {-CLOCK_PERIOD 3.33}}
solution options set /Output/OutputVerilog true
solution options set /Output/OutputVHDL false
solution library add nangate-45nm_beh
go analyze

go compile
solution library add ccs_sample_mem
go assembly
go extract
exit
