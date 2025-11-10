#=============================================================================
# TCL Script for Example 7: HLS Stream of Blocks
# Synthesizes both Version B (pointer-based) and Version D (stream-based)
# for performance comparison
#=============================================================================

set hls_prj ex07.prj

# Try to find Vitis HLS include path automatically
set vitis_inc ""
if {[info exists env(VITIS_HLS)]} {
    set vitis_inc "-I$env(VITIS_HLS)/include"
} elseif {[file exists "/opt/xilinx/Vitis_HLS/2023.2/include"]} {
    set vitis_inc "-I/opt/xilinx/Vitis_HLS/2023.2/include"
} elseif {[file exists "/opt/xilinx/Vitis/2023.2/data/system_compiler/include"]} {
    set vitis_inc "-I/opt/xilinx/Vitis/2023.2/data/system_compiler/include"
} elseif {[file exists "/opt/xilinx/Vitis_HLS/2022.2/include"]} {
    set vitis_inc "-I/opt/xilinx/Vitis_HLS/2022.2/include"
}

#=============================================================================
# Synthesize Version B (Pointer-based) - Baseline
#=============================================================================
puts "\n========================================="
puts "Synthesizing Version B (Pointer-based)"
puts "=========================================\n"

open_project ${hls_prj} -reset
open_solution -reset solution_b -flow_target vivado

add_files kernel.cpp
if {$vitis_inc != ""} {
    add_files -tb tb.cpp -cflags "-std=c++11 $vitis_inc"
} else {
    add_files -tb tb.cpp -cflags "-std=c++11"
}

set_top stream_blocks_b
open_solution "solution_b"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design
csynth_design
csynth_design
csynth_design
csynth_design
csynth_design
csynth_design

# Save report for comparison
file copy -force ${hls_prj}/solution_b/syn/report/stream_blocks_b_csynth.rpt ${hls_prj}/solution_b/syn/report/version_b_baseline.rpt

close_project

#=============================================================================
# Synthesize Version D (Stream-based) - Optimized
#=============================================================================
puts "\n========================================="
puts "Synthesizing Version D (Stream-based)"
puts "=========================================\n"

open_project ${hls_prj} -reset
open_solution -reset solution_d -flow_target vivado

add_files kernel.cpp
if {$vitis_inc != ""} {
    add_files -tb tb.cpp -cflags "-std=c++11 $vitis_inc"
} else {
    add_files -tb tb.cpp -cflags "-std=c++11"
}

set_top stream_blocks_d
open_solution "solution_d"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design
csynth_design
csynth_design
csynth_design
csynth_design
csynth_design
csynth_design

# Save report for comparison
file copy -force ${hls_prj}/solution_d/syn/report/stream_blocks_d_csynth.rpt ${hls_prj}/solution_d/syn/report/version_d_stream.rpt

close_project

#=============================================================================
# Print Comparison Summary
#=============================================================================
puts "\n========================================="
puts "Synthesis Comparison Summary"
puts "========================================="
puts ""
puts "Version B (Pointer-based) Report:"
puts "  ${hls_prj}/solution_b/syn/report/version_b_baseline.rpt"
puts ""
puts "Version D (Stream-based) Report:"
puts "  ${hls_prj}/solution_d/syn/report/version_d_stream.rpt"
puts ""
puts "To compare, run:"
puts "  ./compare_reports.sh"
puts ""

exit
