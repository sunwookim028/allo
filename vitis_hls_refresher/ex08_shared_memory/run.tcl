#=============================================================================
# TCL Script for Example 8: Shared Static Memory
# Synthesizes both AXI-based and direct shared memory versions
# to demonstrate AXI overhead and interfacing choices
#=============================================================================

set hls_prj ex08.prj

#=============================================================================
# Synthesize Version 1: AXI-based (matmul_kernel)
# Demonstrates: AXI interface overhead for external memory access
#=============================================================================
puts "\n========================================="
puts "Synthesizing Version 1: AXI-based (matmul_kernel)"
puts "Interface: m_axi ports for external memory access"
puts "=========================================\n"

open_project ${hls_prj} -reset
open_solution -reset solution_axi -flow_target vivado

add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=c++11"

set_top matmul_shared
open_solution "solution_axi"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design -O
csynth_design

# Save report
file copy -force ${hls_prj}/solution_axi/syn/report/matmul_kernel_csynth.rpt ${hls_prj}/solution_axi/syn/report/version_axi.rpt

close_project

#=============================================================================
# Synthesize Version 2: Direct Shared Memory (matmul_shared)
# Demonstrates: Direct shared memory access without AXI overhead
#=============================================================================
puts "\n========================================="
puts "Synthesizing Version 2: Direct Shared Memory (matmul_shared)"
puts "Interface: s_axilite ports for offsets only"
puts "=========================================\n"

open_project ${hls_prj} -reset
open_solution -reset solution_shared -flow_target vivado

add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=c++11"

set_top matmul_shared
open_solution "solution_shared"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design -O
csynth_design

# Save report
file copy -force ${hls_prj}/solution_shared/syn/report/matmul_shared_csynth.rpt ${hls_prj}/solution_shared/syn/report/version_shared.rpt

close_project

#=============================================================================
# Print Summary
#=============================================================================
puts "\n========================================="
puts "Synthesis Comparison Summary"
puts "========================================="
puts ""
puts "Version 1 (AXI-based) Report:"
puts "  ${hls_prj}/solution_axi/syn/report/version_axi.rpt"
puts ""
puts "Version 2 (Direct Shared Memory) Report:"
puts "  ${hls_prj}/solution_shared/syn/report/version_shared.rpt"
puts ""
puts "To compare, run:"
puts "  ./compare_axi_overhead.sh"
puts ""

exit
