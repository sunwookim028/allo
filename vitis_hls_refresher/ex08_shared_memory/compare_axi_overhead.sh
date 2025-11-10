#!/bin/bash
#=============================================================================
# Compare AXI-based vs Direct Shared Memory Access
# Demonstrates AXI overhead and interfacing choices
#=============================================================================

PROJECT="ex08.prj"

# Find actual report files
REPORT_AXI=$(find ${PROJECT} -path "*solution_axi*" -name "*matmul_kernel*.rpt" | head -1)
REPORT_SHARED=$(find ${PROJECT} -path "*solution_shared*" -name "*matmul_shared*.rpt" | head -1)

if [ -z "$REPORT_AXI" ] && [ -z "$REPORT_SHARED" ]; then
    echo "Error: Synthesis reports not found."
    echo "Run 'make synth' first."
    find ${PROJECT} -name "*csynth*.rpt" -type f 2>/dev/null | head -5
    exit 1
fi

echo "========================================="
echo "AXI Overhead Comparison"
echo "========================================="
echo ""

if [ -n "$REPORT_AXI" ]; then
    echo "Version 1 (AXI-based): $REPORT_AXI"
else
    echo "Version 1 (AXI-based): Report not found (check synthesis log)"
fi

if [ -n "$REPORT_SHARED" ]; then
    echo "Version 2 (Direct Shared Memory): $REPORT_SHARED"
else
    echo "Version 2 (Direct Shared Memory): Report not found"
fi
echo ""

echo "--- Interface Types (from synthesis log) ---"
echo "Version 1 (AXI-based):"
echo "  - Uses m_axi ports for data transfer (W, X, result)"
echo "  - Requires AXI Master Interface logic"
grep -E "matmul_kernel.*m_axi|Setting interface.*matmul_kernel" synth_axi_comparison.log 2>/dev/null | head -3 || echo "  Check synthesis log for m_axi interface details"
echo ""
echo "Version 2 (Direct Shared Memory):"
echo "  - Uses s_axilite ports for offsets only"
echo "  - Direct access to shared_memory (BRAM)"
grep -E "matmul_shared.*s_axilite|Setting interface.*matmul_shared" synth_axi_comparison.log 2>/dev/null | head -5
echo ""

if [ -n "$REPORT_AXI" ]; then
    echo "--- Resource Usage: Version 1 (AXI-based) ---"
    grep -E "^BRAM|^DSP|^FF|^LUT" "$REPORT_AXI" 2>/dev/null | head -8 || echo "  Not found in standard format"
    echo ""
fi

if [ -n "$REPORT_SHARED" ]; then
    echo "--- Resource Usage: Version 2 (Direct Shared Memory) ---"
    grep -E "^BRAM|^DSP|^FF|^LUT" "$REPORT_SHARED" 2>/dev/null | head -8 || echo "  Not found in standard format"
    echo ""
fi

echo "--- Key Differences from Synthesis Log ---"
echo "Version 1 (AXI-based) generates:"
grep -E "m_axi.*ARVALID|m_axi.*AWVALID|m_axi.*WVALID" synth_axi_comparison.log 2>/dev/null | head -3 | sed 's/^/  /' || echo "  - AXI Master Interface with read/write channels"
echo ""
echo "Version 2 (Direct Shared Memory) generates:"
grep -E "s_axilite.*port|Bundling.*AXI-Lite" synth_axi_comparison.log 2>/dev/null | head -3 | sed 's/^/  /' || echo "  - AXI-Lite Slave Interface only (control)"
echo ""

echo "========================================="
echo "Key Insights:"
echo "  ✅ AXI interfaces add overhead (logic, latency, resources)"
echo "  ✅ Direct shared memory access is more efficient"
echo "  ✅ Trade-off: AXI allows external access, direct is internal only"
echo ""
echo "  Version 1: m_axi → External memory access (more overhead)"
echo "  Version 2: s_axilite → Internal shared memory (less overhead)"
echo "========================================="
