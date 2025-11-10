#!/bin/bash
#=============================================================================
# Helper script: Quick resource report viewer
#=============================================================================

PROJECT_DIR=$1

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 <project_directory>"
    echo "Example: $0 ex01_basic_static/ex01.prj"
    exit 1
fi

RPT_FILE="$PROJECT_DIR/solution1/syn/report/top_csynth.rpt"

if [ ! -f "$RPT_FILE" ]; then
    echo "Error: Synthesis report not found at $RPT_FILE"
    echo "Run 'make synth' first to generate the report"
    exit 1
fi

echo "========================================="
echo "Resource Usage Report"
echo "========================================="
echo ""

# Extract key sections
echo "--- Timing Summary ---"
grep -A 5 "Timing \(ns\):" "$RPT_FILE" | head -10

echo ""
echo "--- Resource Usage ---"
grep -E "BRAM|DSP|FF|LUT" "$RPT_FILE" | head -20

echo ""
echo "--- Initiation Interval ---"
grep -A 2 "Initiation Interval" "$RPT_FILE" | head -10

echo ""
echo "--- Latency ---"
grep -A 2 "Latency" "$RPT_FILE" | head -10

echo ""
echo "Full report: $RPT_FILE"

