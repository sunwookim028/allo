#!/bin/bash
#=============================================================================
# Helper script: Quick comparison between static and non-static versions
#=============================================================================

EX_DIR=$1
VERSION_A=$2
VERSION_B=$3

if [ -z "$EX_DIR" ] || [ -z "$VERSION_A" ] || [ -z "$VERSION_B" ]; then
    echo "Usage: $0 <example_dir> <version_a> <version_b>"
    echo "Example: $0 ex02_static_variables top_a top_b"
    exit 1
fi

cd "$EX_DIR" || exit 1

echo "========================================="
echo "Comparing $VERSION_A vs $VERSION_B"
echo "========================================="

# Test version A
echo ""
echo "--- Testing $VERSION_A ---"
sed -i "s/set_top top_[a-z]/set_top $VERSION_A/" run.tcl
vitis_hls -f run.tcl > /tmp/hls_a.log 2>&1
grep -A 20 "Result\|expected" /tmp/hls_a.log

# Test version B
echo ""
echo "--- Testing $VERSION_B ---"
sed -i "s/set_top top_[a-z]/set_top $VERSION_B/" run.tcl
vitis_hls -f run.tcl > /tmp/hls_b.log 2>&1
grep -A 20 "Result\|expected" /tmp/hls_b.log

echo ""
echo "--- Differences ---"
diff /tmp/hls_a.log /tmp/hls_b.log | head -50

