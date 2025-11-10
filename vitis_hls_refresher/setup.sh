#!/bin/bash
#=============================================================================
# Setup script for Vitis HLS Refresher Environment
#=============================================================================

echo "========================================="
echo "Vitis HLS Refresher - Setup"
echo "========================================="
echo ""

# Check if Vitis HLS is available
if command -v vitis_hls &> /dev/null; then
    echo "✅ Vitis HLS found: $(which vitis_hls)"
    VITIS_HLS_PATH=$(dirname $(dirname $(which vitis_hls)))
    echo "   Path: $VITIS_HLS_PATH"
else
    echo "⚠️  Vitis HLS not found in PATH"
    echo ""
    echo "Please either:"
    echo "  1. Source the Vitis HLS setup script:"
    echo "     source /path/to/Xilinx/Vitis_HLS/<version>/settings64.sh"
    echo ""
    echo "  2. Or set VITIS_HLS environment variable:"
    echo "     export VITIS_HLS=/path/to/vitis_hls"
    echo ""
    
    if [ -n "$VITIS_HLS" ]; then
        echo "Using VITIS_HLS environment variable: $VITIS_HLS"
        export PATH="$VITIS_HLS:$PATH"
    else
        echo "Please set up Vitis HLS and run this script again."
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Environment Setup Complete"
echo "========================================="
echo ""
echo "Quick Start:"
echo "  cd ex01_basic_static && make csim"
echo ""
echo "For help:"
echo "  cat README.md"
echo "  cat QUICK_REFERENCE.md"
echo ""

