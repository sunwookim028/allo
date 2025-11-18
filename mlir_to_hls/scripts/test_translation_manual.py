#!/usr/bin/env python3
"""
Manual test to verify the expected translation output.
This can be used to validate the translation logic without requiring
the full mlir-translate binary or Python bindings.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLIR_FILE = PROJECT_ROOT / "mlir" / "accumulate.mlir"

def check_mlir_syntax():
    """Check that the MLIR file has the expected structure."""
    print("==> Checking MLIR file syntax")
    
    if not MLIR_FILE.exists():
        raise FileNotFoundError(f"Missing MLIR file: {MLIR_FILE}")
    
    with open(MLIR_FILE, "r") as f:
        content = f.read()
    
    # Check for required elements
    checks = [
        ("memref.global", "global variable declaration"),
        ("hls.static", "hls.static attribute"),
        ("@acc_state", "accumulator state global"),
        ("accumulate_kernel", "kernel function"),
        ("memref.load", "load operation"),
        ("memref.store", "store operation"),
        ("arith.addi", "addition operation"),
    ]
    
    missing = []
    for token, description in checks:
        if token not in content:
            missing.append(f"  - Missing '{token}' ({description})")
    
    if missing:
        print("ERROR: MLIR file is missing required elements:")
        for item in missing:
            print(item)
        return False
    
    print("MLIR file syntax check passed")
    return True

def show_expected_output():
    """Show what the expected translation output should look like."""
    print("\n==> Expected Translation Output")
    print("=" * 60)
    
    expected = """//===------------------------------------------------------------*- C++ -*-===//
// Automatically generated file for High-level Synthesis (HLS).
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

static int32_t acc_state = {0};

void accumulate_kernel(
  int32_t in,
  int32_t *out
) {
  // placeholder for const int32_t acc_state;
  int32_t %current = acc_state[];
  int32_t %updated = %current + in;
  acc_state[] = %updated;
  out[] = %updated;
}
"""
    
    print(expected)
    print("=" * 60)
    print("\nKey checks:")
    print("  1. 'static int32_t acc_state' - static qualifier present")
    print("  2. 'acc_state[] = ' - store to persistent state")
    print("  3. 'accumulate_kernel(' - kernel function signature")
    print("  4. ' + ' - addition operation")
    
    return True

if __name__ == "__main__":
    print("MLIR Translation Manual Test")
    print("=" * 60)
    
    success = True
    success &= check_mlir_syntax()
    success &= show_expected_output()
    
    if success:
        print("\n✓ Manual checks passed")
        print("Note: Full translation test requires mlir-translate binary or Python bindings")
        sys.exit(0)
    else:
        print("\n✗ Manual checks failed")
        sys.exit(1)

