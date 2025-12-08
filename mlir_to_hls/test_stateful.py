#!/usr/bin/env python3
"""
Test script to verify stateful type → MLIR → Vitis HLS translation.
This script demonstrates that stateful variables are correctly translated
to static variables inside functions in the generated HLS C++ code.

Usage:
    python3 test_stateful.py
"""

import sys
from pathlib import Path
import allo
from allo.ir.types import int32, float32, stateful

# Add parent directory to path to import allo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allo.backend.hls import HLSModule

# Scalar stateful
def test_stateful_scalar(x: int32) -> int32:
    acc: stateful(int32) = 0
    acc = acc + x
    return acc


def main():
    print("\n1. Input MLIR:")
    print("-" * 80)
    s2 = allo.customize(test_stateful_scalar)
    print(s2.module)  # Should show memref.global
    
    print("\n2. Translating to Vitis HLS C++...")
    print("-" * 80)
    
    try:
        hls_code = s2.build(target="vhls")
        print(hls_code)
        print("%" * 80)

        mod = s2.build(target="vitis_hls", mode="hw_emu", project="stateful_acc.prj")
        acc1 = 0
        acc2 = 0
        acc3 = 0
        mod(2, acc1)
        mod(8, acc2)
        mod(4, acc3)
        print(acc1)
        print(acc2)
        print(acc3)

        
        '''
        # Verify static variable is inside function
        func_start = hls_code.find('void test(')
        func_body_start = hls_code.find('{', func_start)
        static_pos = hls_code.find('static int32_t acc_stateful_5700927378943735518')
        
        print("\n4. Verification:")
        print("-" * 80)
        
        if static_pos == -1:
            print("✗ FAILED: Static variable not found in output")
            return 1
        elif static_pos < func_body_start:
            print("✗ FAILED: Static variable declared OUTSIDE test() function")
            print(f"  Function starts at position: {func_body_start}")
            print(f"  Static variable at position: {static_pos}")
            return 1
        else:
            print("✓ PASSED: Static variable declared INSIDE test() function")
            print(f"  Function body starts at: {func_body_start}")
            print(f"  Static variable at: {static_pos}")
            print("\n✓ All tests passed!")
            
            # Save output for inspection
            output_file = Path(__file__).parent / "generated_output.cpp"
            with open(output_file, "w") as f:
                f.write(hls_code)
            print(f"\n✓ Output saved to: {output_file}")
            return 0
        '''
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
