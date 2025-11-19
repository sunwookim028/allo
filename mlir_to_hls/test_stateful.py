#!/usr/bin/env python3
"""
Test script to verify Stateful type → MLIR → Vitis HLS translation.
This script demonstrates that stateful variables are correctly translated
to static variables inside functions in the generated HLS C++ code.

Usage:
    python3 test_stateful.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import allo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allo.backend.hls import HLSModule

def main():
    print("=" * 80)
    print("TEST: Stateful Variable Translation to Static Inside Function")
    print("=" * 80)
    
    # MLIR with stateful variable (matches what Allo generates)
    mlir_content = """module {
  memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<1>
  func.func @test(%arg0: i32) -> i32 attributes {itypes = "s", otypes = "s"} {
    %0 = memref.get_global @acc_stateful_5700927378943735518 : memref<i32>
    %1 = memref.load %0[] : memref<i32>
    %2 = arith.addi %1, %arg0 : i32
    memref.store %2, %0[] : memref<i32>
    return %2 : i32
  }
}"""
    
    print("\n1. Input MLIR:")
    print("-" * 80)
    print(mlir_content)
    
    print("\n2. Translating to Vitis HLS C++...")
    print("-" * 80)
    
    try:
        hls_mod = HLSModule(
            mod=mlir_content,
            top_func_name="test",
            platform="vitis_hls",
            func_args=[('arg0', 'i32')],
        )
        hls_code = hls_mod.hls_code
        
        print("\n3. Generated HLS C++ Code:")
        print("-" * 80)
        print(hls_code)
        
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
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
