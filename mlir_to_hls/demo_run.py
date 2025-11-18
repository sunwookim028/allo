#!/usr/bin/env python3
"""
Demo script to translate MLIR to Vitis HLS C++ code.
"""

from pathlib import Path
from allo.backend.hls import HLSModule

# Read the test MLIR file
mlir_file = Path(__file__).parent / "test_input.mlir"
print("=" * 80)
print("MLIR TO VITIS HLS TRANSLATION DEMO")
print("=" * 80)
print(f"\nReading MLIR from: {mlir_file}\n")

with open(mlir_file, "r") as f:
    mlir_content = f.read()

print("INPUT MLIR PROGRAM:")
print("-" * 80)
print(mlir_content)
print("-" * 80)

# Translate to HLS
print("\nTranslating MLIR to Vitis HLS C++...\n")
try:
    mod = HLSModule(
        mod=mlir_content,
        top_func_name="test",
        platform="vivado_hls",
    )
    hls_code = mod.hls_code
    
    print("=" * 80)
    print("GENERATED VITIS HLS C++ CODE:")
    print("=" * 80)
    print(hls_code)
    print("=" * 80)
    
    # Save output
    output_file = Path(__file__).parent / "generated_output.cpp"
    with open(output_file, "w") as f:
        f.write(hls_code)
    print(f"\n✓ Translation successful!")
    print(f"✓ Output saved to: {output_file}")
    
except Exception as e:
    print(f"\n✗ Translation failed: {e}")
    import traceback
    traceback.print_exc()
