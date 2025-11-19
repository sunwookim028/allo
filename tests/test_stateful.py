# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, Stateful


def test(x: int32) -> int32:
    acc: Stateful[int32] = 1
    acc += x
    return acc


if __name__ == "__main__":
    # Test that Stateful type works
    sch = allo.customize(test)
    mlir_str = str(sch.module)
    
    # Verify stateful variable is in MLIR
    assert "_stateful_" in mlir_str, "Stateful variable not found in MLIR"
    print("✓ Stateful variable found in MLIR")
    
    # Test HLS translation
    from allo.backend.hls import HLSModule
    hls_mod = HLSModule(
        mod=mlir_str,
        top_func_name="test",
        platform="vitis_hls",
        func_args=[('x', 'i32')],
    )
    hls_code = hls_mod.hls_code
    
    # Verify static variable is in HLS output
    assert "static int32_t" in hls_code, "Static variable not found in HLS output"
    print("✓ Static variable found in HLS output")
    print("\nTest passed!")
