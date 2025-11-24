<<<<<<< HEAD
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
=======
import allo
from allo.ir.types import int32, float32, Stateful

# Scalar stateless
def test_stateless_scalar(x: int32) -> int32:
    acc: int32 = 0
    acc = acc + x
    return acc

# Scalar stateful
def test_stateful_scalar(x: int32) -> int32:
    acc: Stateful[int32] = 0
    acc = acc + x
    return acc

# Array stateless
def test_stateless_array(x: float32) -> float32:
    buffer: float32[10] = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]

# Array stateful
def test_stateful_array(x: float32) -> float32:
    buffer: Stateful[float32[10]] = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]

s1 = allo.customize(test_stateless_scalar)
print(s1.module)  # Should show alloc

s2 = allo.customize(test_stateful_scalar)
print(s2.module)  # Should show memref.global

s3 = allo.customize(test_stateless_array)
print(s3.module)  # Should show alloc with shape

s2 = allo.customize(test_stateful_array)
print(s2.module)  # Should show memref.global with shape
>>>>>>> c90a814e51c6447a50364e4722b331984e5b722f
