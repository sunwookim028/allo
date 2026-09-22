# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re
import shutil
import subprocess
import tempfile

import pytest
import allo
from allo.ir.types import bool, int32, uint32, float16, float32
from allo.memory import Memory
import numpy as np
import allo.backend.hls as hls
from allo.passes import generate_input_output_buffers


@pytest.mark.parametrize("flatten", [True, False])
def test_io_buffer_gemm(flatten):
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    allo.passes.generate_input_output_buffers(
        s.module, s.top_func_name, flatten=flatten
    )
    print(s.module)
    mod = s.build()
    if not flatten:
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
    else:
        np_A = np.random.randint(0, 10, size=(32 * 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32 * 32)).astype(np.int32)
        np_C = np.matmul(np_A.reshape(32, 32), np_B.reshape(32, 32)).reshape(32 * 32)
    np_C_allo = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
    print("Passed!")


def test_vitis_gemm():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        print(mod.hls_code)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((32, 32), dtype=np.int32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
            print("Passed!")


def test_vitis_gemm_template():
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[int32, 32, 32, 32])
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((32, 32), dtype=np.int32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
            print("Passed!")

    s = allo.customize(gemm, instantiate=[float32, 64, 64, 64])
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            np_A = np.random.random(size=(64, 64)).astype(np.float32)
            np_B = np.random.random(size=(64, 64)).astype(np.float32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((64, 64), dtype=np.float32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
            print("Passed!")


def test_vitis_io_stream():
    def foo(A: int32[32, 32], B: int32[32, 32]):
        pass

    def top(A: int32[32, 32]) -> int32[32, 32]:
        B: int32[32, 32]
        foo(A, B)
        return B

    s = allo.customize(top)
    s.dataflow("top")
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            print(s.module)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.zeros((32, 32), dtype=np.int32)
            hls_mod(np_A, np_B)


def test_csim_write_back():
    N = 256

    def compute(x: int32[N], y: int32[N]):
        for i in range(N):
            y[i] = x[i]

    s = allo.customize(compute)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
            A = np.random.randint(0, 10, size=(N)).astype(np.int32)
            B = np.zeros((N), dtype=np.int32)
            mod(A, B)
            np.testing.assert_allclose(A, B, rtol=1e-5)
            print("Passed!")


def test_pointer_generation():
    def top(inst: bool, C: int32[3]):
        if inst:
            C[0] = C[0] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "bool v" in mod.hls_code and ",," not in mod.hls_code
        if hls.is_available("vitis_hls"):
            inst = np.array([1], dtype=np.bool_)
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(inst, C)
            np.testing.assert_allclose(C, [2, 2, 3], rtol=1e-5)
            print("Passed!")


def test_scalar_not_array():
    def top(inst: bool, C: int32[3]):
        flag: bool = inst[0]
        if flag:
            C[0] = C[0] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "bool v" in mod.hls_code and ",," not in mod.hls_code
        if hls.is_available("vitis_hls"):
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(1, C)
            np.testing.assert_allclose(C, [2, 2, 3], rtol=1e-5)
            print("Passed!")


def test_scalar():
    def case1(C: int32) -> int32:
        return C + 1

    s = allo.customize(case1)
    mod = s.build()
    assert mod(1) == 2
    print("Passed CPU simulation!")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        assert "int32_t *v1" in mod.hls_code
        # Note: Should not expect it to run using csim! Need to generate correct binding for mutable scalars in PyBind.


def test_size1_array():
    def top(A: int32[1]) -> int32[1]:
        A[0] = A[0] + 1
        return A

    s = allo.customize(top)
    mod = s.build()
    np_A = np.array([1], dtype=np.int32)
    np.testing.assert_allclose(mod(np_A), [2], rtol=1e-5)
    print("Passed CPU simulation!")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="csim", project=tmpdir)
        print(mod.hls_code)
        assert "[1]" in mod.hls_code
        if hls.is_available("vitis_hls"):
            np_B = np.array([0], dtype=np.int32)
            mod(np_A, np_B)
            np.testing.assert_allclose(np_A, [2], rtol=1e-5)
            print("Passed!")


@pytest.mark.parametrize("flatten", [True, False])
def test_wrap_nonvoid(flatten):
    M, N = 4, 4

    def matrix_add(A: float32[M, N]) -> float32[M, N]:
        B: float32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(matrix_add)
    generate_input_output_buffers(s.module, "matrix_add", flatten=flatten)
    module = str(s.module)

    if flatten:
        # Top Function Argument
        assert (
            f"func.func @matrix_add(%arg0: memref<16xf32>) -> memref<16xf32>" in module
        )
        # Movement Function Generation
        assert (
            f"func.func @load_buf0(%arg0: memref<16xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        assert (
            f"func.func @store_res1(%arg0: memref<4x4xf32>, %arg1: memref<16xf32>)"
            in module
        )
        # Buffer Allocation
        assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<4x4xf32>' in module
        # Function Call
        assert (
            f"call @load_buf0(%arg0, %alloc) : (memref<16xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        assert (
            f"call @store_res1(%alloc_1, %alloc_0) : (memref<4x4xf32>, memref<16xf32>) -> ()"
            in module
        )
        # Return Value Allocation
        assert f'%alloc_0 = memref.alloc() {{name = "res1"}} : memref<16xf32>' in module
        # ReturnOP Update
        assert f"return %alloc_0 : memref<16xf32>" in module
    else:
        # Top Function Argument
        assert (
            f"func.func @matrix_add(%arg0: memref<4x4xf32>) -> memref<4x4xf32>"
            in module
        )
        # Movement Function Generation
        assert (
            f"func.func @load_buf0(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        assert (
            f"func.func @store_res1(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>)"
            in module
        )
        # Buffer Allocation
        assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<4x4xf32>' in module
        # Function Call
        assert (
            f"call @load_buf0(%arg0, %alloc) : (memref<4x4xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        assert (
            f"call @store_res1(%alloc_1, %alloc_0) : (memref<4x4xf32>, memref<4x4xf32>) -> ()"
            in module
        )
        # Return Value Allocation
        assert (
            f'%alloc_0 = memref.alloc() {{name = "res1"}} : memref<4x4xf32>' in module
        )
        # ReturnOP Update
        assert f"return %alloc_0 : memref<4x4xf32>" in module

    print("Passed!")


def test_wrap_io_linearized_index():
    M, N = 4, 4

    def matrix_copy(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N, name="copy"):
            B[i, j] = A[i, j]
        return B

    s = allo.customize(matrix_copy)

    # Test wrap_io=True
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=True)
        hls_code = mod.hls_code

        # Check that function arguments are declared as pointers
        assert (
            "int32_t *v" in hls_code
        ), "Expected pointer declaration for function arguments with wrap_io=True"

        # Check that helper functions (load_buf/store_res) exist for data movement
        assert (
            "load_buf0" in hls_code
        ), "Expected load_buf0 helper function with wrap_io=True"
        assert (
            "store_res1" in hls_code
        ), "Expected store_res1 helper function with wrap_io=True"

        # Check that linearized index pattern exists in helper functions
        # Pattern: ((var * 4) + var) for 4x4 array
        assert (
            "* 4) +" in hls_code
        ), "Expected linearized index pattern '* 4) +' in helper functions"
        print("wrap_io=True: All specific assertions passed!")

    # Test wrap_io=False
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_no_wrap = s.build(
            target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=False
        )
        hls_code_no_wrap = mod_no_wrap.hls_code

        # Check that function arguments are pointers with direct linearized access
        assert (
            "int32_t *v" in hls_code_no_wrap
        ), "Expected pointer declaration with wrap_io=False"

        # Check for direct linearized indexing pattern in main kernel
        # Pattern: v0[(i) * 4 + (j)] for direct pointer access
        assert (
            ") * 4 + (" in hls_code_no_wrap
        ), "Expected direct linearized index pattern ') * 4 + (' with wrap_io=False"

        # Verify no helper functions are generated
        assert (
            "load_buf0" not in hls_code_no_wrap
        ), "Should not have load_buf0 helper with wrap_io=False"
        print("wrap_io=False: All specific assertions passed!")

    print("Passed!")


# Module-level kernel functions for Memory HLS tests
# (Functions need to be at module level for proper AST parsing)
_MemUram = Memory(resource="URAM")
_MemBram2P = Memory(resource="BRAM", storage_type="RAM_2P")
_MemBram = Memory(resource="BRAM")
_MemLutram = Memory(resource="LUTRAM")


def _kernel_uram(a: int32[32] @ _MemUram) -> int32[32]:
    """Kernel with URAM memory annotation."""
    b: int32[32]
    for i in range(32):
        b[i] = a[i] + 1
    return b


def _kernel_bram_2p(a: float32[16, 16] @ _MemBram2P) -> float32[16, 16]:
    """Kernel with BRAM RAM_2P memory annotation."""
    b: float32[16, 16]
    for i, j in allo.grid(16, 16):
        b[i, j] = a[i, j] * 2.0
    return b


def _kernel_multi_mem(
    a: int32[32] @ _MemBram, b: int32[32] @ _MemUram, c: int32[32] @ _MemLutram
):
    """Kernel with multiple memory annotations."""
    for i in range(32):
        c[i] = a[i] + b[i]


def _kernel_local_mem(a: int32[32]) -> int32[32]:
    """Kernel with local variable using Memory annotation."""
    # Local buffer with URAM annotation
    buf: int32[32] @ _MemUram
    for i in range(32):
        buf[i] = a[i] * 2
    b: int32[32]
    for i in range(32):
        b[i] = buf[i] + 1
    return b


def _kernel_local_bram(a: float32[16, 16]) -> float32[16, 16]:
    """Kernel with local variable using BRAM RAM_2P annotation."""
    # Local buffer with BRAM RAM_2P annotation
    temp: float32[16, 16] @ _MemBram2P
    for i, j in allo.grid(16, 16):
        temp[i, j] = a[i, j] + 1.0
    b: float32[16, 16]
    for i, j in allo.grid(16, 16):
        b[i, j] = temp[i, j] * 2.0
    return b


def test_memory_uram_hls():
    """Test kernel with URAM Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_uram)
    print("=== MLIR Module (URAM) ===")
    print(s.module)

    # Check the memory space is in the memref type
    mlir_str = str(s.module)
    # URAM = impl_code 2, no storage = 0 -> memory_space = 32
    assert "32 : i32" in mlir_str, "Memory space 32 (URAM) should be in MLIR"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with URAM
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=uram" in mod.hls_code


def test_memory_bram_2p_hls():
    """Test kernel with BRAM RAM_2P Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_bram_2p)
    print("=== MLIR Module (BRAM RAM_2P) ===")
    print(s.module)

    # Check the memory space is in the memref type
    mlir_str = str(s.module)
    # BRAM = 1, RAM_2P = 2 -> memory_space = 1*16 + 2 = 18
    assert "18 : i32" in mlir_str, "Memory space 18 (BRAM+RAM_2P) should be in MLIR"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with BRAM and RAM_2P
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=bram" in mod.hls_code
    assert "type=ram_2p" in mod.hls_code


def test_multiple_memory_hls():
    """Test kernel with multiple Memory annotations generates multiple pragmas."""
    s = allo.customize(_kernel_multi_mem)
    print("=== MLIR Module (Multiple Memory) ===")
    print(s.module)

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Count bind_storage pragmas - should have 3 (BRAM, URAM, LUTRAM)
    pragma_count = mod.hls_code.count("#pragma HLS bind_storage")
    assert pragma_count == 3, f"Expected 3 bind_storage pragmas, got {pragma_count}"

    # Check all implementation types are present
    assert "impl=bram" in mod.hls_code
    assert "impl=uram" in mod.hls_code
    assert "impl=lutram" in mod.hls_code


def test_memory_local_variable_uram():
    """Test local variable with URAM Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_local_mem)
    print("=== MLIR Module (Local URAM) ===")
    print(s.module)

    # Check the memory space is in the memref type for the local buffer
    mlir_str = str(s.module)
    # URAM = 2, no storage = 0 -> memory_space = 32
    assert (
        "32 : i32" in mlir_str
    ), "Memory space 32 (URAM) should be in MLIR for local buffer"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with URAM for the local buffer
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=uram" in mod.hls_code


def test_memory_local_variable_bram():
    """Test local variable with BRAM RAM_2P Memory annotation generates bind_storage pragma."""
    s = allo.customize(_kernel_local_bram)
    print("=== MLIR Module (Local BRAM RAM_2P) ===")
    print(s.module)

    # Check the memory space is in the memref type for the local buffer
    mlir_str = str(s.module)
    # BRAM = 1, RAM_2P = 2 -> memory_space = 18
    assert (
        "18 : i32" in mlir_str
    ), "Memory space 18 (BRAM+RAM_2P) should be in MLIR for local buffer"

    # Build HLS code
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check for bind_storage pragma with BRAM and RAM_2P for the local buffer
    assert "#pragma HLS bind_storage variable=" in mod.hls_code
    assert "impl=bram" in mod.hls_code
    assert "type=ram_2p" in mod.hls_code


def test_ihls():
    def top(A: int32[1]) -> int32[1]:
        A[0] = A[0] + 1
        return A

    s = allo.customize(top)
    mod = s.build(target="ihls")
    assert "h.single_task<Top>([=]() [[intel::kernel_args_restrict]]" in mod.hls_code


def test_while_basic():
    """Test basic while loop support in HLS backend."""

    def kernel(A: int32[10]):
        i: int32 = 0
        while i < 10:
            A[i] = i
            i += 1

    s = allo.customize(kernel)
    print("=== MLIR Module ===")
    print(s.module)

    # Build for vhls target
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check that while loop is generated
    assert "while (true)" in mod.hls_code or "while (" in mod.hls_code
    assert "break" in mod.hls_code
    print("test_while_basic passed!")


def test_while_with_array():
    """Test while loop with array operations."""

    def kernel(A: int32[10], B: int32[10]):
        i: int32 = 0
        while i < 10:
            B[i] = A[i] * 2
            i += 1

    s = allo.customize(kernel)
    print("=== MLIR Module ===")
    print(s.module)

    # Build for vhls target
    mod = s.build(target="vhls")
    print("\n=== HLS Code ===")
    print(mod.hls_code)

    # Check that while loop is generated
    assert "while (true)" in mod.hls_code or "while (" in mod.hls_code
    assert "break" in mod.hls_code
    print("test_while_with_array passed!")


def test_wrap_io_false_nested_function():
    """Test that wrap_io=False does not flatten array indexing in nested functions."""
    M, N = 4, 8

    def top(A: "float32[M * N]", B: "float32[M * N]"):
        C: float32[M, N]
        inner(A, B, C)

    def inner(A: "float32[M * N]", B: "float32[M * N]", C: "float32[M, N]"):
        for m, n in allo.grid(M, N):
            C[m, n] = A[m * N + n]
        for m, n in allo.grid(M, N):
            B[m * N + n] = C[m, n]

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir, wrap_io=False)
        hls_code = mod.hls_code
        print("\n=== Generated HLS Code ===")
        print(hls_code)
        # Check that inner function uses 2D indexing for C
        assert (
            "v2[(m1) * 8 + (n1)]" not in hls_code
        ), "C should use 2D indexing in inner function"
        assert "v2[m1][n1]" in hls_code, "C should use 2D indexing in inner function"
        print("test_wrap_io_false_nested_function passed!")


def test_wrap_io_false_nested_function_2D():
    M, N = 4, 8

    def inner(A: float32[M, N], B: float32[M, N]):
        for m, n in allo.grid(M, N):
            B[m, n] = A[m, n]

    def top(A: float32[M, N], B: float32[M, N]):
        inner(A, B)

    s = allo.customize(top)
    with pytest.raises(RuntimeError):
        s.build(target="vitis_hls", mode="sw_emu", project="", wrap_io=False)


def test_floordiv():
    def floordiv(A: int32[10], B: int32[10]) -> int32[10]:
        C: int32[10] = 0
        for i in range(10):
            C[i] = A[i] // B[i]
        return C

    s = allo.customize(floordiv)
    print(s.module)
    # CPU simulation
    mod = s.build()
    np_A = np.random.randint(1, 10, size=(10,)).astype(np.int32)
    np_B = np.random.randint(1, 10, size=(10,)).astype(np.int32)
    np_C = np_A // np_B
    np_C_allo = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)

    # Vitis HLS code generation
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        hls_code = mod.hls_code
        print(hls_code)
        # Check if the division operator is used for FloorDivSIOp
        assert " / " in hls_code


def test_fp16_half_type_and_hls_math():
    def top(A: float16[8]) -> float16[8]:
        B: float16[8]
        for i in range(8):
            B[i] = allo.exp(A[i]) + allo.sqrt(A[i])
        return B

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        hls_code = mod.hls_code
        assert "half" in hls_code
        assert "hls::exp" in hls_code
        assert "hls::sqrt" in hls_code


def _vitis_include_dir():
    """Directory holding Vitis HLS' ``ap_int.h``, or None if unavailable."""
    roots = [os.environ.get("XILINX_HLS"), os.environ.get("XILINX_VITIS")]
    candidates = [os.path.join(r, "include") for r in roots if r]
    candidates += sorted(
        glob.glob("/opt/xilinx/Vitis_HLS/*/include")
        + glob.glob("/tools/Xilinx/Vitis_HLS/*/include"),
        reverse=True,
    )
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "ap_int.h")):
            return cand
    return None


def test_generated_names_do_not_collide():
    """The emitter's two name generators share one namespace.

    Explicit names (``loop_name``, function ``inputs``/``outputs``) and the
    default ``v%d`` names used to be tracked separately, so naming a loop
    variable ``v1`` handed ``v1`` out twice: the loop counter shadowed the
    array parameter the emitter had already called ``v1``, and the body then
    read ``v1[v1] = ...``, which does not compile (HLS 207-3746).
    """

    def kernel(A: int32[8], B: int32[8]):
        for v1 in allo.grid(8):
            B[v1] = A[v1] + 1

    s = allo.customize(kernel)
    hls_code = str(s.build(target="vhls"))
    print(hls_code)

    params = set(re.findall(r"^\s+\w+ (\w+)\[\d+\],?$", hls_code, re.M))
    loop_vars = set(re.findall(r"for \(int (\w+) = ", hls_code))
    assert params and loop_vars, hls_code
    assert not params & loop_vars, (
        "loop variable shadows a parameter of the same function: "
        f"{sorted(params & loop_vars)}\n{hls_code}"
    )

    inc = _vitis_include_dir()
    if inc is None or shutil.which("g++") is None:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "kernel.cpp")
        with open(src, "w", encoding="utf-8") as f:
            f.write(hls_code)
        subprocess.run(
            ["g++", "-w", "-fsyntax-only", "-std=c++14", f"-I{inc}", src], check=True
        )


def test_bit_slice_is_unsigned():
    """A bit slice must reach HLS as an *unsigned* field.

    ``x[lo:hi]`` is typed ``UInt(hi - lo)`` by the type inferencer, lowered
    with logical (zero-filling) shifts, and widened with ``arith.extui``, so
    the LLVM simulator reads an 8-bit field holding 200 back as 200.  The
    Vivado emitter used to declare the same slice signed (``int8_t`` /
    ``ap_int<8>``), which reads it back as -56: silent RTL/simulator
    divergence that no functional test on the simulator could catch.
    """

    def kernel(A: uint32[4], B: int32[4]):
        for i in range(4):
            # 8-bit field; 200 and 255 have the field's top bit set.
            B[i] = A[i][0:8]

    s = allo.customize(kernel)

    np_A = np.array([200, 64, 255, 1], dtype=np.uint32)
    golden = (np_A & 0xFF).astype(np.int32)

    # Simulator path.
    sim_B = np.zeros(4, dtype=np.int32)
    s.build()(np_A, sim_B)
    np.testing.assert_array_equal(sim_B, golden)

    # HLS path.  The sliced operand is 32-bit, so the only 8-bit scalar
    # declaration in the generated code is the slice result itself.
    hls_code = str(s.build(target="vhls"))
    print(hls_code)
    assert re.search(r"\buint8_t v\d+;", hls_code), (
        "bit slice was not declared unsigned:\n" + hls_code
    )
    assert not re.search(r"\bint8_t v\d+;", hls_code), (
        "bit slice declared as a signed 8-bit type; an 8-bit field holding "
        "200 reads back as -56 in RTL while the simulator reads 200:\n" + hls_code
    )

    # When Vitis' ap_int headers are installed, compile and run the emitted
    # device code so the two paths are compared for real.
    inc = _vitis_include_dir()
    if inc is None or shutil.which("g++") is None:
        return
    harness = (
        hls_code
        + """
#include <cstdio>
int main() {
  uint32_t A[4] = {200, 64, 255, 1};
  int32_t B[4] = {0, 0, 0, 0};
  kernel(A, B);
  printf("%d %d %d %d\\n", B[0], B[1], B[2], B[3]);
  return 0;
}
"""
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "kernel.cpp")
        exe = os.path.join(tmpdir, "kernel")
        with open(src, "w", encoding="utf-8") as f:
            f.write(harness)
        subprocess.run(
            ["g++", "-w", "-std=c++14", f"-I{inc}", src, "-o", exe], check=True
        )
        out = subprocess.run(
            [exe], check=True, capture_output=True, text=True
        ).stdout.split()
    np.testing.assert_array_equal(np.array(out, dtype=np.int32), golden)


def _loop_body(code, header_re):
    """The lines of the loop whose header matches `header_re`, header first,
    up to (not including) the next `for (`."""
    lines = code.splitlines()
    i = next(k for k, l in enumerate(lines) if re.search(header_re, l))
    body = [lines[i]]
    for l in lines[i + 1 :]:
        if "for (" in l:
            break
        body.append(l)
    return body


def test_dependence_pragma():
    """`s.dependence` emits `#pragma HLS dependence` inside the named loop,
    for a local buffer (by its name) and for an argument (by its C name)."""

    def kernel(A: int32[16], B: int32[16], n: int32[1]):
        buf: int32[16] = 0
        for i in range(n[0]):
            buf[A[i]] = buf[A[i]] + B[i]
        for j in range(16):
            B[j] = buf[j]

    s = allo.customize(kernel)
    s.dependence(
        "i",
        "buf",
        dep_type="inter",
        dependent=False,
        because="the caller never repeats an index in A",
    )
    s.dependence(
        "kernel:i",
        "B",
        direction="RAW",
        distance=4,
        dependent=True,
        dep_class="array",
        because="B is read here and written only in the next loop",
    )
    code = str(s.build(target="vhls"))
    print(code)
    # B is the second argument; its emitted name is whatever the signature says.
    b_name = re.search(
        r"void kernel\(\s*int32_t \w+\[16\],\s*int32_t (\w+)\[16\]", code
    )
    assert b_name, code
    body = _loop_body(code, r"for \(int \w+ = 0; \w+ < \w+; \w+ \+= 1\)")
    assert "#pragma HLS dependence variable=buf inter false" in "\n".join(body), body
    assert (
        f"#pragma HLS dependence variable={b_name.group(1)} array inter RAW "
        "distance=4 true" in "\n".join(body)
    ), body
    # Only the loop it was applied to carries it.
    assert code.count("#pragma HLS dependence") == 2

    # The simulator ignores the pragma: same answer either way.
    np_A = np.array([0, 1, 0, 2] * 4, dtype=np.int32)
    np_B = np.arange(16, dtype=np.int32)
    gold = np.zeros(16, dtype=np.int32)
    for i in range(16):
        gold[np_A[i]] += np_B[i]
    s.build()(np_A, np_B, np.array([16], dtype=np.int32))
    np.testing.assert_array_equal(np_B, gold)

    inc = _vitis_include_dir()
    if inc is not None and shutil.which("g++") is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "kernel.cpp")
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            subprocess.run(
                ["g++", "-w", "-fsyntax-only", "-std=c++14", f"-I{inc}", src],
                check=True,
            )


def test_pipeline_style():
    """`s.pipeline(..., style=)` adds Vitis's pipeline control style to the
    pragma; without it the pragma is unchanged, and a bad style is refused."""

    def kernel(A: int32[16], B: int32[16]):
        for i in range(16):
            B[i] = A[i] + 1
        for j in range(16):
            A[j] = B[j] * 2

    s = allo.customize(kernel)
    s.pipeline("i", style="flp")
    s.pipeline("j")
    code = str(s.build(target="vhls"))
    assert "#pragma HLS pipeline II=1 style=flp" in code, code
    assert code.count("style=") == 1, code
    assert "#pragma HLS pipeline II=1\n" in code, code
    with pytest.raises(Exception, match="stp/flp/frp"):
        allo.customize(kernel).pipeline("i", style="fast")


def test_pipeline_style_is_refused_where_no_emitter_writes_it():
    """Only the Vivado/Vitis emitter writes `style=`. A style is set to stop an
    RTL deadlock, so an emitter that would drop it refuses the build and names
    the loop, rather than producing RTL with the tool's default style."""

    def kernel(A: int32[16], B: int32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    for target in ("catapult", "ihls", "tapa"):
        s = allo.customize(kernel)
        s.pipeline("i", style="flp")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match=r"i \(style=flp\)"):
                s.build(target=target, mode="csyn", project=tmpdir)

    # An unstyled pipeline reaches every backend as before.
    s = allo.customize(kernel)
    s.pipeline("i")
    with tempfile.TemporaryDirectory() as tmpdir:
        s.build(target="catapult", mode="csyn", project=tmpdir)


def test_dependence_pragma_rejects_bad_claims():
    def kernel(A: int32[16]):
        for i in range(16):
            tmp: int32[4] = 0
            tmp[i % 4] = A[i]
            A[i] = tmp[(i + 1) % 4]

    s = allo.customize(kernel)
    with pytest.raises(Exception, match="inter/intra"):
        s.dependence("i", "A", dep_type="across")
    with pytest.raises(Exception, match="RAW/WAR/WAW"):
        s.dependence("i", "A", direction="RWA")
    with pytest.raises(Exception, match="distance"):
        s.dependence("i", "A", distance=2)  # a distance on a false claim
    with pytest.raises(Exception, match="declared inside the loop"):
        s.dependence("i", "tmp")


def test_dependence_pragma_dataflow_region():
    """Reachable on a dataflow region the way `s.partition` is: through
    `allo.dataflow.customize`, naming the kernel instance's loop."""
    import allo.dataflow as df
    from allo.ir.types import Stream

    @df.region()
    def top(X: int32[8], Y: int32[8]):
        q: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[X])
        def prod(lx: int32[8]):
            for i in range(8):
                q.put(lx[i])

        @df.kernel(mapping=[1], args=[Y])
        def cons(ly: int32[8]):
            acc: int32[8] = 0
            n: int32 = 8
            for x in range(n):
                v: int32 = q.get()
                acc[v & 7] = acc[v & 7] + v
            for j in range(8):
                ly[j] = acc[j]

    s = df.customize(top)
    s.dependence(
        "cons_0:x",
        "acc",
        dep_type="inter",
        dependent=False,
        because="the producer never puts the same value twice in a row",
    )
    code = str(s.build(target="vhls"))
    print(code)
    cons = code[code.index("void cons_0(") :]
    body = _loop_body(cons, r"for \(int \w+ = 0; \w+ < \w+; \w+ \+= 1\)")
    assert "#pragma HLS dependence variable=acc inter false" in "\n".join(body), body
    assert code.count("#pragma HLS dependence") == 1


def _vitis_top_signature(code, top):
    """The argument list of the emitted `extern "C"` top function."""
    i = code.index(f"void {top}(")
    return code[i : code.index(") {", i)]


def test_align_value_attribute():
    """`configs={"align_value": N}` puts `__attribute__((align_value(N)))` on
    every `m_axi` pointer of the vitis_hls top, and on nothing else.

    It is a *promise* to Vitis, not a fact it checks: a false one produces wrong
    RTL while every software simulation still passes, so its emission is worth
    asserting. See `docs/source/backends/vitis.rst`."""

    def vadd(A: int32[16], B: int32[16], n: int32):
        for i in range(16):
            B[i] = A[i] + n

    s = allo.customize(vadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(
            target="vitis_hls",
            mode="csim",
            project=tmpdir,
            configs={"align_value": 64},
        )
        code = mod.hls_code
    sig = _vitis_top_signature(code, "vadd")
    attr = "__attribute__((align_value(64)))"
    # One per array argument, which is exactly what becomes an m_axi port.
    ports = re.findall(r"#pragma HLS interface m_axi port=(\w+)", code)
    assert len(ports) == 2, code
    for port in ports:
        assert re.search(rf"\*{re.escape(attr)} {port}\b", sig), sig
    assert sig.count(attr) == len(ports), sig
    # The scalar argument is not a pointer, so it carries no alignment promise.
    scalar = [
        ln for ln in sig.splitlines() if ln.strip() and "*" not in ln and "(" not in ln
    ]
    assert scalar, sig
    assert all(attr not in ln for ln in scalar), scalar


def test_align_value_absent_by_default():
    """No `align_value` key, no attribute: it must be opt-in, because the HOST
    is the one that has to keep the promise."""

    def vadd(A: int32[16], B: int32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    for configs in (None, {}, {"align_value": None}):
        s = allo.customize(vadd)
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(
                target="vitis_hls", mode="csim", project=tmpdir, configs=configs
            )
            code = mod.hls_code
        assert "align_value" not in code, (configs, code)
        assert "#pragma HLS interface m_axi" in code, code


if __name__ == "__main__":
    pytest.main([__file__])
