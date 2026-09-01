"""Direct-TOSA checks for TinyTPU, independent of torch-mlir."""

from __future__ import annotations

import argparse

import numpy as np

from .isa import tpu


def _compile_adds(n_inputs: int):
    args = ", ".join(f"%a{i}: tensor<8xf32>" for i in range(n_inputs))
    lines = [
        f"    %p{i} = tosa.add %a{2 * i}, %a{2 * i + 1} : "
        "(tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>"
        for i in range(n_inputs // 2)
    ]
    for i in range(1, n_inputs // 2):
        lhs = "%p0" if i == 1 else f"%s{i - 1}"
        lines.append(
            f"    %s{i} = tosa.add {lhs}, %p{i} : "
            "(tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>"
        )
    result = "%p0" if n_inputs == 2 else f"%s{n_inputs // 2 - 1}"
    source = "\n".join(
        [
            "module {",
            f"  func.func @main({args}) -> tensor<8xf32> {{",
            *lines,
            f"    return {result} : tensor<8xf32>",
            "  }",
            "}",
        ]
    )
    return tpu.compile_program(source)


def _check(n_inputs: int, expected_instructions: int, expected_vstores: int):
    xs = [np.full(8, i, dtype=np.float32) for i in range(n_inputs)]
    prog = _compile_adds(n_inputs)
    assert len(prog.emits) == expected_instructions
    assert sum(e.name == "vstore" for e in prog.emits) == expected_vstores
    np.testing.assert_allclose(prog(*xs), np.sum(xs, axis=0))
    print(
        f"{n_inputs}-input TOSA add: {len(prog.emits)} instructions, "
        f"{expected_vstores} vstore(s): PASS"
    )


def _check_negate() -> None:
    source = """module {
  func.func @main(%a: tensor<8xf32>) -> tensor<8xf32> {
    %zp = "tosa.const"() {values = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.negate %a, %zp, %zp : (tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}"""
    x = np.arange(-4, 4, dtype=np.float32)
    prog = tpu.compile_program(source)
    assert [emit.name for emit in prog.emits] == [
        "dma_load", "vload", "vneg", "vstore", "dma_store"
    ]
    np.testing.assert_allclose(prog(x), -x)
    print(f"TOSA negate: {len(prog.emits)} instructions; output matches NumPy: PASS")


def _gemm_source(M: int, K: int, N: int) -> str:
    def tensor(*shape: int) -> str:
        return f"tensor<{'x'.join(map(str, shape))}xf32>"

    return f"""module {{
  func.func @main(%a: {tensor(1, M, K)}, %b: {tensor(1, K, N)}) -> {tensor(1, M, N)} {{
    %zp = \"tosa.const\"() {{values = dense<0.000000e+00> : tensor<1xf32>}} : () -> tensor<1xf32>
    %c = tosa.matmul %a, %b, %zp, %zp : ({tensor(1, M, K)}, {tensor(1, K, N)}, tensor<1xf32>, tensor<1xf32>) -> {tensor(1, M, N)}
    return %c : {tensor(1, M, N)}
  }}
}}"""


def _check_tiled_gemm(M: int, K: int, N: int) -> None:
    rng = np.random.default_rng(M * 10000 + K * 100 + N)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    b = rng.standard_normal((1, K, N)).astype(np.float32)
    prog = tpu.compile_program(_gemm_source(M, K, N))
    names = [emit.name for emit in prog.emits]
    n_tiles = (M // 4) * (K // 4) * (N // 4)
    assert names.count("matmul") == n_tiles
    assert names.count("vadd") == (M // 4) * (N // 4) * (K // 4 - 1) * 2
    np.testing.assert_allclose(prog(a, b), a @ b, rtol=1e-4, atol=1e-4)
    print(
        f"TOSA GEMM {M}x{K}x{N}: {len(names)} tiled ISA instructions; "
        "tiled output matches NumPy: PASS"
    )


def _check_gemm_tail_rejected() -> None:
    try:
        tpu.compile_program(_gemm_source(6, 8, 8))
    except AssertionError as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("non-divisible GEMM unexpectedly compiled")
    print("TOSA GEMM 6x8x8: explicit tail-policy rejection: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hls", action="store_true", help="also export HLS C++")
    args = parser.parse_args()
    _check(2, expected_instructions=7, expected_vstores=1)
    _check(20, expected_instructions=73, expected_vstores=4)
    _check_negate()
    _check_tiled_gemm(8, 8, 8)
    _check_tiled_gemm(8, 16, 32)
    _check_gemm_tail_rejected()
    if args.hls:
        from .microarch import OP_VNEG, top_s

        hls = top_s.export("vitis").hls_code
        assert OP_VNEG == 9
        assert "tinytpu" in hls and "dma_store" in hls and "mxu" in hls
        print(f"HLS export: {len(hls)} bytes: PASS")


if __name__ == "__main__":
    main()
