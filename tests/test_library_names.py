# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A user symbol named like a library op is the user's (A3).

``decompose_library_function`` replaced every call whose callee *started with*
``gelu``/``layernorm``/``tril`` by the library body, and erased every
module-level symbol starting with them: a constant ``gelu_tab`` was deleted
(``'memref.get_global' op 'gelu_tab' does not reference a valid global
memref``), a user function ``gelu_addr`` was replaced by GELU. Only the
builder's own declarations (``@gelu_<hash>``, bodiless) are library ops. See
``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import numpy as np

import allo
import allo.dataflow as df
from allo.ir.types import int32, float32

ROM = (np.arange(64, dtype=np.int32) * 3) % 17
N = 8


def _region(prefix):
    # The constant's name is the point of the test, so it is spelled out per
    # case rather than built from a template.
    if prefix == "gelu":

        @df.region()
        def top(X: int32[N], R: int32[N]):
            @df.kernel(mapping=[1], args=[X, R])
            def k(x: int32[N], r: int32[N]):
                gelu_tab: int32[64] = ROM
                for i in range(N):
                    r[i] = gelu_tab[x[i]]

    elif prefix == "layernorm":

        @df.region()
        def top(X: int32[N], R: int32[N]):
            @df.kernel(mapping=[1], args=[X, R])
            def k(x: int32[N], r: int32[N]):
                layernorm_tab: int32[64] = ROM
                for i in range(N):
                    r[i] = layernorm_tab[x[i]]

    else:

        @df.region()
        def top(X: int32[N], R: int32[N]):
            @df.kernel(mapping=[1], args=[X, R])
            def k(x: int32[N], r: int32[N]):
                tril_tab: int32[64] = ROM
                for i in range(N):
                    r[i] = tril_tab[x[i]]

    return top


def test_constant_named_like_library_simulator():
    for prefix in ("gelu", "layernorm", "tril"):
        mod = df.build(_region(prefix), target="simulator")
        x = np.arange(N, dtype=np.int32) * 5
        r = np.zeros(N, dtype=np.int32)
        mod(x, r)
        np.testing.assert_array_equal(r, ROM[x], err_msg=prefix)


def gelu_addr(v: int32) -> int32:
    return v * 2 + 1


def tril_rows(A: int32[N]) -> int32[N]:
    B: int32[N] = 0
    for i in range(N):
        B[i] = A[i] - 3
    return B


def test_user_function_named_like_library_llvm():
    # ``tril_rows`` is called at the top level of the kernel body, where the
    # pass looks for library calls; ``gelu_addr`` inside a loop.
    def kernel(A: int32[N]) -> int32[N]:
        C: int32[N] = 0
        for i in range(N):
            C[i] = gelu_addr(A[i])
        B = tril_rows(C)
        return B

    s = allo.customize(kernel)
    a = np.arange(N, dtype=np.int32)
    np.testing.assert_array_equal(s.build()(a), a * 2 + 1 - 3)
    # Both functions survive to the HLS text under their own names.
    code = str(s.build(target="vhls"))
    assert "gelu_addr(" in code and "tril_rows(" in code


def test_library_gelu_beside_user_gelu_symbol():
    """The real library op is still replaced when a user symbol shares its prefix."""
    L, D = 4, 8
    TAB = np.linspace(-1, 1, D).astype(np.float32)

    def kernel(inp: float32[L, D]) -> float32[L, D]:
        gelu_bias: float32[D] = TAB
        out = allo.gelu(inp)
        for i, j in allo.grid(L, D):
            out[i, j] = out[i, j] + gelu_bias[j]
        return out

    s = allo.customize(kernel)
    inp = np.random.randn(L, D).astype(np.float32)
    want = 0.5 * inp * (1 + np.tanh(0.797885 * (inp + 0.044715 * inp**3))) + TAB
    np.testing.assert_allclose(s.build()(inp), want, atol=1e-3)


if __name__ == "__main__":
    test_constant_named_like_library_simulator()
    test_user_function_named_like_library_llvm()
    test_library_gelu_beside_user_gelu_symbol()
    print("PASS")
