# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling TOSA GEMMs into MINISA streams via ``fx.compile_program``.

The source is one un-tiled ``tosa.matmul`` over whole row-major matrices —
act-backend's input granularity, nothing about it tiled or placed. The search
matches it to the ``gemm`` layer instruction, whose ``@expand`` body lowers it to
the MINISA stream: the three ``Set*VNLayout`` setters once, then the Load /
ExecuteMapping / Store tile loop. Staging is ``scratch()``, so every address —
the dram blocks, the on-chip tiles, the ``cfg`` register rows — is the
allocator's answer.

What each compiled program is checked for, beyond matching NumPy:

- **The MINISA shape.** The emit stream opens with exactly one setter group, and
  ``prog.epochs()`` reads it as such: the three setters *join the first epoch*
  (they configure; the first Load's segment is where their effect first reaches a
  run), and every epoch of the program runs under the installed orders — the fold
  that one setter group configures the whole layer with.
- **The obligations.** ``prog.check()`` is empty: definedness, bounds,
  dependences all discharged on the serialized stream alone.

A one-tile layer (4x4x4) deliberately does *not* take this path: ``gemm`` ties
with bare ``mm`` at cost 1 and the earlier-declared tile op wins, so the program
is the four instructions a single ExecuteMapping needs and no configuration —
compiled, checked, and correct, but degenerate as MINISA. The multi-tile cases
are the real thing.

Run:  ``python -m examples.accelerator.featherx_dsa.program``
"""

import numpy as np

from .isa import TILE, fx


def source(M: int, K: int, N: int) -> str:
    """One ``tosa.matmul`` over whole matrices — no tiling, no memories named."""

    def t(*dims):
        return f"tensor<{'x'.join(map(str, dims))}xf32>"

    sig = f"({t(1, M, K)}, {t(1, K, N)}, tensor<1xf32>, tensor<1xf32>) -> {t(1, M, N)}"
    return f"""
func.func @main(%a: {t(1, M, K)}, %b: {t(1, K, N)}) -> {t(1, M, N)} {{
  %zp = "tosa.const"() {{values = dense<0.000000e+00> : tensor<1xf32>}} : () -> tensor<1xf32>
  %c = tosa.matmul %a, %b, %zp, %zp : {sig}
  return %c : {t(1, M, N)}
}}
"""


def compile_gemm(M: int, K: int, N: int, seed: int = 0):
    """Compile, verify the MINISA shape and the obligations, diff against NumPy."""
    prog = fx.compile_program(source(M, K, N))
    names = [e.name for e in prog.emits]

    # One setter group, at the head of the stream, and nowhere else.
    assert names[:3] == ["set_ivn", "set_wvn", "set_ovn"]
    assert sum(n.startswith("set_") for n in names) == 3
    n_em = names.count("mm") + names.count("mac")
    assert n_em == (M // TILE) * (K // TILE) * (N // TILE)

    # The epoch reading: the setters join the first run's segment, and the fold
    # carries their installed orders to every epoch of the layer.
    eps = prog.epochs()
    assert [n for n, _ in eps[0].members] == ["set_ivn", "set_wvn", "set_ovn", "load_i"]
    orders = {"ivn_order": 0, "wvn_order": 0, "ovn_order": 0}
    assert all(orders.items() <= e.config.schedule.items() for e in eps)
    assert prog.check() == []

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    b = rng.standard_normal((1, K, N)).astype(np.float32)
    got = np.asarray(prog(a, b), np.float32).reshape(1, M, N)
    np.testing.assert_allclose(got, a @ b, rtol=1e-3, atol=1e-3)
    print(
        f"    [ok] {M}x{K}x{N}: {len(names)} instructions "
        f"(1 setter group + {n_em} ExecuteMappings), {len(eps)} epochs, "
        f"check() clean, matches NumPy"
    )
    return prog


def one_tile_note() -> None:
    """The degenerate case: a single-tile layer skips configuration entirely."""
    prog = fx.compile_program(source(TILE, TILE, TILE))
    names = [e.name for e in prog.emits]
    assert not any(n.startswith("set_") for n in names)
    a = np.arange(TILE * TILE, dtype=np.float32).reshape(1, TILE, TILE)
    b = np.eye(TILE, dtype=np.float32).reshape(1, TILE, TILE)
    got = np.asarray(prog(a, b), np.float32).reshape(1, TILE, TILE)
    np.testing.assert_allclose(got, a @ b)
    print(f"    [ok] one-tile layer: {names} — no setters, bare ExecuteMapping")


def main() -> None:
    one_tile_note()
    for shape in ((8, 8, 8), (8, 16, 32), (16, 16, 16)):
        compile_gemm(*shape)
    print("All TOSA GEMMs compiled to MINISA streams, checked, and matched NumPy.")


if __name__ == "__main__":
    main()
