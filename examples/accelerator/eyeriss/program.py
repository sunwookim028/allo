# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling CONV layers onto Eyeriss, and what the configuration search does.

Eyeriss is a *schedule ISA* (see ``todos/schedule-isa.md``): its hardware computes
one fixed thing, and its per-layer configuration says how the layer's ``N x M x C``
logical PE sets are folded onto the 12x14 array. That fold is neither a shape nor a
value, so it is an ``@I.schedule`` param -- chosen by the compiler, constrained by
``pass_fits``, and emitted as an instruction-word field (printed ``@n``).

The four layers below exercise the three outcomes:

- **one pass.** A layer small enough to configure directly compiles to a single
  ``conv_pass``, at the widest legal spatial fold -- that costs the fewest sequential
  set-times, and ``configure`` minimizes cost.
- **phase-2 folding.** A batch whose working set will not fit on-chip goes through
  ``conv_layer``, whose ``@I.expand`` stages one image at a time. Each pass it issues
  is configured on its own: an expansion is the compiler's own lowering, so the
  schedule params of what it issues are the compiler's to choose, not the ISA
  author's to hard-code.
- **refusal.** A layer needing more sets than any configuration admits is rejected at
  *selection* time, with a message that says so. It used to compile to one confident,
  physically impossible instruction.

**Why there is no NumPy check here.** ``conv_pass`` is parametric in ``Cp`` / ``Mp``
-- it has to be, since those are what the fold is chosen against -- and MLIR's
``tosa-to-linalg-named`` will not legalize a ``tosa.conv2d`` with dynamic dims, so
the functional simulator cannot execute it. Isolated: a *static* conv2d instruction
simulates fine (with a length-1 or a length-OC bias alike), and a parametric
``tosa.matmul`` simulates fine too -- MiniNPU's ``matmul_layer`` does. It is
specifically the named-conv lowering. What is checked instead is the compiled
instruction stream: which instruction was selected, how many passes, what
configuration each carries, and the expansion's address arithmetic.

Run:  ``python -m examples.accelerator.eyeriss.program``
"""

from .isa import (
    E_TILE,
    GLB_WORDS,
    H_TILE,
    MAX_SPATIAL,
    R,
    S,
    SET_SPAD,
    SPAD_WORDS,
    eyeriss,
    pass_fits,
    stage_words,
)
from allo.exp.dsa.errors import CompileError


def _source(N: int, C: int, M: int) -> str:
    """``psum + conv2d(ifmap, filter)`` over one ``H_TILE x H_TILE`` window, in the
    NHWC / OHWI layout the frontend's TOSA contract uses."""
    x, w = f"{N}x{H_TILE}x{H_TILE}x{C}xf32", f"{M}x{R}x{S}x{C}xf32"
    o = f"{N}x{E_TILE}x{E_TILE}x{M}xf32"
    return f"""
func.func @main(%i: tensor<{x}>, %w: tensor<{w}>, %p: tensor<{o}>) -> tensor<{o}> {{
  %z = "tosa.const"() {{values = dense<0.0> : tensor<1xf32>}} : () -> tensor<1xf32>
  %b = "tosa.const"() {{values = dense<0.0> : tensor<1xf32>}} : () -> tensor<1xf32>
  %r = "tosa.conv2d"(%i, %w, %b, %z, %z) <{{acc_type = f32,
       dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>,
       stride = array<i64: 1, 1>}}>
       : (tensor<{x}>, tensor<{w}>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
       -> tensor<{o}>
  %o = tosa.add %p, %r : (tensor<{o}>, tensor<{o}>) -> tensor<{o}>
  return %o : tensor<{o}>
}}"""


def _compile(N: int, C: int, M: int, note: str):
    """Compile one layer and print its stream; ``None`` if it was refused."""
    print(f"=== N={N} C={C} M={M} — {N * C * M} logical PE sets — {note}")
    try:
        prog = eyeriss.compile_program(_source(N, C, M))
    except CompileError as e:
        print(f"  refused: {e}\n")
        return None
    for rec in prog.emits:
        args = [str(a) for a in rec.addr] + [f"@{v}" for v in rec.schedule]
        print(f"  {rec.name}({', '.join(args)})")
    print()
    return prog


def main() -> None:
    per_pass = MAX_SPATIAL * (SPAD_WORDS // SET_SPAD)
    print(
        f"Eyeriss: one processing pass admits at most {MAX_SPATIAL} spatial x "
        f"{SPAD_WORDS // SET_SPAD} temporal = {per_pass} logical PE sets,\n"
        f"and its on-chip working set must fit {GLB_WORDS} words of global buffer.\n"
    )

    # 1. Small enough to configure directly: one pass, at the widest legal fold.
    one = _compile(1, 16, 8, "fits one pass")
    assert one is not None
    passes = [e for e in one.emits if e.name == "conv_pass"]
    assert len(passes) == 1, passes
    assert passes[0].schedule == [MAX_SPATIAL], passes[0]

    # 2/3. Batches: `conv_layer` stages one image at a time, and every pass its
    #      expansion issues carries a configuration the compiler chose for it.
    for n in (2, 8):
        note = "the whole layer would not fit on-chip; one image does"
        prog = _compile(n, 16, 8, note)
        assert prog is not None
        passes = [e for e in prog.emits if e.name == "conv_pass"]
        assert len(passes) == n, passes
        assert all(p.schedule == [MAX_SPATIAL] for p in passes), passes
        # The expansion hoists the filter load out of the image loop; per image it
        # stages ifmap + psum in, runs the pass, and writes the result back.
        assert len(prog.emits) == 1 + 4 * n, prog.emits
        # Each pass reads the same staging slots — the working set is one image.
        assert len({tuple(p.addr[:4]) for p in passes}) == 1, passes

    # 4. No configuration admits it, so it is not compiled at all.
    assert _compile(1, 64, 32, "no configuration admits it") is None

    # The two constraints the refusal rests on, stated directly.
    assert pass_fits(1 * 16 * 8, MAX_SPATIAL) and not pass_fits(
        1 * 64 * 32, MAX_SPATIAL
    )
    assert stage_words(16, 8) <= GLB_WORDS
    print("All 4 layers compiled to the expected configuration, or were refused.")


if __name__ == "__main__":
    main()
