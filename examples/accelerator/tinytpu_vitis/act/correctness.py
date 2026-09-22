# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact against two anchors, and it says which one moved.

`isa_ref.run` is the ISA as numpy and `spec.gold` is the einsum in int64: the
program is right when they agree over the spec's output region and nothing
outside the write window moved. The design is right when the built module
reproduces `isa_ref.run` on all MAXDIM*MAXDIM bytes. Prose:
docs/source/extensions/act_specs.rst."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis import isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import MAXDIM  # noqa: E402
from examples.accelerator.tinytpu_vitis.stress_isa import execute  # noqa: E402
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402

DISTS = ("small", "mid", "full", "corner")
REPEATS = 2


def build_module():
    """The one simulator build every spec is run on."""
    import allo.dataflow as df
    from examples.accelerator.tinytpu_vitis.microarch_isa import tinytpu_isa
    return df.build(tinytpu_isa, target="simulator")


def program_vs_spec(sp, prog, dist, seed):
    A, B, C0 = spec_mod.buffers(sp, dist, seed)
    want = spec_mod.gold(sp, A, B, C0)
    got = isa_ref.run(prog, A, B, C0)
    diff = spec_mod.compare(sp, got, want)
    return (A, B, C0, got,
            None if diff is None else
            f"the program does not compute the spec on {dist} operands "
            f"(seed {seed}): {diff}. The ISA's own semantics were followed, "
            f"so this is the mapping, not the hardware.")


def design_vs_isa(mod, prog, A, B, C0, want, dist, seed):
    got = execute(mod, prog, A, B, C0)
    bad = int((got != want).sum())
    if not bad:
        return None
    where = np.transpose(np.nonzero(
        got.reshape(MAXDIM, MAXDIM) != want.reshape(MAXDIM, MAXDIM)))
    return (f"the built design disagrees with the ISA on {dist} operands "
            f"(seed {seed}): {bad} of {MAXDIM * MAXDIM} bytes of C differ, "
            f"first at row {where[0][0]} column {where[0][1]}. This is a "
            f"hardware or simulator fault, not a program fault.")


def check(sp, prog, mod=None, dists=DISTS, repeats=REPEATS):
    """Every failure line for one (spec, program); empty means bit-exact."""
    fails = []
    for i, dist in enumerate(dists):
        A, B, C0, want, bad = program_vs_spec(sp, prog, dist, 700 + i)
        if bad:
            fails.append(bad)
            continue
        for r in range(repeats if mod is not None else 0):
            bad = design_vs_isa(mod, prog, A, B, C0, want, dist, 700 + i)
            if bad:
                fails.append(bad + (f" (invocation {r + 1} of {repeats} on one "
                                    f"build; a later one failing alone means "
                                    f"state the previous call left behind)"
                                    if r else ""))
                break
    return fails
