# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 24: five checks pass a TinyTPU-isa program whose cosim never finishes.

The cheap half runs in seconds and needs no Vitis: it asserts that the minimal
hanging program is accepted by `check_program`, reported deadlock-free by
`kpn_model`, and bit-exact against `isa_ref` on the dataflow simulator -- which
is the reproducible part of the claim, that a stack of cheap checks all pass.

`ALLO_LIMITS_COSIM=1` additionally runs Vitis: one `csynth_design`, then one
`cosim_design` per program under `ACT_COSIM_TIMEOUT` (default 600 s). The
minimal program's RTL does not complete; every neighbour with one knob halved
does. `TPU_PRJ` controls where the project lands and it is removed afterwards.
"""
import os
import shutil
import sys

import numpy as np

import _worktree
from _worktree import verdict

ITEM = 24

EX = os.path.join(_worktree.ROOT, "examples", "accelerator", "tinytpu_vitis")
sys.path.insert(0, _worktree.ROOT)

from examples.accelerator.tinytpu_vitis import isa_ref, kpn_model  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    check_program,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402
from examples.accelerator.tinytpu_vitis.act.rtl_hang import (  # noqa: E402
    family, host_spec,
)

#: The names in `act.rtl_hang.family()` whose RTL did not complete, measured
#: 2026-09-22; the rest completed with 0 of 256 bytes wrong.
HANGS = ("tiled n2 b2 k2 r4", "pointwise n4 r16 relu", "pointwise n4 r4 relu")


def cheap(prog, mod):
    from examples.accelerator.tinytpu_vitis.stress_isa import execute
    check_program(prog)
    good, _ = kpn_model.run(prog)
    A, B, C0 = spec_mod.buffers(host_spec(), "full", 900)
    want = isa_ref.run(prog, A, B, C0)
    got = execute(mod, prog, A, B, C0)
    return good, int((np.asarray(got) != np.asarray(want)).sum())


def main():
    import allo.dataflow as df
    from examples.accelerator.tinytpu_vitis.microarch_isa import tinytpu_isa

    mod = df.build(tinytpu_isa, target="simulator")
    res = {}
    for name, prog in family():
        good, wrong = cheap(prog, mod)
        res[name] = (good, wrong)
    all_cheap_pass = all(good and not wrong for good, wrong in res.values())
    print({k: ("kpn ok" if g else "kpn DEADLOCK", f"{w} wrong")
           for k, (g, w) in res.items()})

    if os.environ.get("ALLO_LIMITS_COSIM") != "1":
        verdict(ITEM, all_cheap_pass,
                f"{len(res)} programs accepted by check_program, kpn_model and "
                f"the simulator; {len(HANGS)} of them did not complete in "
                f"cosim when last measured. Set ALLO_LIMITS_COSIM=1 to re-run "
                f"the RTL half.")
        return

    from examples.accelerator.tinytpu_vitis.act import measure
    prj = measure.PRJ
    sp = host_spec()
    try:
        measure.synthesize(prj)
        hung = []
        for name, prog in family():
            n, line = measure.measure(sp, prog, prj, tag=name.replace(" ", "_"))
            print(f"  {name:24s} cosim={n}   {line}")
            if n is None:
                hung.append(name)
    finally:
        shutil.rmtree(prj, ignore_errors=True)
    verdict(ITEM, all_cheap_pass and bool(hung),
            f"cheap checks pass for all {len(res)}; RTL did not complete for "
            f"{hung}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
