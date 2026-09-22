# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cheap static cycle gate, and the KPN round count it was measured against.

Every unit is one flat loop over the work count the header promises it, and the
units run concurrently, so the steady state is the busiest unit's count and
everything else is fixed cost. `CRITICAL_WORK_FIT` calibrates those two
constants; what it is worth out of sample is in
docs/source/extensions/act_specs.rst."""

import collections
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis import kpn_model  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    NHDR, QD, assemble, expand,
)

UNITS = ("dma_ld", "spm", "vru", "accu", "dma_st")
HEADER_SLOT = {"dma_ld": 1, "spm": 2, "vru": 3, "accu": 5, "dma_st": 6}

PUBLISHED_CYCLES = {"gemm_4x4x4": 172, "gemm_8x8x8": 262, "gemm_12x12x12": 418,
                    "gemm_16x16x8": 484, "gemm_16x16x16": 686}
CRITICAL_WORK_FIT = (173.2, 1.621)


def work(prog):
    """Per-unit dynamic work counts, straight off the header `assemble` writes."""
    hdr = assemble(prog)[:NHDR]
    out = {u: hdr[HEADER_SLOT[u]] for u in UNITS}
    out["static"] = len(prog)
    out["dynamic"] = len(expand(prog))
    out["n_mm"] = hdr[4] & 0xFFFF
    out["mm_rows"] = hdr[4] >> 16
    return out


def critical_work(prog):
    w = work(prog)
    return max(w[u] for u in UNITS)


def critical_unit(prog):
    w = work(prog)
    return max(UNITS, key=lambda u: w[u])


def estimate(prog):
    """The gate's cycle estimate: microseconds to compute, no tool involved."""
    fixed, marginal = CRITICAL_WORK_FIT
    return fixed + marginal * critical_work(prog)


def kpn_rounds(prog, QD=QD):
    """`kpn_model`'s process network stepped as a barrier-synchronous machine:
    every process gets one channel action per round, so the round count is the
    cycle number a KPN with II=1 links and no memory latency would give."""
    procs = kpn_model.build(prog)
    q = collections.defaultdict(collections.deque)
    pending = {n: None for n in procs}
    recv = {n: None for n in procs}
    done = set()
    rounds = 0
    while len(done) < len(procs):
        rounds += 1
        moved = False
        for name, gen in procs.items():
            if name in done:
                continue
            if pending[name] is None:
                try:
                    pending[name] = gen.send(recv[name])
                    recv[name] = None
                except StopIteration:
                    done.add(name)
                    moved = True
                    continue
            act = pending[name]
            if act[0] == "get":
                if not q[act[1]]:
                    continue
                recv[name] = q[act[1]].popleft()
            elif len(q[act[1]]) >= kpn_model.depth(act[1], QD):
                continue
            else:
                q[act[1]].append(act[2])
            pending[name] = None
            moved = True
        if not moved:
            return None
    return rounds


def refit():
    """`CRITICAL_WORK_FIT`, recomputed from `PUBLISHED_CYCLES`.

    The two constants are typed in so that `estimate` needs neither numpy nor
    the corpus, and this recovers them so that a typo cannot survive: it is
    called by this module's `__main__`. `PUBLISHED_CYCLES` are Vitis
    measurements this fork attributes to `dev/records/tinytpu/logs/cosim_isa_landed_sweep.log` at
    `e24e433b`, not measurements made here."""
    import numpy as np
    from examples.accelerator.tinytpu_vitis.act import baseline
    from examples.accelerator.tinytpu_vitis.act import spec as spec_mod
    x, y = [], []
    for name, measured in PUBLISHED_CYCLES.items():
        x.append(critical_work(baseline.program(spec_mod.by_name(name))))
        y.append(measured)
    fixed, marginal = np.linalg.lstsq(
        np.stack([np.ones(len(x)), np.array(x, float)], 1),
        np.array(y, float), rcond=None)[0]
    return round(float(fixed), 1), round(float(marginal), 3)


def report(prog):
    w = work(prog)
    return (f"{w['static']:3d} static, {w['dynamic']:4d} dynamic instructions; "
            f"critical unit {critical_unit(prog)} at {critical_work(prog)} work "
            f"items; estimate {estimate(prog):.0f} cycles")


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.act import baseline
    from examples.accelerator.tinytpu_vitis.act import spec as spec_mod
    assert refit() == CRITICAL_WORK_FIT, (
        f"CRITICAL_WORK_FIT is {CRITICAL_WORK_FIT}, least squares over "
        f"PUBLISHED_CYCLES gives {refit()}")
    print(f"cycles = {CRITICAL_WORK_FIT[0]} + {CRITICAL_WORK_FIT[1]} * "
          f"critical work, refitted from the published points")
    print(f"{'spec':28s} " + " ".join(f"{u:>8s}" for u in UNITS)
          + "  critical  estimate  rounds  published")
    for sp in spec_mod.corpus():
        if spec_mod.fits_build(sp) is not None:
            print(f"{sp['name']:28s} {spec_mod.fits_build(sp)}")
            continue
        prog = baseline.program(sp)
        w = work(prog)
        pub = PUBLISHED_CYCLES.get(sp["name"])
        print(f"{sp['name']:28s} " + " ".join(f"{w[u]:8d}" for u in UNITS)
              + f"  {critical_unit(prog):>8s}  {estimate(prog):8.0f}  "
              f"{kpn_rounds(prog):6d}  {pub if pub else '':>9}")
