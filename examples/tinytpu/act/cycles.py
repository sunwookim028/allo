# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cheap static cycle gate, and the KPN round count it was measured against.

Every unit is one flat loop over the work count the header promises it, and the
units run concurrently, so the steady state is the busiest unit's count and
everything else is fixed cost. `CRITICAL_WORK_FIT` calibrates those two
constants; what it is worth out of sample is in
docs/source/extensions/act_specs.rst."""

import collections
import functools
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu import kpn_model  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    NHDR, QD, assemble, expand,
)

UNITS = ("dma_ld", "spm", "vru", "accu", "dma_st")
HEADER_SLOT = {"dma_ld": 1, "spm": 2, "vru": 3, "accu": 5, "dma_st": 6}

@functools.lru_cache(maxsize=1)
def published():
    """The five published cycle counts, DERIVED from `reproduce.sh`'s EXPECTED.

    Not a literal. `PUBLISHED_CYCLES` was one, and it went stale: it still said
    172 / 262 / 418 / 484 / 686 after the design shipped
    175 / 265 / 421 / 482 / 674, so the fit below was calibrated against a row
    the design had not produced for days. `EXPECTED` is the one copy of the row
    a gate checks on every run (`reproduce.sh` exits nonzero if cosim disagrees
    with it), which makes it the only honest source. Same reasoning, and the
    same regex, as `chia_agent/control.reproduced`.

    **Read on first use, never at import.** This module is in the CHIA
    evaluator's frozen import closure (`evaluate.import_closure`), and the tree
    it composes carries python modules and a few data files -- not
    `reproduce.sh`. Reading it at import time made every candidate die at stage
    `import`, measured 2026-09-25 by `test_harness.py`. Nothing the evaluation
    runs touches this row (it uses `estimate` and `work`), so deferring the
    read costs nothing and keeps the frozen set exactly as it was.
    """
    text = (pathlib.Path(__file__).resolve().parents[1]
            / "reproduce.sh").read_text()
    row = re.search(r'^EXPECTED="([^"]+)"', text, re.M).group(1)
    out = {f"gemm_{s}": int(c) for s, c in (kv.split("=") for kv in row.split())}
    assert len(out) == 5, f"reproduce.sh EXPECTED is not five shapes: {out}"
    return out


def __getattr__(name):
    """`cycles.PUBLISHED_CYCLES` still works, and still costs nothing to import."""
    if name == "PUBLISHED_CYCLES":
        return published()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
#: Least squares over `PUBLISHED_CYCLES`, recovered by `refit()` -- never typed
#: in from a previous row, and `__main__` fails if the two disagree. Refitted
#: 2026-09-25, against the row re-measured on this host the same day
#: (`reproduce.sh` -> REPRODUCED 175 / 265 / 421 / 482 / 674). It was
#: (173.2, 1.621) against 172 / 262 / 418 / 484 / 686.
#:
#: THE REFIT MADE THE MODEL WORSE, and that is a finding about the model rather
#: than something to tune away. The critical-work column `x` did not move at
#: all -- 9 / 48 / 144 / 192 / 320, the same header counts off the same design
#: -- so the whole change is in `y`. Between the two rows the design took two
#: steps (`reproduce.sh` documents both): derived memory sizing, -1 at every
#: shape, and `TPU_QD=16`, +4 / +4 / +4 / -1 / -11. The second one is the
#: problem. It is a term that GROWS with work (deeper-FIFO pipeline skew, paid
#: by the three smallest shapes) fighting a term that SHRINKS with it (a deeper
#: queue lets the sequencer run ahead of the units, which only the two largest
#: shapes run long enough to collect), and one slope over one variable cannot
#: hold both. Worst residual 15.8 -> 18.2 cycles, rms 10.3 -> 12.3, on 5
#: points; the 10% gate this feeds is 17.5 cycles at the smallest shape, so the
#: model is now outside its own tolerance there and `--cosim` is the answer.
CRITICAL_WORK_FIT = (179.0, 1.573)


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
    called by this module's `__main__`. `PUBLISHED_CYCLES` comes from
    `reproduce.sh`'s EXPECTED, which is Vitis HLS 2023.2 + xsim cosim at
    `TPU_MAXDIM=16`; that row was last re-measured on this host on 2026-09-25
    and `reproduce.sh` re-measures it on every run."""
    import numpy as np
    from examples.tinytpu.act import baseline
    from examples.tinytpu.act import spec as spec_mod
    x, y = [], []
    for name, measured in published().items():
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
    from examples.tinytpu.act import baseline
    from examples.tinytpu.act import spec as spec_mod
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
        pub = published().get(sp["name"])
        print(f"{sp['name']:28s} " + " ".join(f"{w[u]:8d}" for u in UNITS)
              + f"  {critical_unit(prog):>8s}  {estimate(prog):8.0f}  "
              f"{kpn_rounds(prog):6d}  {pub if pub else '':>9}")
