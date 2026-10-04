# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The track E gate: every composition prototype, one line per check.

    source examples/minitpu/harness/env-zhang21.sh
    PYTHONPATH=$PWD $ALLO_PYTHON -m examples.minitpu.template.run_u3e [--quick]

Verdicts: ``U3E-OK`` (what the record claims holds), ``U3E-FINDING`` (a tool
behaviour the record lists; the gate passes), ``U3E-FAIL``. Exit status is
non-zero on any FAIL. ``--quick`` skips the DIM=16 builds (~20 s).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import allo.dataflow as df
from allo.compose import Architecture, Channel

from examples.minitpu.template import legality as lg
from examples.minitpu.template import mac_pe
from examples.minitpu.template import matrix_engine as me
from examples.minitpu.template import optional as opt
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32

FAILS = []


def verdict(ok, label, detail="", finding=False):
    tag = "U3E-OK" if ok else ("U3E-FINDING" if finding else "U3E-FAIL")
    if not ok and not finding:
        FAILS.append(label)
    print(f"{tag:12s} {label}{'  -- ' + detail if detail else ''}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    # (1) engine interface + (2) instantiation: one PE source, two MACs
    for eng, n in ((BF16_ACC24, 4096), (INT8_INT32, 1024)):
        out, want = mac_pe.run_rig(mac_pe.pe_rig(eng, n), eng, n, seed=3)
        verdict(np.array_equal(out, want), f"mac_pe at {eng.name}, own region",
                f"{np.sum(out == want)}/{n} bit-exact vs U1 references")
    (oa, wa), (ob, wb) = mac_pe.run_rig_two(mac_pe.pe_rig_two(BF16_ACC24, INT8_INT32, 512),
                                            BF16_ACC24, INT8_INT32, 512)
    verdict(np.array_equal(oa, wa) and np.array_equal(ob, wb),
            "H10: one PE source, two engines, ONE region (instantiate.instance)",
            f"{np.sum(oa == wa)}/512 + {np.sum(ob == wb)}/512")

    # (3) the engine's schedule directive travels with it
    arch = mac_pe.pe_rig(BF16_ACC24, 16)
    s = df.customize(arch.region())
    try:
        arch.directives(s)
        verdict(True, "C10: engine directive applied through the unit that binds it",
                "s.unroll('leading_zeros19:offset') found its band")
    except Exception as e:  # pylint: disable=broad-except
        verdict(False, "C10: engine directive", str(e)[:100])

    # (1) the matrix engine: two orders behind one interface, each vs the
    # contract reference evaluated with ITS order
    shapes = [(4, 64)] if args.quick else [(4, 64), (16, 32)]
    for mat in (me.SYSTOLIC, me.TREE):
        for dim, n in shapes:
            got, want = me.run_rig(mat, BF16_ACC24, dim, n, seed=5)
            verdict(np.array_equal(got, want),
                    f"matrix engine {mat.name} ({mat.order}) at bf16, DIM={dim}",
                    f"{np.sum(got == want)}/{got.size}; model latency {mat.latency_model(dim, BF16_ACC24)}")
    for mat in (me.SYSTOLIC, me.TREE):
        try:
            got, want = me.run_rig(mat, INT8_INT32, 16, 16, seed=5)
            verdict(np.array_equal(got, want), f"matrix engine {mat.name} at int8, DIM=16",
                    f"{np.sum(got == want)}/{got.size}")
        except Exception as e:  # pylint: disable=broad-except
            verdict(False, f"matrix engine {mat.name} at int8, DIM=16", str(e).splitlines()[0][:100])
    try:
        me.run_rig(me.SYSTOLIC, INT8_INT32, 4, 8)
        verdict(True, "matrix engine systolic at int8, DIM=4 (32-bit packed word)")
    except Exception as e:  # pylint: disable=broad-except
        verdict(False, "matrix engine systolic at int8, DIM=4 (32-bit packed word)",
                "lowering fails: " + " ".join(str(e).split())[:90], finding=True)
    # H11, the reference half: the two orders are different functions at bf16
    for exact in (False, True):
        A, W = me.stimulus(BF16_ACC24, 16, 512, seed=7, exact=exact)
        seq = me.matrix_rows(A, W, BF16_ACC24, "sequential")
        tree = me.matrix_rows(A, W, BF16_ACC24, "tree")
        d = int(np.sum(seq != tree))
        verdict((d > 0) != exact, f"H11: sequential vs tree at bf16 DIM=16, {'exact-sum' if exact else 'random'} data",
                f"{d}/{seq.size} results differ")
    A, W = me.stimulus(INT8_INT32, 16, 256, seed=1)
    sa, sw = me._signed_in(INT8_INT32, A), me._signed_in(INT8_INT32, W)
    d = int(np.sum(me.matrix_rows(sa, sw, INT8_INT32, "sequential") != me.matrix_rows(sa, sw, INT8_INT32, "tree")))
    verdict(d == 0, "H11: sequential vs tree at int8 (no rounding)", f"{d} differ")
    try:
        me.mxu_rig(me.TREE, BF16_ACC24, 6, 4)
        verdict(False, "tree engine legality refuses DIM=6")
    except AssertionError as e:
        verdict(True, "tree engine legality refuses DIM=6", str(e)[:60])

    # (4) optional modules
    base = opt.vpu_lane_base(64)
    for label, options, with_sfu, prog_ok in (("without SFU", (), False, False), ("with SFU", (opt.SFU_OPTION,), True, True)):
        arch, isa = opt.assemble(base, *options)
        out, want = opt.run_lane(arch, 64, with_sfu)
        verdict(np.array_equal(out, want), f"VPU lane {label}", f"{np.sum(out == want)}/64; isa={isa}")
        try:
            opt.assemble_program(["vadd", "vgelu"], isa)
            verdict(prog_ok, f"  program with vgelu {label}: assembled")
        except AssertionError:
            verdict(not prog_ok, f"  program with vgelu {label}: refused, naming the module")
    for label, make in (
        ("H15: SFU added without rebinding writeback (two readers)",
         lambda: opt.assemble(base, opt.SFU_OPTION_NO_REBIND)),
        ("H15: SFU removed, its channel left declared (dangling)",
         lambda: Architecture(name="lane_dangling", parameters=base.parameters, memories=base.memories,
                              channels=base.channels + (Channel("sfu_out", "UInt(64)", "QD"),),
                              units=base.units)),
    ):
        try:
            make()
            verdict(False, label, "ACCEPTED by compose")
        except AssertionError as e:
            verdict(True, label, "refused by compose: " + str(e)[:70])

    # (5) derived parameters
    rows = lg.check_against_phase0()
    verdict(True, "derived numbers == Phase 0 measurements", f"{len(rows)} relations")
    for label, result in lg.refusals():
        verdict(result.startswith("refused"), f"legality: {label}", result[:80])

    print(f"{'U3E-GATE OK' if not FAILS else 'U3E-GATE FAIL ' + str(FAILS)}  ({time.time() - t0:.0f} s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
