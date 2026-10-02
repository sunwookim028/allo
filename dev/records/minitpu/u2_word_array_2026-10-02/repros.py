# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repros of the ``vpu_word_array`` findings (U2), one function each.

    source examples/minitpu/harness/env-zhang21.sh
    $ALLO_PYTHON dev/records/minitpu/u2_word_array_2026-10-02/repros.py [name ...]

Output in ``logs/repros.txt``. Each prints ``REPRO <name>: <verdict>``.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, int32, uint1, uint64  # noqa: E402

PRJ = os.environ.get("REPRO_PRJ", "/tmp/u2wa_repros")


def s8_csim_u64_parse():
    """S8 (SystemC emitter, bug, silent): the csim testbench reads every port
    through ``long long _v; _f >> _v`` -- a ``uint64`` value >= 2**63 fails
    extraction (failbit), ``_v`` becomes LLONG_MAX and every later value of
    that file is lost. Repro: copy 4 words; the one >= 2**63 and all after it
    come back wrong. The simulator is exact."""
    n = 4

    @df.region()
    def top(A: uint64[n], B: uint64[n]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: uint64[n], b: uint64[n]):
            for t in range(n):
                b[t] = a[t]

    a = np.array([1, 2**63 - 1, 2**63 + 5, 7], dtype=np.uint64)
    res = {}
    for be in ("simulator", "systemc"):
        b = np.zeros(n, dtype=np.uint64)
        mod = df.build(top, target="simulator") if be == "simulator" else df.build(
            top, target="systemc", mode="csim", project=os.path.join(PRJ, "s8"))
        mod(a, b)
        res[be] = [hex(int(x)) for x in b]
    ok = res["systemc"] == res["simulator"]
    return f"{'no bug' if ok else 'BUG'}: in {[hex(int(x)) for x in a]} simulator {res['simulator']} csim {res['systemc']}"


def negative_step_range():
    """Minor (frontend, loud): ``for k in range(2, 0, -1)`` is refused by the
    affine lowering ("expected step to be representable as a positive signed
    integer") instead of being normalised; a forward loop with ``p[n - k]``
    indices is the workaround."""
    n = 4

    @df.region()
    def top(A: int32[n], B: int32[n]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[n], b: int32[n]):
            p: int32[3]
            for t in range(n):
                for j in range(2, 0, -1):
                    p[j] = p[j - 1]
                p[0] = a[t]
                b[t] = p[2]

    try:
        df.build(top, target="simulator")
        return "accepted (fixed?)"
    except Exception as e:  # noqa: BLE001
        return f"refused: {type(e).__name__}: {str(e)[:120]}"


def closure_bool_if():
    """Minor (frontend, loud): a Python ``bool`` captured from the enclosing
    scope and used as ``if flag:`` is lowered as an ``i32`` constant, and the
    verifier refuses the ``scf.if`` ("operand #0 must be 1-bit signless
    integer, but got 'i32'"). Expected: fold it, or refuse it by name."""
    n = 4
    flag = True

    @df.region()
    def top(A: int32[n], B: int32[n]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[n], b: int32[n]):
            for t in range(n):
                b[t] = a[t]
                if flag:
                    b[t] = a[t] + 1

    try:
        mod = df.build(top, target="simulator")
        a = np.arange(n, dtype=np.int32)
        b = np.zeros(n, dtype=np.int32)
        mod(a, b)
        return f"accepted: {b.tolist()}"
    except Exception as e:  # noqa: BLE001
        return f"refused: {type(e).__name__}: {str(e)[:160]}"


def rolled_shift_loop_emission():
    """C-W1 (SystemC emitter -> Catapult, workaround): a constant-trip inner
    loop (a shift-register pipe ``for k in range(1, 3): p[3-k] = p[2-k]``)
    is emitted rolled, with no unroll pragma. Catapult then MERGES it into the
    pipelined steady-state loop (SCHD-7 "2 c-steps"), so the kernel samples
    its Wire inputs every second cycle: 19,502/67,717 against MiniTPU. With
    ``s.unroll`` on both loops the emission carries ``#pragma hls_unroll`` and
    the RTL is cycle-exact. Shown here on the emitted C++ only (no Catapult)."""
    n = 4
    W = UInt(16)

    @df.region()
    def top(A: W[n], B: W[n]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: W[n], b: W[n]):
            p: W[3]
            for t in range(n):
                for j in range(1, 3):
                    p[3 - j] = p[2 - j]
                p[0] = a[t]
                b[t] = p[2]

    out = []
    for unroll in (False, True):
        s = df.customize(top)
        if unroll:
            s.unroll("k_0:j")
        code = s.build(target="systemc", mode="csyn", project=os.path.join(PRJ, f"roll_{unroll}")).hls_code
        out.append(f"unroll={unroll}: hls_unroll pragmas in emission = {code.count('hls_unroll')}")
    return "; ".join(out)


ALL = [s8_csim_u64_parse, negative_step_range, closure_bool_if, rolled_shift_loop_emission]

if __name__ == "__main__":
    names = sys.argv[1:]
    for fn in ALL:
        if names and fn.__name__ not in names:
            continue
        try:
            print(f"REPRO {fn.__name__}: {fn()}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"REPRO {fn.__name__}: CRASH\n{traceback.format_exc()[-800:]}", flush=True)
