# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``@ Stateful(reset=False)``: declared unreset storage (README D-14).

The SystemC emitter writes such storage from a clock-edge ``SC_METHOD`` with
no reset action, and ``run.tcl`` sets ``-RESET_CLEARS_ALL_REGS no``, so
Catapult builds plain flops (``examples/minitpu/units/vpu_regfile.py``
variant ``comb_unreset``, measured against MiniTPU in
``dev/records/minitpu/u2_unreset_impl_2026-10-02.rst``). Every other HLS
backend refuses it, naming the storage; the simulator runs it as ordinary
storage. The emit and refusal tests run anywhere; the csim test compiles and
runs the emitted testbench and skips without Catapult (``MGC_HOME``) and
``SYSTEMC_HOME``.
"""

import os
import re
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import Stateful, Stream, UInt, Wire, comb, int32, uint1

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

A5 = UInt(5)
D16 = UInt(16)


def _regfile(n):
    """The legal form: ``test_systemc_comb.py``'s regfile with ``mem``
    declared unreset."""

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D16]
        w_we: Wire[uint1]
        w_qa: Wire[D16, comb]

        @df.kernel(mapping=[1], args=[RA, WA, WD, WE])
        def src(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def _regfile_rmw(n):
    """The write's data reads the storage (an accumulate): the clock-edge
    write may read only Wire inputs, constants and iteration temporaries."""

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D16]
        w_we: Wire[uint1]
        w_qa: Wire[D16, comb]

        @df.kernel(mapping=[1], args=[RA, WA, WD, WE])
        def src(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                if e:
                    mem[x] = mem[x] + d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def _regfile_stream(n):
    """The writing kernel takes a stream: its iterations are not clock
    cycles, so a per-edge write would not match them."""

    @df.region()
    def top(WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
        s_wa: Stream[A5, 2]
        s_wd: Stream[D16, 2]
        s_we: Stream[uint1, 2]
        s_q: Stream[D16, 2]

        @df.kernel(mapping=[1], args=[WA, WD, WE])
        def src(wa: A5[n], wd: D16[n], we: uint1[n]):
            for t in range(n):
                s_wa.put(wa[t])
                s_wd.put(wd[t])
                s_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            for _ in range(n):
                x5: A5 = s_wa.get()
                x: int32 = x5
                d: D16 = s_wd.get()
                e: uint1 = s_we.get()
                s_q.put(mem[x])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = s_q.get()

    return top


def _regfile_inner_loop(n):
    """The store is inside a loop within the iteration."""

    @df.region()
    def top(WD: D16[n], WE: uint1[n]):
        w_wd: Wire[D16]
        w_we: Wire[uint1]

        @df.kernel(mapping=[1], args=[WD, WE])
        def src(wd: D16[n], we: uint1[n]):
            for t in range(n):
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            for _ in range(n):
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                if e:
                    for k in range(4):
                        mem[k] = d

    return top


def _plain_regfile(n, reset):
    """Array ports (no Wire): for the simulator and the C++ HLS backends."""
    if reset:

        @df.region()
        def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
            @df.kernel(mapping=[1], args=[RA, WA, WD, WE, QA])
            def rf(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n], qa: D16[n]):
                mem: D16[32] @ Stateful = 0
                for t in range(n):
                    ia: int32 = ra[t]
                    qa[t] = mem[ia]
                    iw: int32 = wa[t]
                    dw: D16 = wd[t]
                    if we[t]:
                        mem[iw] = dw

        return top

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
        @df.kernel(mapping=[1], args=[RA, WA, WD, WE, QA])
        def rf(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n], qa: D16[n]):
            mem: D16[32] @ Stateful(reset=False)
            for t in range(n):
                ia: int32 = ra[t]
                qa[t] = mem[ia]
                iw: int32 = wa[t]
                dw: D16 = wd[t]
                if we[t]:
                    mem[iw] = dw

    return top


def _kernel_module(code, name):
    m = re.search(r"SC_MODULE\(" + name + r"\) \{(.*?)\n\};", code, flags=re.S)
    assert m, f"no SC_MODULE({name})"
    return m.group(1)


def test_unreset_marker():
    """Declared, never inferred: only ``reset=False`` marks the global."""
    assert Stateful(reset=False).reset is False and Stateful().reset is True
    with pytest.raises(TypeError):
        Stateful(reset=1)
    ir = str(df.customize(_regfile(4)).module)
    assert re.search(r"memref\.global .*__stateful_rf_0_mem.* \{allo\.unreset = \"mem\"", ir), ir
    ir = str(df.customize(_plain_regfile(4, reset=True)).module)
    assert "__stateful_" in ir and "allo.unreset" not in ir


def test_unreset_bad_argument_refused(capfd):
    @df.region()
    def top(WD: D16[4]):
        @df.kernel(mapping=[1], args=[WD])
        def k(wd: D16[4]):
            mem: D16[4] @ Stateful(reset=1)
            for t in range(4):
                mem[t] = wd[t]

    # the frontend reports the RuntimeError and exits (its error handler)
    with pytest.raises((RuntimeError, SystemExit)):
        df.customize(top)
    out, err = capfd.readouterr()
    assert "Stateful(...) takes only `reset=True` or `reset=False`" in out + err


def test_unreset_emit_shape():
    """F3 form f, generalised: the write is a clock-edge SC_METHOD with no
    reset action; the thread keeps no store to the storage."""
    code = df.build(_regfile(8), target="systemc").hls_code
    rf = _kernel_module(code, "rf_0")
    sym = re.search(r"sc_signal< .* > (__stateful_rf_0_mem\w*)\[32\];  // @ Stateful\(reset=False\)", rf)
    assert sym, rf
    mem = sym.group(1)
    assert re.search(r"SC_METHOD\(wr\);\n\s*sensitive << clk\.pos\(\);", rf)
    assert f"// allo unreset storage: {mem}" in rf
    assert f"{mem}[_sr]" not in rf  # no reset-action write
    thread = rf[rf.index("void run()") : rf.index("void comb()")]
    wr = rf[rf.index("void wr()") :]
    assert f"{mem}[" not in thread.replace(f"// {mem}:", "").replace(
        f"placeholder for const int16_t {mem}", ""
    )
    assert re.search(re.escape(mem) + r"\[.*\]\.write\(", wr)
    assert "if (" in wr  # the enable
    # the comb read (D-13) is unchanged: the method reads the same signals
    assert re.search(re.escape(mem) + r"\[.*\]\.read\(\);", rf[rf.index("void comb()") : rf.index("void wr()")])


def test_unreset_run_tcl():
    """``-RESET_CLEARS_ALL_REGS no`` only when unreset storage exists."""
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_regfile(8), target="systemc", mode="csyn", project=tmp + "/u")
        assert "directive set -RESET_CLEARS_ALL_REGS no\n" in open(tmp + "/u/run.tcl").read()
        df.build(_plain_regfile(8, reset=True), target="systemc", mode="csyn", project=tmp + "/r")
        assert "RESET_CLEARS_ALL_REGS" not in open(tmp + "/r/run.tcl").read()


@pytest.mark.parametrize(
    "make,why",
    [
        (_regfile_rmw, "not storage"),
        (_regfile_stream, "only when every port is a Wire"),
        (_regfile_inner_loop, "inside a loop within the iteration"),
    ],
)
def test_unreset_refusals(make, why):
    with pytest.raises(Exception) as ex:
        df.build(make(8), target="systemc")
    assert "unreset storage `mem` (rf_0)" in str(ex.value), str(ex.value)
    assert why in str(ex.value), str(ex.value)


def test_unreset_other_backends():
    """vhls and the Catapult C++ flow refuse, naming the storage; the
    simulator runs it as ordinary storage."""
    for tgt in ("vhls", "catapult"):
        with pytest.raises(NotImplementedError) as ex:
            df.build(_plain_regfile(8, reset=False), target=tgt)
        assert "unreset storage `mem`" in str(ex.value), str(ex.value)
    n = 64
    ra, wa, wd, we = _trace(n)
    qa = np.zeros(n, dtype=np.uint16)
    mod = df.build(_plain_regfile(n, reset=False), target="simulator")
    mod(ra, wa, wd, we, qa)
    want = _model(ra, wa, wd, we)
    defined = want >= 0
    assert np.array_equal(qa[defined], want[defined])


def _trace(n, seed=0):
    rng = np.random.default_rng(seed)
    ra = rng.integers(0, 32, n).astype(np.uint8)
    wa = rng.integers(0, 32, n).astype(np.uint8)
    wd = rng.integers(0, 1 << 16, n).astype(np.uint16)
    we = (rng.random(n) < 0.6).astype(np.uint8)
    return ra, wa, wd, we


def _model(ra, wa, wd, we):
    """Read, then write, per cycle; -1 marks a read of a never-written entry."""
    mem = [-1] * 32
    out = []
    for a, x, d, e in zip(ra, wa, wd, we):
        out.append(mem[int(a)])
        if e:
            mem[int(x)] = int(d)
    return np.array(out)


@needs_csim
def test_unreset_regfile_csim():
    """Compiles and runs the emitted testbench; as ``test_systemc_comb.py``,
    compared at the constant offset where every defined slot agrees (a Wire
    link is not cycle-locked in csim, limitation 22)."""
    n = 64
    ra, wa, wd, we = _trace(n)
    qa = np.zeros(n, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_regfile(n), target="systemc", mode="csim", project=tmp)
        mod(ra, wa, wd, we, qa)
    want = _model(ra, wa, wd, we)
    best = None
    for k in range(0, 6):
        if all(want[t] < 0 or int(qa[t + k]) == want[t] for t in range(n - k)):
            best = k
            break
    assert best is not None, f"no constant offset matches: got {qa[:12]}, want {want[:12]}"
    print(f"unreset regfile csim: values match at offset +{best}")


def _regfile_stateful(n, D, reset):
    """``_regfile`` with ``mem`` a ``@ Stateful`` -- comb storage written by the
    thread (``reset=True``) or unreset storage written by ``wr`` -- so that it
    persists across csim calls (README D-11)."""
    st = Stateful if reset else Stateful(reset=False)

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D[n], WE: uint1[n], QA: D[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D]
        w_we: Wire[uint1]
        w_qa: Wire[D, comb]

        @df.kernel(mapping=[1], args=[RA, WA, WD, WE])
        def src(ra: A5[n], wa: A5[n], wd: D[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D[32] @ st = 0
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


@needs_csim
@pytest.mark.parametrize("reset", [True, False], ids=["comb_storage", "unreset"])
def test_signal_storage_persists_across_csim_calls(reset):
    """D-11 over D-13/D-14 signal storage (``sc_signal<T> mem[32]``): saved
    through ``.read()``, reloaded through ``.write()`` by the process that owns
    the signal -- the thread after its reset action for comb storage,
    ``start_of_simulation`` for unreset storage (the ``wr`` method has no reset
    action, and a second writer would break sc_signal's one-writer rule). An
    80-bit element takes the decimal-text path (_rdwide/_wrwide, S7)."""
    n, D = 64, UInt(80)
    val = lambda i: (1 << 79) + (i + 1) * 977

    def obj(vals):  # > 64-bit elements travel as Python ints
        out = np.empty(len(vals), dtype=object)
        out[:] = [int(v) for v in vals]
        return out

    t = list(range(n))
    wa = np.array([i % 32 for i in t], np.uint8)
    wd = obj([val(i % 32) for i in t])
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_regfile_stateful(n, D, reset), target="systemc", mode="csim",
                       project=tmp)
        assert ("start_of_simulation" in mod.hls_code) == (not reset)
        mod(np.zeros(n, np.uint8), wa, wd, np.ones(n, np.uint8), obj([0] * n))
        q = obj([0] * n)  # second call: no writes, read every address
        mod(wa, wa, wd, np.zeros(n, np.uint8), q)
        got = [int(v) for v in q]
        assert any(
            all(got[i + k] == val(i % 32) for i in range(n - k)) for k in range(6)
        ), f"state not resumed: {got[:6]}"
        mod.reset()
        q = obj([0] * n)
        mod(wa, wa, wd, np.zeros(n, np.uint8), q)
        assert all(int(v) == 0 for v in q), "reset() did not clear the state"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
