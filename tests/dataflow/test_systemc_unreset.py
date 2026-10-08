# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``@ Stateful(reset=False)``: declared unreset storage (README D-14).

The SystemC emitter writes such storage from a clock-edge ``SC_METHOD`` with
no reset action, and ``run.tcl`` sets ``-RESET_CLEARS_ALL_REGS no`` on that
process alone (``/<top>/<kernel>/wr``), so Catapult builds plain flops for it
and keeps every other reset (``examples/minitpu/units/vpu_regfile.py`` variant
``comb_unreset``, measured against MiniTPU in
``dev/records/minitpu/u2_unreset_impl_2026-10-02.rst``; the scoped directive
and a mixed design through Catapult and Verilator in
``u2_d14_followups_2026-10-04.rst``). Every other HLS
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
    cycles, so the per-edge write method does not apply; the storage is a
    plain member the thread writes, with no reset action (README D-14, the
    lowering extended; U4 track C, C2)."""

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


def _run_tcl(make, tmp, **configs):
    df.build(make(8), target="systemc", mode="csyn", project=tmp, configs=configs or None)
    return open(tmp + "/run.tcl").read()


def test_unreset_run_tcl():
    """``-RESET_CLEARS_ALL_REGS no`` only when unreset storage exists, and then
    on the kernel's write process alone, after ``go compile`` (the process
    exists from then; the directive admits ``Solution,Design,Process``). The
    design-wide form also dropped the reset of a register in the same
    kernel's thread (u2_d14_followups_2026-10-04.rst, A1)."""
    with tempfile.TemporaryDirectory() as tmp:
        tcl = _run_tcl(_regfile, tmp + "/u")
        assert "go compile\ndirective set /top/rf_0/wr -RESET_CLEARS_ALL_REGS no\n" in tcl, tcl
        assert "directive set -RESET_CLEARS_ALL_REGS" not in tcl
        tcl = _run_tcl(_regfile, tmp + "/k", synth_top="rf_0")
        assert "go compile\ndirective set /rf_0/wr -RESET_CLEARS_ALL_REGS no\n" in tcl, tcl
        # a synth_top that does not contain the unreset storage gets no line
        tcl = _run_tcl(_regfile, tmp + "/s", synth_top="src_0")
        assert "RESET_CLEARS_ALL_REGS" not in tcl, tcl
        tcl = _run_tcl(lambda n: _plain_regfile(n, reset=True), tmp + "/r")
        assert "RESET_CLEARS_ALL_REGS" not in tcl


def _mixed(n):
    """Mixed reset in one kernel: unreset storage ``mem`` (comb read), reset
    storage ``tag`` (thread-written) and a reset counter ``cnt``. Through
    Catapult with the scoped directive, ``mem`` keeps its contents across a
    mid-run reset pulse and ``tag``/``cnt`` reset (u2_d14_followups, A2)."""

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n], QT: D16[n], QC: D16[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D16]
        w_we: Wire[uint1]
        w_qa: Wire[D16, comb]
        w_qt: Wire[D16]
        w_qc: Wire[D16]

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
            tag: D16[4] @ Stateful = 0
            cnt: D16[1] @ Stateful = 0
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qt.put(tag[a & 3])
                c: D16 = cnt[0]
                w_qc.put(c)
                cnt[0] = c + 1
                if e:
                    mem[x] = d
                    tag[x & 3] = d

        @df.kernel(mapping=[1], args=[QA, QT, QC])
        def sink(qa: D16[n], qt: D16[n], qc: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qt[t] = w_qt.get()
                qc[t] = w_qc.get()

    return top


def test_mixed_reset_emit():
    """One kernel, reset and unreset storage side by side: only ``mem`` is
    signal storage written by ``wr``; ``tag`` and ``cnt`` stay in the reset
    action; the directive names ``wr`` once. The state save in ``sc_main``
    is outside ``__SYNTHESIS__`` (Catapult parses ``sc_main`` too: CRD-135
    "class has no member __allo_state_save" aborted ``go analyze``)."""
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_mixed(8), target="systemc", mode="csyn", project=tmp)
        tcl = open(tmp + "/run.tcl").read()
    assert tcl.count("RESET_CLEARS_ALL_REGS") == 1
    assert "directive set /top/rf_0/wr -RESET_CLEARS_ALL_REGS no\n" in tcl
    rf = _kernel_module(mod.hls_code, "rf_0")
    assert re.search(r"// allo unreset storage: __stateful_rf_0_mem\w*\n", rf)
    assert "__stateful_rf_0_tag" not in rf[rf.index("void wr()") :]
    for g in ("__stateful_rf_0_tag", "__stateful_rf_0_cnt"):
        assert re.search(g + r"\w*\[_sr\w*\] = ", rf), g  # reset-action write
    main = mod.hls_code[mod.hls_code.index("int sc_main(") :]
    save = main.index("__allo_state_save(_f)")
    assert main.rfind("#ifndef __SYNTHESIS__", 0, save) > main.rfind("#endif", 0, save), main


def _model_mixed(ra, wa, wd, we):
    """``_model`` plus ``tag`` (reset to 0) and the counter."""
    mem, tag = [-1] * 32, [0] * 4
    qa, qt, qc = [], [], []
    for t, (a, x, d, e) in enumerate(zip(ra, wa, wd, we)):
        qa.append(mem[int(a)])
        qt.append(tag[int(a) & 3])
        qc.append(t)
        if e:
            mem[int(x)] = int(d)
            tag[int(x) & 3] = int(d)
    return np.array(qa), np.array(qt), np.array(qc)


def _offset(got, want, n):
    """The constant offset at which every defined ``want`` agrees, or None."""
    for k in range(0, 6):
        if all(want[t] < 0 or int(got[t + k]) == want[t] for t in range(n - k)):
            return k
    return None


@needs_csim
def test_mixed_reset_csim():
    n = 64
    ra, wa, wd, we = _trace(n)
    qa, qt, qc = (np.zeros(n, dtype=np.uint16) for _ in range(3))
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_mixed(n), target="systemc", mode="csim", project=tmp)
        mod(ra, wa, wd, we, qa, qt, qc)
    want = _model_mixed(ra, wa, wd, we)
    ks = [_offset(g, w, n) for g, w in zip((qa, qt, qc), want)]
    assert None not in ks, f"no constant offset matches: {ks}, got {qa[:8]} {qt[:8]} {qc[:8]}"
    print(f"mixed csim: mem/tag/cnt match at offsets {ks}")


def test_uint_stateful_unsigned():
    """A ``UInt`` ``Stateful`` is declared unsigned on every backend (the
    global carries the ``unsigned`` attribute, as a local alloc does). Before,
    every emitter declared it signed: ``ac_int<16, true>``, ``int16_t``
    (u2_unreset_impl_2026-10-02.rst, H4)."""
    ir = str(df.customize(_regfile(4)).module)
    assert re.search(r"memref\.global .*__stateful_rf_0_mem\w* .*\{[^}]*unsigned", ir), ir
    code = df.build(_regfile(8), target="systemc").hls_code
    assert re.search(r"sc_signal< ac_int<16, false> > __stateful_rf_0_mem\w*\[32\];", code), code
    assert "ac_int<16, true> > __stateful" not in code
    code = df.build(_regfile_stateful(8, D16, True), target="systemc").hls_code
    assert re.search(r"sc_signal< ac_int<16, false> > __stateful_rf_0_mem\w*\[32\];", code), code
    code = df.build(_plain_regfile(8, reset=True), target="systemc").hls_code
    assert re.search(r"uint16_t __stateful_rf_0_mem\w*\[32\];", code), code
    for tgt in ("vhls", "catapult"):
        code = df.build(_plain_regfile(8, reset=True), target=tgt).hls_code
        assert re.search(r"uint16_t __stateful_rf_0_mem\w*\[32\]", code), (tgt, code)
        assert "int16_t __stateful" not in code.replace("uint16_t __stateful", ""), tgt


@pytest.mark.parametrize(
    "make,why",
    [
        (_regfile_rmw, "not storage"),
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


def test_unreset_stream_kernel_emit():
    """C2 (README D-14, lowering extended): a kernel with Stream ports holds
    unreset storage as a plain module member -- not an ``sc_signal``, no
    ``wr`` method, no reset-action write -- and ``run.tcl`` scopes
    ``-RESET_CLEARS_ALL_REGS no`` to that kernel's thread. An array-port kernel
    (one ``@df.kernel`` with boundary arrays, the DMA's ``bits`` form) the same."""
    for make, name in ((_regfile_stream, "rf_0"), (lambda n: _plain_regfile(n, reset=False), "rf_0")):
        with tempfile.TemporaryDirectory() as tmp:
            mod = df.build(make(8), target="systemc", mode="csyn", project=tmp)
            tcl = open(tmp + "/run.tcl").read()
        rf = _kernel_module(mod.hls_code, name)
        sym = re.search(r"\n\s*(?:uint16_t|ac_int<16, false>) (__stateful_rf_0_mem\w*)\[32\];  "
                        r"// @ Stateful\(reset=False\), unreset member", rf)
        assert sym, rf
        mem = sym.group(1)
        assert f"> {mem}[" not in rf  # not an sc_signal array
        assert "SC_METHOD(wr)" not in rf and "void wr()" not in rf
        assert f"// allo unreset storage: {mem}" in rf
        assert "// allo unreset process: run" in rf
        assert not re.search(re.escape(mem) + r"\w*\[_sr\w*\] = ", rf)  # no reset action
        assert f"// {mem}: @ Stateful(reset=False), not reset (D-14)" in rf
        thread = rf[rf.index("void run()") :]
        assert re.search(re.escape(mem) + r"\[.*\] = ", thread)  # the thread writes it
        assert f"go compile\ndirective set /top/{name}/run -RESET_CLEARS_ALL_REGS no\n" in tcl, tcl
        assert tcl.count("RESET_CLEARS_ALL_REGS") == 1


def _regfile_stream_comb(n):
    """A comb output reading unreset storage in a kernel that is not Wire-only:
    comb storage is a signal the thread writes, which Catapult resets
    (CIN-233) -- the one form that genuinely cannot be unreset there."""

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], QA: D16[n]):
        w_ra: Wire[A5]
        s_wa: Stream[A5, 2]
        s_wd: Stream[D16, 2]
        w_qa: Wire[D16, comb]

        @df.kernel(mapping=[1], args=[RA, WA, WD])
        def src(ra: A5[n], wa: A5[n], wd: D16[n]):
            for t in range(n):
                w_ra.put(ra[t])
                s_wa.put(wa[t])
                s_wd.put(wd[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = s_wa.get()
                x: int32 = x5
                d: D16 = s_wd.get()
                w_qa.put(mem[a])
                mem[x] = d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def test_unreset_stream_kernel_comb_refused():
    with pytest.raises(Exception) as ex:
        df.build(_regfile_stream_comb(8), target="systemc")
    msg = str(ex.value)
    assert "unreset storage `mem` (rf_0): a comb output reads it" in msg, msg


def _stream_calls(n):
    """Two calls with no reset between them (D-11): the first writes every
    address, the second reads every address and writes nothing."""
    wa = np.array([i % 32 for i in range(n)], np.uint8)
    wd = np.array([(i * 977 + 5) & 0xFFFF for i in range(n)], np.uint16)
    return [(wa, wd, np.ones(n, np.uint8)), (wa, wd, np.zeros(n, np.uint8))]


def _run_calls(mod, n):
    outs = []
    for wa, wd, we in _stream_calls(n):
        q = np.zeros(n, np.uint16)
        mod(wa, wd, we, q)
        outs.append(q)
    return outs


@needs_csim
def test_unreset_stream_kernel_csim():
    """C2: a Stream-port kernel holding unreset storage builds and runs in
    csim, and the storage survives the reset-free call sequence exactly as
    the simulator shows -- every iteration of both calls equal, the
    never-written words of the first call masked (undefined, D-14)."""
    n = 64
    sim = _run_calls(df.build(_regfile_stream(n), target="simulator"), n)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_regfile_stream(n), target="systemc", mode="csim", project=tmp)
        sc = _run_calls(mod, n)
    # call 1 reads mem[x] before writing it: defined only once x was written
    seen, defined = set(), []
    for wa, _, _ in _stream_calls(n)[:1]:
        for x in wa:
            defined.append(int(x) in seen)
            seen.add(int(x))
    defined = np.array(defined)
    assert np.array_equal(sim[0][defined], sc[0][defined]), (sim[0], sc[0])
    assert np.array_equal(sim[1], sc[1]), (sim[1], sc[1])
    # and the second call returns what the first wrote (the last write per word)
    last = {}
    wa, wd, _ = _stream_calls(n)[0]
    for x, d in zip(wa, wd):
        last[int(x)] = int(d)
    assert [int(v) for v in sc[1]] == [last[int(x)] for x in wa]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
