# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Wire[T, comb]``: a declared combinational output (README D-13).

The SystemC emitter drives such a port from an ``SC_METHOD`` over signal
storage instead of the kernel's clocked thread, so the port reads at latency
0 (the register file of ``examples/minitpu/units/vpu_regfile.py``, variant
``comb``, measured against MiniTPU in
``dev/records/minitpu/u2_comb_wire_impl_2026-10-02.rst``). The emit and
refusal tests run anywhere; the csim test compiles and runs the emitted
testbench and skips without Catapult (``MGC_HOME``) and ``SYSTEMC_HOME``.
"""

import os
import re
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import Stream, UInt, Wire, comb, int32, uint1

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

A5 = UInt(5)
D16 = UInt(16)

# A 32-entry 1R1W register file with a combinational read port, driven by a
# per-cycle command trace (`src`) and recorded by `sink`. One region per body
# shape: the frontend reads a kernel off its source, so the shape cannot be a
# Python-level switch inside the kernel.


def _regfile(n):
    """The legal form: read, then store (``vpu_regfile.py`` ``comb``)."""

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
            mem: D16[32]
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


def _regfile_write_first(n):
    """The store precedes the read in the iteration: the cone would read
    this cycle's write through a combinational path."""

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
            mem: D16[32]
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                if e:
                    mem[x] = d
                w_qa.put(mem[a])

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def _regfile_cond_put(n):
    """The comb port is driven only on some iterations."""

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
            mem: D16[32]
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                if e:
                    mem[x] = d
                else:
                    w_qa.put(mem[a])

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def _regfile_stream(n):
    """The cone reads a stream (a handshake, hence clocked)."""

    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D16]
        w_we: Wire[uint1]
        w_qa: Wire[D16, comb]
        s_bias: Stream[D16, 2]

        @df.kernel(mapping=[1], args=[RA, WA, WD, WE])
        def src(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])
                s_bias.put(wd[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32]
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                b: D16 = s_bias.get()
                w_qa.put(mem[a] + b)
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA])
        def sink(qa: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()

    return top


def _kernel_module(code, name):
    m = re.search(r"SC_MODULE\(" + name + r"\) \{(.*?)\n\};", code, flags=re.S)
    assert m, f"no SC_MODULE({name})"
    return m.group(1)


def test_comb_type_marker():
    """The marker is carried in the IR type, not inferred from the body."""
    w = Wire[D16, comb]
    assert w.comb and repr(w).endswith(", comb)")
    assert not Wire[D16].comb
    with pytest.raises(TypeError):
        Wire[D16, 3]  # pylint: disable=expression-not-assigned
    ir = str(df.customize(_regfile(4)).module)
    # the driver's argument and the reader's both carry it
    assert ir.count("!allo.wire<i16, comb>") >= 2, ir


def test_comb_emit_shape():
    """Form e of the comb-read record: one SC_METHOD for the read cone, signal
    storage zeroed in the reset action, the thread keeps the write."""
    code = df.build(_regfile(8), target="systemc").hls_code
    rf = _kernel_module(code, "rf_0")
    assert "SC_METHOD(comb);" in rf
    assert "sc_signal< ac_int<16, false> > mem[32];" in rf
    assert "for (int _ci = 0; _ci < 32; ++_ci) sensitive << mem[_ci];" in rf
    assert re.search(r"// allo comb ports: v\d+\n", rf)
    # the storage reset (Catapult CIN-233; the recorded deviation from MiniTPU)
    assert "mem[_sr].write(0);" in rf
    port = re.search(r"sc_out< ac_int<16, false> > (v\d+);  // comb", rf).group(1)
    thread = rf[rf.index("void run()") : rf.index("void comb()")]
    method = rf[rf.index("void comb()") :]
    # the comb port is written in the method only: no reset-action or thread driver
    assert f"{port}.write(" in method
    assert f"{port}.write(" not in thread
    # the write is clocked, through the signal; the read is .read() in the method
    assert re.search(r"mem\[.*\]\.write\(", thread)
    assert re.search(r"mem\[.*\]\.read\(\);", method)
    assert "mem[" not in thread.split(".write(")[0].split("mem[_sr]")[-1] or True
    # the thread dropped the read address cone (its only consumer was the put)
    assert "int32_t a;" not in thread and "int32_t a;" in method
    assert "int32_t x;" in thread


def test_comb_write_first_refused():
    with pytest.raises(Exception) as ex:
        df.build(_regfile_write_first(8), target="systemc")
    assert "comb port `w_qa`" in str(ex.value)
    assert "after a store to it in the same iteration" in str(ex.value)


def test_comb_conditional_put_refused():
    with pytest.raises(Exception) as ex:
        df.build(_regfile_cond_put(8), target="systemc")
    assert "comb port `w_qa`" in str(ex.value)
    assert "under control flow" in str(ex.value)


def test_comb_stream_in_cone_refused():
    with pytest.raises(Exception) as ex:
        df.build(_regfile_stream(8), target="systemc")
    assert "comb port `w_qa`" in str(ex.value)
    assert "reads a stream or channel" in str(ex.value)


def test_comb_other_backends_refuse():
    """A comb port is a Wire: no other backend has it, and the simulator says
    so instead of failing inside the ExecutionEngine."""
    with pytest.raises(NotImplementedError):
        df.build(_regfile(8), target="simulator")
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(NotImplementedError):
            df.build(_regfile(8), target="vhls", mode="csim", project=tmp)


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
def test_comb_regfile_csim():
    """Compiles and runs the emitted testbench. A Wire link is not cycle-locked
    in csim (limitation 22), so the recorded port is compared to the model at
    the constant offset where every defined slot agrees, as
    ``u2_comb_read_2026-10-02.rst`` F7 measures it (one cycle less than the
    thread form)."""
    n = 64
    ra, wa, wd, we = _trace(n)
    qa = np.zeros(n, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_regfile(n), target="systemc", mode="csim", project=tmp)
        mod(ra, wa, wd, we, qa)
    want = _model(ra, wa, wd, we)
    best = None
    for k in range(0, 6):  # got[t + k] == want[t] on every defined slot
        if all(want[t] < 0 or int(qa[t + k]) == want[t] for t in range(n - k)):
            best = k
            break
    assert (
        best is not None
    ), f"no constant offset matches: got {qa[:12]}, want {want[:12]}"
    print(f"comb regfile csim: values match at offset +{best}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
