# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12: ``compose.Memory`` declares its ports; each port has one owner.

Composition-level refusals (two owners, wrong direction, too many accesses per
iteration, an access under control flow, a dangling port, binding the memory
itself, the collision rule, Vitis) and acceptance, on the register-file
prototype ``examples/minitpu/units/vpu_regfile_d12.py``; then the four-unit
register file built and run on the simulator and in SystemC csim (both
lowerings, Stream links: exact per iteration), and the Wire form's csim
compile-and-run (limitation 22: compared at a constant offset).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401
from examples.minitpu.units import vpu_regfile_d12 as d

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)


# --- units that break one rule each ----------------------------------------


@unit(memories=("vreg.rc",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c_writes(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5
        qc.put(mem[c])
        mem[c] = 0


@unit(memories=("vreg.rc",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c_twice(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5
        qc.put(mem[c] + mem[0])


@unit(memories=("vreg.rc",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c_cond(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5
        if c > 0:
            qc.put(mem[c])
        else:
            qc.put(0)


@unit(memories=("vreg.ra",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c_on_ra(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5
        qc.put(mem[c])


@unit(memories=("vreg",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c_whole(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5
        qc.put(mem[c])


@unit(memories=("vreg.w",), reads=("wa", "wd", "we"), writes=(), parameters=("N", "W"))
def wb_reads(mem):
    for _ in range(N):
        x5: UInt(5) = wa.get()
        x: int32 = x5
        d: UInt(W) = wd.get()
        e: uint1 = we.get()
        y: UInt(W) = mem[x]
        if e:
            mem[x] = d + y


def _swap(old, new):
    units = tuple(
        new if u is old else u for u in (d.src, d.rd_a, d.rd_b, d.rd_c, d.wb, d.sink)
    )
    return d.architecture(8, units=units)


def _refused(fn, *needles):
    with pytest.raises((AssertionError, NotImplementedError)) as e:
        fn()
    for x in needles:
        assert x in str(e.value), str(e.value)
    return str(e.value)


def test_accepts_regfile():
    a = d.architecture(8)
    assert a.plan("systemc") == {"vreg": "server"}
    assert a.plan("systemc", {"vreg": "replica"}) == {"vreg": "replica"}
    assert a.port_kernels("systemc") == [
        "rd_a_0",
        "rd_b_0",
        "rd_c_0",
        "wb_0",
        "vreg_mem_0",
    ]
    src = a.source("systemc", {"vreg": "server"})
    assert (
        "def vreg_mem():" in src and "mem: UInt(W)[32] @ Stateful(reset=False)" in src
    )
    assert "vreg_ra_q: Wire[UInt(W), comb]" in src  # latency 0: a comb read
    sim = a.source("simulator", {"vreg": "server"})
    assert "Wire" not in sim and "Stateful" not in sim
    rep = a.source("systemc", {"vreg": "replica"})
    assert rep.count("@ Stateful(reset=False)") == 3  # one copy per read port
    m = a.memory_manifest("systemc", {"vreg": "replica"})["vreg"]
    assert m["lowering"] == "replica" and m["ports"]["ra"]["owner"] == "rd_a"
    assert m["ports"]["w"]["visible"] == 1 and m["ports"]["ra"]["latency"] == 0


def test_two_owners_refused():
    _refused(
        lambda: _swap(d.rd_c, rd_c_on_ra),
        "port 'vreg.ra' is bound by both rd_a and rd_c_on_ra",
        "exactly one owner",
    )


def test_dangling_port_refused():
    units = (d.src, d.rd_a, d.rd_b, d.wb, d.sink)
    with pytest.raises(AssertionError) as e:
        d.architecture(8, units=units)
    # rd_c's channels go unread first, unless declared away: check the port rule directly
    assert "nothing" in str(e.value) or "vreg.rc has no owner" in str(e.value)
    vreg = Memory(
        "vreg",
        "UInt(W)",
        rows="32",
        reset=False,
        ports=d.VREG.ports + (Port("rd", "r", latency=0),),
    )
    _refused(lambda: d.architecture(8, vreg=vreg), "port vreg.rd has no owner")


def test_direction_refused():
    _refused(
        lambda: _swap(d.rd_c, rd_c_writes),
        "rd_c_writes",
        "vreg.rc",
        "writes through a read-only port",
    )
    _refused(
        lambda: _swap(d.wb, wb_reads),
        "wb_reads",
        "vreg.w",
        "reads through a write-only port",
    )


def test_accesses_per_iteration_refused():
    _refused(
        lambda: _swap(d.rd_c, rd_c_twice), "vreg.rc", "2 reads per iteration", "count=1"
    )
    _refused(lambda: _swap(d.rd_c, rd_c_cond), "vreg.rc", "under control flow")


def test_bind_whole_memory_refused():
    _refused(lambda: _swap(d.rd_c, rd_c_whole), "binds memory 'vreg' itself", "vreg.ra")


def test_collision_rule():
    _refused(
        lambda: Memory(
            "vm", "UInt(8)", rows="8", ports=(Port("c", "rw", 3), Port("d", "rw", 2))
        ),
        "collision='refuse'",
    )
    Memory(
        "vm",
        "UInt(8)",
        rows="8",
        collision="undefined",
        ports=(Port("c", "rw", 3), Port("d", "rw", 2)),
    )
    _refused(lambda: Port("x", "w", latency=1), "a write port has no read latency")
    _refused(lambda: Port("x", "r"), "declares its read latency")


def test_vitis_refuses_multi_owner():
    _refused(
        lambda: d.architecture(8).region("vhls"),
        "Vitis refuses a memory with more than one owner",
        "rd_a",
    )


def test_allo_memory_latency_depth_refused():
    from allo.memory import Memory as M

    _refused(lambda: M(resource="LUTRAM", latency=0), "README D-12")
    _refused(lambda: M(resource="LUTRAM", depth=32), "README D-12")
    M(resource="LUTRAM", storage_type="RAM_1WNR")  # the fields every backend honours


def test_word_array_two_rw_ports():
    """The two-port VMEM (``vpu_word_array_d12``): two ``rw`` ports with
    different read latencies, two owners, the collision an obligation."""
    from examples.minitpu.units import vpu_word_array_d12 as w

    a = w.architecture(8)
    assert a.plan("systemc") == {"vmem": "server"}
    assert a.port_kernels("systemc") == ["port_c_0", "port_d_0", "vmem_mem_0"]
    m = a.memory_manifest("systemc")["vmem"]
    assert m["collision"] == "obligation" and "issue #21" in m["obligation"]
    assert m["ports"]["c"]["latency"] == 3 and m["ports"]["d"]["latency"] == 2
    src = a.source("systemc")
    assert "_c_p: UInt(W)[3]" in src and "_d_p: UInt(W)[2]" in src  # the pipes as data
    # the obligation is not optional
    _refused(
        lambda: Architecture(
            name="x",
            parameters=a.parameters,
            memories=a.memories,
            channels=a.channels,
            units=a.units,
        ),
        "collision='obligation'",
        "Architecture(obligations=",
    )
    # the replica lowering takes one `w` port only
    _refused(
        lambda: a.plan("systemc", {"vmem": "replica"}),
        "replica lowering takes one `w` port",
    )
    _refused(lambda: a.region("vhls"), "Vitis refuses")


def _trace(n, seed=0):
    rng = np.random.default_rng(seed)
    ra, rb, rc, wa = (rng.integers(0, 32, n).astype(np.uint8) for _ in range(4))
    wd = rng.integers(0, 1 << 16, n).astype(np.uint16)
    we = (rng.random(n) < 0.6).astype(np.uint8)
    return ra, rb, rc, wa, wd, we


def _model(ra, rb, rc, wa, wd, we):
    """Read, then write, per cycle; -1 marks a read of a never-written entry."""
    mem = [-1] * 32
    out = [[], [], []]
    for t in range(len(ra)):
        for k, r in enumerate((ra, rb, rc)):
            out[k].append(mem[int(r[t])])
        if we[t]:
            mem[int(wa[t])] = int(wd[t])
    return [np.array(o) for o in out]


def _check(outs, want, k=0):
    n = len(want[0])
    return all(
        want[j][t] < 0 or int(outs[j][t + k]) == want[j][t]
        for j in range(3)
        for t in range(n - k)
    )


@pytest.mark.parametrize("lowering", ["server", "replica"])
def test_regfile_simulator(lowering):
    n = 48
    ins = _trace(n, 1)
    outs = [np.zeros(n, dtype=np.uint16) for _ in range(3)]
    mod = d.architecture(n).build("simulator", {"vreg": lowering})
    mod(*ins, *outs)
    assert _check(outs, _model(*ins))


@needs_csim
@pytest.mark.parametrize("lowering", ["server", "replica"])
def test_regfile_csim(lowering):
    """The four-unit register file through SystemC csim: Stream links (exact
    per iteration), then the Wire form (D-13 comb reads, D-14 unreset
    storage), which compiles and runs and is compared at the constant offset
    limitation 22 gives a Wire link in csim."""
    n = 48
    ins = _trace(n, 2)
    want = _model(*ins)
    with tempfile.TemporaryDirectory() as tmp:
        outs = [np.zeros(n, dtype=np.uint16) for _ in range(3)]
        a = d.architecture(n)
        import allo.dataflow as df

        mod = df.build(
            a.region("simulator", {"vreg": lowering}),
            target="systemc",
            mode="csim",
            project=os.path.join(tmp, "s"),
        )
        mod(*ins, *outs)
        assert _check(outs, want)
        outs = [np.zeros(n, dtype=np.uint16) for _ in range(3)]
        prj = os.path.join(tmp, "w")
        mod = a.build("systemc", {"vreg": lowering}, mode="csim", project=prj)
        manifest = json.load(open(os.path.join(prj, "memory.json")))
        assert manifest["vreg"]["lowering"] == lowering
        mod(*ins, *outs)
        offs = []
        for j in range(3):  # per port: the comb form measured +3/+2/+2 (csim_offset.py)
            best = next(
                (
                    k
                    for k in range(6)
                    if all(
                        want[j][t] < 0 or int(outs[j][t + k]) == want[j][t]
                        for t in range(n - k)
                    )
                ),
                None,
            )
            assert (
                best is not None
            ), f"port {j}: no constant offset matches: {outs[j][:12]} vs {want[j][:12]}"
            offs.append(best)
        print(
            f"d12 {lowering} wire csim: values match at offsets {offs} (limitation 22)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
