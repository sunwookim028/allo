# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12's stated lowerings as ASIC-flavoured swap-ins
(``dev/records/minitpu/asic_memories_2026-10-04.rst``, stage 3).

``registers`` is the flop array with combinational read muxes (the D-12
prototype's ``server``, kept as an alias); ``sram`` maps the same server onto
a declared macro (``Memory(impl=Sram(...))``) and refuses a port the macro
cannot honour, naming it; ``replica`` is FPGA-only (LUTRAM-style copies) and
is refused unless the target says FPGA. ``memory.json`` reports the choice,
and the SystemC build's ``run.tcl`` adds the macro's Catapult library and
the ``MAP_TO_MODULE`` directive.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from allo.compose import Architecture, Memory, Port, Sram
from examples.minitpu.units import vpu_regfile_d12 as rf
from examples.minitpu.units import vpu_word_array_d12 as wa

REC = os.path.join(
    os.path.dirname(__file__), "..", "..", "dev", "records", "minitpu",
    "asic_memories_2026-10-04", "openram")
V = os.path.join(REC, "sram_2rw_64x32_freepdk45.v")
LIB = os.path.join(REC, "sram_2rw_64x32_freepdk45_TT_1p0V_25C.lib")


def _refused(fn, *needles):
    with pytest.raises((AssertionError, NotImplementedError)) as e:
        fn()
    msg = str(e.value)
    for n in needles:
        assert n in msg, f"{n!r} not in {msg!r}"
    return msg


def _small(impl, **kw):
    """The two-port VMEM at the macro's size with an explicit lowering."""
    return wa.architecture(16, "small", impl=impl, **kw)


def test_sram_from_openram():
    s = Sram.from_openram(V, LIB)
    assert s.module == "sram_2rw_64x32_freepdk45"
    assert (s.rows, s.width, s.ports) == (32, 64, ("rw", "rw"))
    assert (s.read_latency, s.visible) == (1, 1)
    assert s.area_um2 == pytest.approx(28295.48, abs=0.01)  # layout 263.275 x 107.475 um
    m = s.manifest()
    assert m["ports"] == ["rw", "rw"] and m["catapult_lib"] is None


def test_registers_is_the_default_and_servers_alias():
    a = rf.architecture(8)  # VREG declares impl="registers"
    assert a.plan("systemc") == {"vreg": "registers"}
    assert a.plan("systemc", {"vreg": "server"}) == {"vreg": "registers"}
    m = a.memory_manifest("systemc")["vreg"]
    assert m["lowering"] == "registers" and "flop array" in m["implementation"]
    assert m["technology"] == "asic"
    assert a.port_kernels("systemc")[-1] == "vreg_mem_0"


def test_replica_is_fpga_only():
    a = rf.architecture(8)
    msg = _refused(lambda: a.plan("systemc", {"vreg": "replica"}), "FPGA-only", "ASIC target")
    assert "technology='fpga'" in msg
    _refused(lambda: a.plan("simulator", {"vreg": "replica"}), "FPGA-only", "says nothing")
    _refused(lambda: a.plan(None, {"vreg": "replica"}), "FPGA-only")
    assert a.plan("systemc", {"vreg": "replica"}, technology="fpga") == {"vreg": "replica"}
    assert a.plan("simulator", {"vreg": "replica"}, technology="fpga") == {"vreg": "replica"}
    # declared on the memory, the same rule
    vreg = Memory(**{**{f.name: getattr(rf.VREG, f.name) for f in rf.VREG.__dataclass_fields__.values()},
                     "impl": "replica"})
    b = rf.architecture(8, vreg=vreg)
    _refused(lambda: b.plan("systemc"), "FPGA-only")
    assert b.plan("systemc", technology="fpga") == {"vreg": "replica"}
    assert b.memory_manifest("systemc", technology="fpga")["vreg"]["technology"] == "fpga"
    _refused(lambda: a.plan("systemc", technology="asic2"), "technology")


def test_sram_accepted_on_the_small_vmem():
    s = Sram.from_openram(V, LIB, catapult_lib="/nowhere/sram.lib")
    a = _small(s)
    assert a.plan("systemc") == {"vmem": "sram"}
    assert a.sram_ports(a.memories[-1]) == {"c": 0, "d": 1}
    src = a.source("systemc")
    assert "def vmem_mem():" in src and "mem: UInt(W)[32]\n" in src  # plain array: the macro
    assert "Stateful" not in src.split("def vmem_mem():")[1]
    assert "_c_p[0] = mem[_c_i]" in src and "mem[_d_i] = _d_d" in src  # one access per port
    m = a.memory_manifest("systemc")["vmem"]
    assert m["lowering"] == "sram" and m["macro"]["module"] == s.module
    assert m["ports"]["c"]["macro_port"] == {"index": 0, "kind": "rw"}
    assert "latency 1" in m["ports"]["c"]["read"] and "3-deep" in m["ports"]["c"]["read"]
    assert "SRAM macro" in m["storage"]
    # the simulator runs it untimed, like every lowering
    assert a.plan("simulator") == {"vmem": "sram"}
    assert a.memory_manifest("simulator")["vmem"]["status"] == "untimed"
    # an explicit lowering still overrides the declared one (the d12_server variants)
    assert a.plan("systemc", {"vmem": "server"}) == {"vmem": "registers"}


def test_sram_refusals_name_the_port():
    s = Sram.from_openram(V, LIB)
    rows = "32"
    ports = (Port("c", "rw", latency=3), Port("d", "rw", latency=2))

    def mem(**kw):
        d = dict(name="vmem", dtype="UInt(W)", rows=rows, ports=ports,
                 collision="obligation", reset=False, impl=s)
        d.update(kw)
        return Memory(**d)

    def arch(m):
        base = _small(s)
        return Architecture(name=base.name, parameters=base.parameters,
                            memories=base.memories[:-1] + (m,), channels=base.channels,
                            units=base.units, obligations=base.obligations)

    _refused(lambda: arch(mem(reset=True)).plan("systemc"), "is not reset", "reset=False")
    _refused(lambda: arch(mem(ports=(Port("c", "rw", latency=0), Port("d", "rw", latency=2)))).plan("systemc"),
             "port c", "latency=0", "asynchronous")
    _refused(lambda: arch(mem(ports=(Port("c", "rw", latency=3), Port("d", "rw", latency=2, visible=2)))).plan("systemc"),
             "port d", "visible=2")
    _refused(lambda: arch(mem(ports=(Port("c", "rw", latency=3, count=2), Port("d", "rw", latency=2)))).plan("systemc"),
             "port c", "count=2")
    _refused(lambda: arch(mem(rows="64")).plan("systemc"), "64 rows", "32 rows")
    _refused(lambda: arch(mem(dtype="UInt(128)")).plan("systemc"), "128-bit", "64")
    # two declared ports on a one-port macro: the second is named
    one = Sram("sram_1rw", V, LIB, 32, 64, ("rw",))
    _refused(lambda: _small(one).plan("systemc"), "port d", "no port of the macro", "['rw']")
    # an `r` port takes a macro `r` port first, then an `rw` one; a `w` port never takes an `r`
    rw_r = Sram("sram_1rw1r", V, LIB, 32, 64, ("rw", "r"))
    a = _small(rw_r)
    _refused(lambda: a.plan("systemc"), "port d", "no port of the macro")  # d is rw: only r left
    # no macro behind the name
    _refused(lambda: Memory("vmem", "UInt(W)", rows=rows, ports=ports, reset=False,
                            collision="obligation", impl="sram"), "impl='sram' names no macro")
    _refused(lambda: _small("registers").plan("systemc", {"vmem": "sram"}), "needs the macro")
    _refused(lambda: Memory("x", "UInt(8)", impl="registers"), "impl= needs rows=")
    _refused(lambda: Sram("m", V, LIB, 32, 64, ("rw",), read_latency=0), "synchronous")
    # Vitis has no macro path
    _refused(lambda: _small(s).build("vhls"), "Vitis refuses")


def test_sram_systemc_emission_adds_library_and_map():
    """``build("systemc", mode="csyn")`` emits ``run.tcl`` with the macro's
    library and the server's array mapped onto it, and ``memory.json`` with
    the macro; no Catapult is run."""
    s = Sram.from_openram(V, LIB, catapult_lib="/nowhere/memgen/sram_2rw_64x32_freepdk45.lib")
    a = _small(s)
    kernels = a.port_kernels("systemc")
    assert kernels == ["port_c_0", "port_d_0", "vmem_mem_0"]
    with tempfile.TemporaryDirectory() as tmp:
        prj = os.path.join(tmp, "p")
        cfg = {"clock_period": 3.33, "synth_group": {"name": "wa_d12g", "kernels": kernels}}
        a.build("systemc", mode="csyn", project=prj, configs=cfg)
        tcl = open(os.path.join(prj, "run.tcl"), encoding="utf-8").read()
        i = tcl.index("go compile")
        assert ("solution library add sram_2rw_64x32_freepdk45 -file "
                "/nowhere/memgen/sram_2rw_64x32_freepdk45.lib") in tcl[i:]
        assert ("directive set /wa_d12g/vmem_mem_0/run/mem:rsc -MAP_TO_MODULE "
                "sram_2rw_64x32_freepdk45.sram_2rw_64x32_freepdk45") in tcl[i:]
        assert tcl.index("-MAP_TO_MODULE") < tcl.index("go assembly")
        mj = json.load(open(os.path.join(prj, "memory.json"), encoding="utf-8"))["vmem"]
        assert mj["lowering"] == "sram" and mj["catapult"]["rsc"] == "/wa_d12g/vmem_mem_0/run/mem:rsc"
        assert mj["macro"]["area_um2"] == pytest.approx(28295.48, abs=0.01)
    # without a built library and without Catapult, the build says what to do
    if not os.environ.get("MGC_HOME"):
        with tempfile.TemporaryDirectory() as tmp:
            _refused_rt(lambda: _small(Sram.from_openram(V, LIB)).build(
                "systemc", mode="csyn", project=os.path.join(tmp, "q"), configs=cfg))


def _refused_rt(fn):
    with pytest.raises(RuntimeError) as e:
        fn()
    assert "memory library" in str(e.value)


def test_word_array_variants_name_the_macro():
    """The harness variants: ``d12_sram`` is the `small` instance's macro and
    refuses an instance without one, naming it."""
    f = wa.make("simulator", lowering="sram")
    assert f.__name__ == "d12_sram_simulator"
    _refused(lambda: f(8, 64, "narrow"), "no SRAM macro", "narrow")
    f(8, 64, "small")  # builds the region
    assert wa.architecture(8, "narrow").plan("systemc") == {"vmem": "registers"}
    assert wa.architecture(8, "small").plan("systemc") == {"vmem": "sram"}
    assert wa.architecture(8, "small").plan("systemc", {"vmem": "server"}) == {"vmem": "registers"}


# asic_memories_2026-10-04.rst s.5-6: banks, port placement, read/write resolution
V512 = os.path.join(REC, "sram_2rw_64x512_freepdk45.v")
LIB512 = os.path.join(REC, "sram_2rw_64x512_freepdk45_TT_1p0V_25C.lib")
V1R1W = os.path.join(REC, "sram_1r1w_64x32_freepdk45.v")
LIB1R1W = os.path.join(REC, "sram_1r1w_64x32_freepdk45_TT_1p0V_25C.lib")
V1RW1R = os.path.join(REC, "sram_1rw1r_64x32_freepdk45.v")
LIB1RW1R = os.path.join(REC, "sram_1rw1r_64x32_freepdk45_TT_1p0V_25C.lib")


def _with_vmem(base, **kw):
    """``base`` with its ``vmem`` declaration changed (collision, ports, ...)."""
    m = base.memories[-1]
    d = {f: getattr(m, f) for f in m.__dataclass_fields__}
    d.update(kw)
    return Architecture(name=base.name, parameters=base.parameters,
                        memories=base.memories[:-1] + (Memory(**d),), channels=base.channels,
                        units=base.units, obligations=base.obligations)


def test_sram_banked_mid():
    """``mid`` (4,096 x 64 b) on eight 512-word macros: declared, never inferred;
    ``run.tcl`` adds ``-BLOCK_SIZE 512`` to the map, ``memory.json`` the banks."""
    a = wa.architecture(8, "mid")
    assert a.plan("systemc") == {"vmem": "sram"}
    m = a.memory_manifest("systemc")["vmem"]
    assert m["macro"]["banks"] == 8 and m["macro"]["rows"] == 512
    assert m["macro"]["total_area_um2"] == pytest.approx(8 * 103782.068775, abs=0.1)
    assert m["port_map"] == {"c": "0 (rw)", "d": "1 (rw)"}
    s512 = Sram.from_openram(V512, LIB512, catapult_lib="/nowhere/memgen/m512.lib")
    _refused(lambda: wa.architecture(8, "mid", impl=s512).plan("systemc"),
             "4096 rows", "512 rows x 1 bank", "Sram.banked(8)", "never inferred")
    _refused(lambda: wa.architecture(8, "mid", impl=s512.banked(4)).plan("systemc"),
             "4096 rows", "x 4 banks")
    _refused(lambda: wa.architecture(8, "mid", impl=s512.banked(9)).plan("systemc"),
             "need 8 bank(s)", "declares 9")
    _refused(lambda: Sram.from_openram(V512, LIB512).banked(0), "banks=0")
    b = wa.architecture(8, "mid", impl=s512.banked(8))
    kernels = b.port_kernels("systemc")
    with tempfile.TemporaryDirectory() as tmp:
        prj = os.path.join(tmp, "p")
        cfg = {"clock_period": 3.33, "synth_group": {"name": "wa_d12g", "kernels": kernels}}
        b.build("systemc", mode="csyn", project=prj, configs=cfg)
        tcl = open(os.path.join(prj, "run.tcl"), encoding="utf-8").read()
        rsc = "/wa_d12g/vmem_mem_0/run/mem:rsc"
        assert f"directive set {rsc} -BLOCK_SIZE 512" in tcl
        assert tcl.index("-MAP_TO_MODULE") < tcl.index("-BLOCK_SIZE") < tcl.index("go assembly")
        mj = json.load(open(os.path.join(prj, "memory.json"), encoding="utf-8"))["vmem"]
        assert mj["catapult"]["block_size"] == 512 and mj["macro"]["banks"] == 8
    # one macro: no BLOCK_SIZE
    from allo.backend.catapult import memory_directives  # noqa: PLC0415
    one = _small(Sram.from_openram(V, LIB, catapult_lib="/x.lib")).sram_configs("systemc")
    assert "BLOCK_SIZE" not in memory_directives(one)


def test_sram_port_kinds_placement():
    """README D-12's port kinds on a macro: an `r` macro port serves only `r`,
    a `w` only `w`, and a declared `rw` port only an `rw` macro port --
    placed by kind, or pinned with ``Sram.mapped`` and checked."""
    s11 = Sram.from_openram(V1R1W, LIB1R1W, catapult_lib="/x.lib")
    assert s11.ports == ("w", "r") and s11.area_um2 == pytest.approx(18562.55, abs=0.01)
    # MiniTPU's VMEM (two rw ports) does not fit a 1R1W or a 1RW+1R macro
    _refused(lambda: _small(s11).plan("systemc"), "port c (rw)", "rw macro port")
    s1r = Sram.from_openram(V1RW1R, LIB1RW1R, catapult_lib="/x.lib")
    assert s1r.ports == ("rw", "r")
    _refused(lambda: _small(s1r).plan("systemc"), "port d (rw)", "rw macro port")
    _refused(lambda: _small(s1r.mapped(c=1)).plan("systemc"),
             "port c (rw) cannot sit on port 1", "'r' port", "only an rw macro port")
    # the port-kind probe (compute reads, DMA writes) fits both
    a = _small(s11, kinds=("r", "w"))
    assert a.sram_ports(a.memories[-1]) == {"c": 1, "d": 0}
    m = a.memory_manifest("systemc")["vmem"]
    assert m["port_map"] == {"c": "1 (r)", "d": "0 (w)"}
    assert m["ports"]["c"]["macro_port"] == {"index": 1, "kind": "r"}
    assert "write" not in m["ports"]["c"] and "read" not in m["ports"]["d"]
    b = _small(s1r, kinds=("r", "w"))
    assert b.sram_ports(b.memories[-1]) == {"c": 1, "d": 0}  # w on the rw port
    # ... but Catapult would bind both to the rw port and tie the r port off
    with tempfile.TemporaryDirectory() as tmp:
        _refused(lambda: b.build("systemc", mode="csyn", project=os.path.join(tmp, "p")),
                 "mixes ReadWrite and r ports", "ties the others off")
    _refused(lambda: _small(s11.mapped(c=0, d=1), kinds=("r", "w")).plan("systemc"),
             "port c (r) cannot sit on port 0", "a 'w' macro port cannot read")
    _refused(lambda: _small(s1r.mapped(d=1), kinds=("r", "w")).plan("systemc"),
             "port d (w) cannot sit on port 1", "a 'r' macro port cannot write")
    _refused(lambda: _small(s11.mapped(x=0), kinds=("r", "w")).plan("systemc"), "names 'x'")
    _refused(lambda: s11.mapped(c=0, d=0), "two ports on one macro port")
    _refused(lambda: s11.mapped(c=2), "port 2", "ports 0..1")


def test_sram_cross_port_independence():
    """Under a collision obligation (or `undefined`) the two ports' accesses are
    independent and ``run.tcl`` says so after ``go architect`` -- what lets the
    two-port macro reach II=1 (s.6); same-port order is never released. A
    `refuse` memory with a reader and a different writer cannot sit on a macro
    that leaves a same-cycle read of a written word unknown."""
    import dataclasses  # noqa: PLC0415
    from allo.backend.catapult import memgen_spec, memory_independence  # noqa: PLC0415
    s2 = Sram.from_openram(V, LIB, catapult_lib="/x.lib")
    a = _small(s2)  # MiniTPU's two rw ports, collision="obligation"
    m = a.memories[-1]
    ops = a._server_ops(m)
    assert ops == {"c": {"write": "*:if:write_mem(mem:rsc*", "read": "*:else:*read_mem(mem:rsc*"},
                   "d": {"write": "*:if#1:write_mem(mem:rsc*", "read": "*:else#1:*read_mem(mem:rsc*"}}
    pairs = a.cross_port_independence(m)
    assert len(pairs) == 6  # c<->d: w-w, w-r, r-w each way; never r-r, never within a port
    assert ("*:if:write_mem(mem:rsc*", "*:if#1:write_mem(mem:rsc*") in pairs
    assert ("*:else:*read_mem(mem:rsc*", "*:else#1:*read_mem(mem:rsc*") not in pairs
    for x, y in pairs:  # one end on each port
        assert {x, y} & set(ops["c"].values()) and {x, y} & set(ops["d"].values())
    tcl = memory_independence(a.sram_configs("systemc"))
    assert tcl.startswith("go architect\n") and tcl.count("ignore_memory_precedences") == 6
    assert a.memory_manifest("systemc")["vmem"]["cross_port_independent"].startswith("Catapult may overlap")
    # the r/w probe: a plain read and a trailing write
    s11 = Sram.from_openram(V1R1W, LIB1R1W, catapult_lib="/x.lib")
    b = _small(s11, kinds=("r", "w"))
    assert b._server_ops(b.memories[-1]) == {"c": {"read": "*while:v*read_mem(mem:rsc*"},
                                              "d": {"write": "*:if:write_mem(mem:rsc*"}}
    assert len(b.cross_port_independence(b.memories[-1])) == 2
    # `undefined` too; `refuse` releases nothing (and refuses a cross r/w on this macro)
    u = _with_vmem(b, collision="undefined")
    assert len(u.cross_port_independence(u.memories[-1])) == 2
    _refused(lambda: _with_vmem(b, collision="refuse").plan("systemc"),
             "collision='refuse'", "old word", "UNKNOWN")
    ok = _with_vmem(b, collision="refuse", impl=dataclasses.replace(s11, rdwr="RBW"))
    assert ok.plan("systemc") == {"vmem": "sram"}
    assert ok.cross_port_independence(ok.memories[-1]) == []
    assert memory_independence(ok.sram_configs("systemc")) == ""
    _refused(lambda: dataclasses.replace(s11, rdwr="FIRST"), "rdwr='FIRST'")
    with tempfile.TemporaryDirectory() as tmp:
        spec, _ = memgen_spec(s11, tmp)
        t = open(spec, encoding="utf-8").read()
        assert "RDWRRESOLUTION   UNKNOWN" in t
        assert "NAME p0 MODE Write" in t and "NAME p1 MODE Read" in t
    # the emitted run.tcl: after `go assembly`, before `go extract`
    kernels = a.port_kernels("systemc")
    with tempfile.TemporaryDirectory() as tmp:
        prj = os.path.join(tmp, "p")
        cfg = {"clock_period": 3.33, "synth_group": {"name": "wa_d12g", "kernels": kernels}}
        a.build("systemc", mode="csyn", project=prj, configs=cfg)
        tcl = open(os.path.join(prj, "run.tcl"), encoding="utf-8").read()
        assert (tcl.index("go assembly") < tcl.index("go architect")
                < tcl.index("ignore_memory_precedences") < tcl.index("go extract"))
