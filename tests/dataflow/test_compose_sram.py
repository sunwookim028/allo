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
