# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-19: an optional module is a declared delta, and an instance's ISA
is the slots its modules bring.

The use case is MiniTPU's SFU as a ``compose.Option`` over a VPU lane built
from real units -- U1's ``bits`` ALU and track A's S1 ``bits`` SFU -- with
the writeback rebound ``alu_out -> sfu_out`` (``vpu.sv:88``, tag pipe
``:157-184``). Both instances are checked against the Phase 0 / U1 oracles
(``ref.vpu_alu``, ``ref.sfu``) on the simulator and in SystemC csim.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.compose import Architecture, Option, check_program, isa_slots
from examples.minitpu.template import vpu_lane as L

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

VARIANTS = {"base": (), "sfu": (L.SFU,)}


def _arch(variant, n):
    return Architecture.with_options(L.base(n), *VARIANTS[variant])


def test_isa_slots_follow_the_options():
    base, with_sfu = _arch("base", 8), _arch("sfu", 8)
    assert isa_slots(base) == {s: "base" for s in L.ALU_SLOTS}
    assert isa_slots(with_sfu) == {s: "base" for s in L.ALU_SLOTS} | {
        s: "sfu" for s in L.SFU_SLOTS}
    assert [u.name for u in with_sfu.units] == ["issue", "alu", "writeback", "sfu"]
    assert with_sfu.units[2].reads == ("sfu_out",)
    assert check_program(with_sfu, ["vadd", "vgelu"]) == ["vadd", "vgelu"]


def test_vgelu_refused_without_the_sfu():
    with pytest.raises(AssertionError, match="'vgelu' is not an instruction of this instance: "
                       "its module 'sfu' is not composed in"):
        check_program(_arch("base", 8), ["vadd", "vgelu"], known=(L.SFU,))
    with pytest.raises(AssertionError, match="no module of this instance brings it"):
        check_program(_arch("base", 8), ["vgelu"])


def test_malformed_options_refused():
    with pytest.raises(AssertionError, match="channel 'alu_out' reads in both writeback and sfu"):
        Architecture.with_options(L.base(8), L.SFU_NO_REBIND)
    with pytest.raises(AssertionError, match="nothing reads 'alu_out'"):
        Architecture.with_options(L.base(8), L.SFU_NO_UNIT)
    with pytest.raises(AssertionError, match="rebinds unit 'wb', which the architecture does not have"):
        Architecture.with_options(L.base(8), Option("x", rebind={"wb": {"alu_out": "y"}}))
    with pytest.raises(AssertionError, match="option 'sfu' composed twice"):
        Architecture.with_options(_arch("sfu", 8), L.SFU)
    with pytest.raises(AssertionError, match="brings ISA slot 'vadd', which base already brings"):
        Architecture.with_options(L.base(8), Option("dup", isa=("vadd",)))


@pytest.mark.parametrize("variant", ["base", "sfu"])
def test_lane_simulator(variant):
    arch = _arch(variant, 512)
    ops, a, b = L.program(isa_slots(arch), 512, seed=3)
    out = L.run(df.build(arch.region(), target="simulator"), ops, a, b)
    assert np.array_equal(out, L.reference(ops, a, b))


@needs_csim
@pytest.mark.parametrize("variant", ["base", "sfu"])
def test_lane_csim(variant):
    arch = _arch(variant, 128)
    ops, a, b = L.program(isa_slots(arch), 128, seed=4)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(arch.region(), target="systemc", mode="csim",
                       project=os.path.join(tmp, variant))
        out = L.run(mod, ops, a, b)
    assert np.array_equal(out, L.reference(ops, a, b))
