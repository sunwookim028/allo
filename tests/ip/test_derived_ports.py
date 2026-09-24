# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Can an Action unit's ports be DERIVED from a composed region's units?

`allo/compose.py` and `allo/actions.py` were two descriptions of one machine
with no type in common, which made the Action layer a second declaration of
every unit -- the failure class it exists to remove. This file is the
measurement of how much of that second declaration a composed `Architecture`
already implies, and, where it implies nothing, of what the missing fact is.

The answer is a SPLIT rather than a yes or a no, and the split is the result:

* a port that carries a CHANNEL or addresses a MEMORY is the composition's,
  and derives. On TinyTPU 20 of the spec's 28 ports are exactly that and 4
  more stand for such ports under a name of the model's own;
* a port that carries ARITHMETIC has no structural counterpart at all and
  never will: a multiplier is not a channel and not a memory. There are 3,
  and one more port that carries nothing physical and exists to be counted;
* a port's CAPACITY is a hardware fact no declaration in either model can
  derive (`alu` is 2 wide because the unit chains two lane operations);
* and the two models disagree about WHAT A UNIT IS wherever a kernel is
  replicated. That disagreement is not a naming difference and is pinned
  here as its own test.

`gen_isa.py --check` runs this comparison as a gate, so the two models cannot
drift apart where they overlap. What they cannot express about each other is
in `UNMODELLED`, `ALIAS`, `AGGREGATE`, `INVENTED` and `UNSPENT` there, each
entry with its reason, and this file pins the shape of each table.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

from dataclasses import replace  # noqa: E402

from allo.actions import (  # noqa: E402
    Action, Instruction, PER_INSTRUCTION, Port, projection, structure)
from examples.tinytpu import gen_isa  # noqa: E402
from examples.tinytpu.isa_encoding import machine  # noqa: E402
from examples.tinytpu.ip import tinytpu as T  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return projection(T.architecture(), machine(), aggregate=gen_isa.AGGREGATE)


def test_a_channel_port_and_a_memory_port_derive(report):
    """The positive half. `spm` and `vru` are the cleanest case: every port
    the Action model gives them is a channel they are declared to touch or a
    memory their body addresses, and nothing about them is stated twice."""
    assert report["spm"]["only_declared"] == []
    assert report["vru"]["only_declared"] == []
    assert set(report["spm"]["derived"]) == {
        "spad.read", "spad.write", "dma2sp", "sp2vr", "wcol"}
    derived = sum(len(r["derived"]) for r in report.values())
    assert derived == 20


def test_arithmetic_is_the_part_that_cannot_derive(report):
    """The negative half, and it is the layer's reason to exist. Three ports
    carry arithmetic and one carries nothing: `dma_ld`'s operand select,
    the array's multiply-accumulate, the accumulator's ALU, and the counting
    port the header's `mm_count` is items of. No composition implies any of
    them, because a composed region declares what a unit is WIRED to and
    never what it computes."""
    undeclared = {u: r["only_declared"] for u, r in report.items()
                  if r["only_declared"]}
    spent = {}
    for i in machine().instructions:
        for a in i.actions:
            spent.setdefault((a.unit, a.port or a.kind), set()).add(a.kind)
    compute = {(u, p) for u, ports in undeclared.items() for p in ports
               if spent.get((u, p)) == {"compute"} and
               (u, p) not in gen_isa.ALIAS}
    assert compute == {("dma_ld", "mux"), ("array", "mac"), ("accu", "alu")}
    # `sequencer.fetch` is both: it stands for the composed `imem.read` port
    # AND the loop-frame computes are spent on it. An alias may carry
    # arithmetic; what it may not do is name a memory the region has not got.
    assert gen_isa.ALIAS[("sequencer", "fetch")] == ("imem.read",)
    assert set(gen_isa.INVENTED) == {("array", "instructions")}
    # And a capacity is a hardware fact even where the port derives: the ALU
    # is two lane operations a step, which is what `vaddrelu` rests on.
    assert machine().unit("accu").port("alu").physical == 2


def test_the_two_models_disagree_about_what_a_unit_is(report):
    """THE REASON `structure()` CANNOT SIMPLY REPLACE THE UNIT TABLE.

    A compose unit is a KERNEL, replicated by `instances` and wired to its
    copies by stream arrays; an Action unit is a DISPATCH DOMAIN -- one work
    counter, fed one row count by the sequencer. They are the same object for
    every unit instantiated once, and a different object for every unit that
    is not. TinyTPU has exactly one such region: `wld` and `pe`, T x T of
    each, wired by five chains, which the ISA sees as one `array`.

    The consequence is not cosmetic. `items(array, 'mac')` is the header's
    `mm_rows`; at the composed grain the same quantity is a per-instance
    count over T*T kernels, and the model has no word for an instance."""
    grains = {u: r["grain"] for u, r in report.items() if r["grain"]}
    assert set(grains) == {"array"}
    assert "wld + pe" in grains["array"]
    composed = {u.name: u.instances for u in T.architecture().units}
    assert composed["pe"] == ("T", "T") and composed["wld"] == ("T", "T")
    assert all(i == ("1",) for n, i in composed.items()
               if n not in ("pe", "wld"))


def test_the_composition_sees_memories_the_isa_cannot_name():
    """Three on-chip memories the Action model is silent about, found by the
    derivation rather than by reading the RTL: the sequencer's prefetch
    window and the two burst landing buffers. They are real state with real
    ports, and no instruction field names a row of any of them, so an action
    over them would have no base to resolve. The model's silence is honest;
    that it was never visible is what the derivation fixes."""
    implied = structure(T.architecture())
    assert {s.name for s in implied.states} - {
        s.name for s in machine().states} == {"program", "a_onchip",
                                              "b_onchip"}
    assert implied.state("a_onchip").rows == "MAXDIM * WPR + DMA_WORDS"


def test_the_control_path_cannot_be_modelled_as_an_action():
    """WHY the five dispatch channels are in `UNMODELLED` rather than being
    written as actions -- the model cannot say what the hardware does.

    Every data unit reads its queue when its row counter runs out, INSIDE the
    step it is already spending. An action is per-row or per-instruction. A
    per-row receive would say every row fetches; a per-instruction one adds a
    head step to the unit's work, and the work count is what the
    instruction-memory header carries, so the sequencer would promise `accu`
    one iteration more than it performs. This test does it and measures the
    damage rather than asserting it."""
    # The port itself is the composition's -- `accu` reads `c_acc` and
    # `ip/tinytpu.py` says so -- so the derivation supplies it and only the
    # action has to be written.
    base = machine()
    m = replace(base, units=tuple(
        u if u.name != "accu" else replace(u, ports=u.ports + (Port("c_acc"),))
        for u in base.units))
    vadd = m.instruction("vadd")
    probe = {"ar_d": 0, "ar_s1": 8, "ar_s2": 16, "nr": 3}
    assert m.work("accu", "vadd", probe) == 6
    fetched = m.with_instruction(Instruction(
        "vadd_with_fetch", rows=vadd.rows,
        actions=(Action("accu", "receive", "c_acc", into="control",
                        per=PER_INSTRUCTION),) + vadd.actions))
    assert fetched.work("accu", "vadd_with_fetch", probe) == 7
