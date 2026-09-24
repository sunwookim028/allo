# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``allo.actions``: an instruction as a composition of per-unit effects.

Two machines, neither of them the one the model was written for. The first is
a vector unit with one accumulator read port, and it exists to show that a
work count is DERIVED: ``vadd`` costs two steps a row because it reads twice
and the port serves one read a step, not because anyone wrote 2 down. The
second is an adder-tree reduction with MiniTPU's leaf order, and it exists to
show what the model can and cannot settle about a unit whose operand mapping
reassociates the sum.
"""

import pytest

from allo.actions import (
    CHECKED, DESCRIPTION, EXACT, ORED, ROUNDING, UNDEFINED, Action,
    ActionError, Contract, Instruction, Machine, Port, State, Unit, check,
)


def vector_machine(**kwargs):
    accu = Unit("accu", ports=(Port("ar.read"), Port("ar.write"), Port("out")))
    alu = Unit("alu", ports=(Port("lane"),))
    return Machine(
        name="vector",
        units=(accu, alu),
        states=(State("ar", rows="NAR", owner="accu", lanes="T"),),
        parameters={"NAR": 64, "T": 4},
        instructions=(
            Instruction("vadd", rows="nr", actions=(
                Action("accu", "read", "ar.read", state="ar", base="ar_s1",
                       into="left", role="a source"),
                Action("accu", "read", "ar.read", state="ar", base="ar_s2",
                       into="right", role="a source"),
                Action("accu", "compute", "out", compute="add",
                       args=("left", "right"), into="sum"),
                Action("accu", "write", "ar.write", state="ar", base="ar_d",
                       args=("sum",)),
            )),
            Instruction("vrelu", rows="nr", actions=(
                Action("accu", "read", "ar.read", state="ar", base="ar_s",
                       into="v", role="a source"),
                Action("accu", "compute", "out", compute="max0", args=("v",),
                       into="r"),
                Action("accu", "write", "ar.write", state="ar", base="ar_d",
                       args=("r",)),
            )),
        ),
        **kwargs,
    )


def test_a_work_count_is_derived_from_the_port_budget():
    m = vector_machine()
    env = {"nr": 8, "ar_d": 0, "ar_s1": 8, "ar_s2": 16, "ar_s": 8}
    assert m.work("accu", "vadd", env) == 16
    assert m.work("accu", "vrelu", env) == 8


def _widen(m, read_ports):
    m.units = (Unit("accu", ports=(Port("ar.read", physical=2),
                                   Port("ar.write"), Port("out"))),
               m.unit("alu"))
    m.states = (State("ar", rows="NAR", owner="accu", lanes="T",
                      read_ports=read_ports),)
    return m


def test_widening_the_read_port_halves_the_work():
    """The one co-design question this model answers without a rebuild: what a
    second accumulator read port would be worth."""
    m = vector_machine()
    env = {"nr": 8, "ar_d": 0, "ar_s1": 8, "ar_s2": 16}
    before = m.work("accu", "vadd", env)
    _widen(m, read_ports=2).recheck(after="widen accu's ar.read port")
    assert m.work("accu", "vadd", env) == before // 2


def test_widening_the_unit_port_without_the_memory_buys_nothing():
    """A read port is cheap and a write port is not -- MiniTPU measured +592
    LUT for a third read and 4.3x the LUTs plus a timing miss for a second
    write. So the two counts are named separately, and the model says which
    one binds: widening the unit's port alone leaves the work where it was,
    because the memory still serves one read a cycle."""
    m = _widen(vector_machine(), read_ports=1).recheck(after="widen the port")
    assert m.work("accu", "vadd", {"nr": 8, "ar_d": 0, "ar_s1": 8,
                                   "ar_s2": 16}) == 16


def test_the_rows_an_instruction_reads_and_writes_are_a_query():
    m = vector_machine()
    env = {"nr": 2, "ar_d": 0, "ar_s1": 8, "ar_s2": 16}
    assert [(e.state, e.row) for e in m.reads("vadd", env)] == [
        ("ar", 8), ("ar", 16), ("ar", 9), ("ar", 17)]
    assert [(e.state, e.row) for e in m.writes("vadd", env)] == [
        ("ar", 0), ("ar", 1)]


def test_an_unknown_port_is_refused():
    m = vector_machine()
    bad = Instruction("vmul", rows="nr", actions=(
        Action("accu", "read", "ar.read", state="ar", base="ar_s1", into="a"),
        Action("accu", "compute", "multiplier", compute="mul", args=("a",),
               into="p"),
        Action("accu", "write", "ar.write", state="ar", base="ar_d",
               args=("p",)),
    ))
    with pytest.raises(ActionError) as e:
        m.with_instruction(bad)
    assert "unknown port" in str(e.value)
    assert "add 'vmul'" in str(e.value)


def test_a_value_that_crosses_units_without_a_channel_is_refused():
    m = vector_machine()
    bad = Instruction("vfoo", rows="nr", actions=(
        Action("alu", "compute", "lane", compute="add", args=(), into="x"),
        Action("accu", "write", "ar.write", state="ar", base="ar_d",
               args=("x",)),
    ))
    with pytest.raises(ActionError) as e:
        m.with_instruction(bad)
    assert "value crosses units" in str(e.value)


def test_a_value_nothing_produces_is_refused():
    m = vector_machine()
    bad = Instruction("vbar", rows="nr", actions=(
        Action("accu", "write", "ar.write", state="ar", base="ar_d",
               args=("nowhere",)),
    ))
    with pytest.raises(ActionError) as e:
        m.with_instruction(bad)
    assert "undefined value" in str(e.value)


# ------------------------------------------------- the second machine ---
def reduction_machine(arithmetic=EXACT, leaves="(i % 4) * 8 + (i // 4)"):
    """An adder-tree reduction unit, which TinyTPU cannot express: its PEs
    bake reduction into the unit graph through the `p_fwd` chain, so the fold
    order is the array's shape rather than a declaration.

    `leaves` is MiniTPU's mapping, `sublane * NUM_LANES + lane`, written over
    the source lane index for a 8-lane by 4-sublane vector. It is deliberately
    not lane-major, and it reassociates the sum."""
    tree = Unit("tree", ports=(Port("vr.read"), Port("leaf", physical=32),
                               Port("out")))
    return Machine(
        name="reduce",
        units=(tree,),
        states=(State("vr", rows="NVR", owner="tree", lanes="32"),
                State("sr", rows="NSR", owner="tree", lanes="1")),
        parameters={"NVR": 64, "NSR": 8},
        arithmetic=arithmetic,
        contracts=(Contract(
            "write_before_read",
            "every vr row a reduce folds was written earlier in the program",
            discharged_by="the assembler's program validator"),),
        instructions=(
            Instruction("vreduce", rows="nr", actions=(
                Action("tree", "read", "vr.read", state="vr", base="vr_s",
                       into="v", role="the vector to fold"),
                Action("tree", "compute", "leaf", compute="reduce_add",
                       args=("v",), into="s", state="vr", lanes=leaves),
                Action("tree", "write", "out", state="sr", base="sr_d",
                       args=("s",)),
            )),
        ),
    )


def test_the_reduction_tree_is_expressible_and_its_leaf_order_is_declared():
    m = reduction_machine()
    env = {"nr": 4, "vr_s": 0, "sr_d": 0}
    assert m.work("tree", "vreduce", env) == 4
    fold = [e for e in m.effects("vreduce", env) if e.compute == "reduce_add"]
    assert fold[0].lanes[:5] == (0, 8, 16, 24, 1)


def test_a_reassociating_fold_is_accepted_and_leaves_an_obligation():
    exact = check(reduction_machine(EXACT))
    rounding = check(reduction_machine(ROUNDING))
    reassoc = [o for o in exact if "lane-major" in o.claim]
    assert len(reassoc) == 1 and reassoc[0].discharged_by
    reassoc = [o for o in rounding if "lane-major" in o.claim]
    assert len(reassoc) == 1 and reassoc[0].discharged_by is None
    assert any("write_before_read" in (o.premise or "") for o in rounding)


def test_a_lane_order_that_loses_a_term_is_refused():
    with pytest.raises(ActionError) as e:
        reduction_machine(leaves="(i // 4) * 4 + (i // 8)")
    assert "lane map is not a permutation" in str(e.value)


def test_a_lane_major_fold_leaves_no_reassociation_obligation():
    m = reduction_machine(ROUNDING, leaves="i")
    assert not [o for o in check(m) if "lane-major" in o.claim]


# ------------------------------- cases this model was NOT designed for ---
#
# An abstraction is graded by applying it to something its author did not
# choose. These three come from elsewhere in the project, on the same day:
# a proposed `s.memory_ports(target, write_ports)` rule that refused a design
# when `stores > write_ports * banks`, and MiniTPU's VREG file.
def two_store_machine(bank, banks=2, ii=1, collision="defined"):
    """A loop with two stores, a partitioned memory, and an initiation
    interval. `s.memory_ports` decided this shape wrongly in BOTH directions:
    it accepted Partition.Block factor 2, where both stores land in bank 0,
    and it refused an unpipelined pair, which never contends."""
    return Machine(
        name="two-store",
        units=(Unit("st", ports=(Port("mem.write", physical=2), Port("alu", physical=2)),
                    ii=ii, elastic=False),),
        states=(State("buf", rows="N", owner="st", banks=str(banks),
                      bank=bank, write_ports=1, collision=collision),),
        parameters={"N": 64},
        instructions=(Instruction("store2", rows="1", actions=(
            Action("st", "compute", "alu", compute="const", into="a"),
            Action("st", "compute", "alu", compute="const", into="b"),
            Action("st", "write", "mem.write", state="buf", base="2 * d",
                   args=("a",)),
            Action("st", "write", "mem.write", state="buf", base="2 * d + 1",
                   args=("b",)),
        )),),
    )


def test_cyclic_partitioning_separates_the_two_stores():
    """Both stores issue in one step, and `r % 2` sends them to banks 0 and 1,
    so one write port per bank is enough. Accepted -- as the scalar rule also
    accepted it, for the wrong reason."""
    m = two_store_machine(bank="r % 2")
    assert m.work("st", "store2", {"d": 0}) == 1
    assert {e.cycle for e in m.effects("store2", {"d": 0})} == {0}


def test_block_partitioning_does_not_and_is_refused():
    """The false ACCEPT of the scalar rule: `banks *= factor` for Block and
    Cyclic alike, so `2 stores <= 1 port x 2 banks` passes -- while the block
    map sends rows 0 and 1 to the same bank and the RTL emits two write
    statements per bank instance. Composing the map decides it correctly."""
    with pytest.raises(ActionError) as e:
        two_store_machine(bank="r // (N // 2)")
    assert "does not fit the initiation interval" in str(e.value)
    assert "bank map" in str(e.value)


def test_an_unpipelined_pair_is_accepted_because_the_ii_is_consulted():
    """The false REFUSE of the scalar rule: it never consulted `pipeline_ii`,
    so it refused a sequential pair that needs one port. Two writes a step
    through one port are legal at II 2, and the same pair at II 1 is not."""
    assert two_store_machine(bank="0", ii=2).work("st", "store2", {"d": 0}) == 1
    with pytest.raises(ActionError):
        two_store_machine(bank="0", ii=1)


def test_an_undefined_same_element_collision_is_refused():
    """MiniTPU's `vpu_word_array` is true dual-port and leaves a same-word
    collision undefined, as on the FPGA -- "a contract our software has never
    had to respect and yours would have to". A model that can only say "this
    unit writes the register file" cannot state it."""
    with pytest.raises(ActionError) as e:
        Machine(
            name="vreg",
            units=(Unit("vpu", ports=(Port("vr.write", physical=2),)),),
            states=(State("vreg", rows="NV", owner="vpu", write_ports=2,
                          collision=UNDEFINED),),
            parameters={"NV": 32},
            instructions=(Instruction("vmerge", rows="1", actions=(
                Action("vpu", "compute", "vr.write", compute="const", into="a"),
                Action("vpu", "compute", "vr.write", compute="const", into="b"),
                Action("vpu", "write", "vr.write", state="vreg", base="d",
                       args=("a",)),
                Action("vpu", "write", "vr.write", state="vreg", base="d",
                       args=("b",)),
            )),),
        )
    assert "undefined write collision" in str(e.value)


def test_an_or_ed_write_port_is_accepted_and_names_its_enforcer():
    """Their write-port calendar lives inside `ifndef SYNTHESIS`: on the board
    two units writing one VREG port in a cycle are OR-ed, one-hot, with no
    arbiter, and the assembler is the only enforcement. That is a CHECKED
    claim with a named checker, not a hardware guarantee, and the model has to
    be able to tell those apart."""
    m = Machine(
        name="vreg-or",
        units=(Unit("vpu", ports=(Port("vr.write"),)),),
        states=(State("vreg", rows="NV", owner="vpu", collision=ORED),),
        parameters={"NV": 32},
        instructions=(Instruction("vmov", rows="1", actions=(
            Action("vpu", "compute", "vr.write", compute="const", into="a"),
            Action("vpu", "write", "vr.write", state="vreg", base="d",
                   args=("a",), status=CHECKED,
                   enforced_by="the assembler's write-port calendar"),
        )),),
    )
    held = [o for o in check(m) if "OR-ed" in o.claim]
    assert len(held) == 1
    assert held[0].discharged_by == "the assembler's write-port calendar"


def test_a_checked_claim_must_name_its_checker():
    with pytest.raises(ActionError) as e:
        Machine(
            name="unchecked",
            units=(Unit("u", ports=(Port("w"),)),),
            states=(State("m", rows="N", owner="u"),),
            parameters={"N": 8},
            instructions=(Instruction("i", rows="1", actions=(
                Action("u", "compute", "w", compute="const", into="a"),
                Action("u", "write", "w", state="m", base="d", args=("a",),
                       status=CHECKED),
            )),),
        )
    assert "checked claim with no checker" in str(e.value)


def test_an_undecorated_action_is_only_a_description():
    m = vector_machine()
    assert all(a.status == DESCRIPTION
               for i in m.instructions for a in i.actions)
    assert [o for o in check(m) if "DESCRIPTION" in o.claim]


# ------------------------------------------------------ the falsifier ---
def shared_port_machine(retire_at=0):
    """Two units that each write one accumulator, and a memory that has one
    write port. Neither unit can reschedule around the other, so the port is
    where a composition can be illegal although each of its actions is not."""
    return Machine(
        name="shared",
        units=(Unit("accu", ports=(Port("alu"), Port("ar.write"))),
               Unit("retire", ports=(Port("ret"), Port("ar.write")))),
        states=(State("ar", rows="NAR", owner="accu", write_ports=1),),
        parameters={"NAR": 64},
        instructions=(
            Instruction("acc_only", rows="nr", actions=(
                Action("accu", "compute", "alu", compute="add", into="a"),
                Action("accu", "write", "ar.write", state="ar", base="ar_d",
                       args=("a",)),
            )),
            Instruction("ret_only", rows="nr", actions=(
                Action("retire", "compute", "ret", compute="copy", into="b"),
                Action("retire", "write", "ar.write", state="ar", base="ar_d",
                       args=("b",)),
            )),
        ),
    )


def both_at(retire_at):
    return Instruction("acc_and_ret", rows="nr", actions=(
        Action("accu", "compute", "alu", compute="add", into="a"),
        Action("accu", "write", "ar.write", state="ar", base="ar_d",
               args=("a",)),
        Action("retire", "compute", "ret", compute="copy", into="b"),
        Action("retire", "write", "ar.write", state="ar", base="ar_e",
               args=("b",), at=retire_at),
    ))


def test_two_legal_actions_compose_into_an_illegal_instruction():
    """THE FALSIFIER. Each of these actions is legal in an instruction of its
    own -- the machine holding both `acc_only` and `ret_only` is accepted. Put
    them in ONE instruction and two units write the same accumulator port in
    the same cycle, which no unit can reschedule around. A rule that could
    only refuse what a count refuses would miss this: the count is one write
    per unit, and one write port per unit is exactly what the machine has."""
    m = shared_port_machine()
    assert m.work("accu", "acc_only", {"nr": 1, "ar_d": 0}) == 1
    with pytest.raises(ActionError) as e:
        m.with_instruction(both_at(0))
    assert "write port shared by units" in str(e.value)
    assert "acc_and_ret" in str(e.value)


def test_declaring_a_latency_moves_the_entry_and_the_composition_is_legal():
    """And the repair is a calendar entry, not a port: declaring that the
    retire path lands one cycle later books a different cycle. Whether that
    latency is TRUE of the hardware is not something this model can settle --
    it has to be measured, by a probe that scans downward, so that it fails
    when the RTL turns out to be faster than declared and not only slower."""
    m = shared_port_machine().with_instruction(both_at(1))
    booked = m.calendar("acc_and_ret", {"nr": 1, "ar_d": 0, "ar_e": 4})
    cycles = sorted(c for (res, c), hits in booked.items()
                    if res[0] == "state" for _ in hits)
    assert cycles == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
