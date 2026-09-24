# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adder tree, described as Actions -- the Action layer's second machine.

The Action hypothesis concluded *not yet as the default, but yes for this
class*, and the stated reason not to promote it was that **only one machine
had exercised it**. `tests/test_actions.py` already builds a reduction machine,
but out of nothing: its units, its state and its lane widths are invented for
the example. This one is built from `ReduceParams` -- the same object that
parametrizes the unit that goes through csynth and Design Compiler -- so every
number here is a number the hardware also has.

What that buys, concretely, is the one field the Action layer could not
discharge on its own. `Action.at` is a declared latency, and its own docstring
says a declared span "has to be MEASURED against the hardware, by a probe that
scans downward so that it fails when the RTL turns out to be faster than
declared". `reduce_latency_probe.py` is that probe, and the `at` values below
are what it measured, not what the geometry suggests they ought to be. The
structural claim and the measurement DISAGREE, and the Action model is where
the disagreement becomes visible rather than a comment: `red_group` is one
adder level above `red_full` and arrives in the same cycle.
"""

import pytest

actions = pytest.importorskip(
    "allo.actions",
    reason="the Action layer is on branch `unit-actions`; this test is the "
           "second consumer it was asked for and lands green when that "
           "branch merges")

from allo.actions import (  # noqa: E402
    Action, Contract, EXACT, Instruction, Machine, Port, State, Unit, check)

from examples.accelerator.tinytpu_vitis.ip.reduce import ReduceParams  # noqa: E402


#: What `reduce_latency_probe.py` measured on the emitted Verilog at 8:2,
#: two-sided and value-checked. NOT derived from RED_DEPTH: the tree is three
#: adder levels deep and Vitis retires the whole of it in two cycles, so a
#: number derived from the geometry would be wrong in the direction that a
#: `>=` check accepts.
MEASURED = {"red_full": 2, "red_group": 2}


def tree_machine(params=None, arithmetic=EXACT, leaves=None):
    """The DotTree region as a Machine, built from `ReduceParams`.

    The leaf mapping is the unit's own, written over the SOURCE lane index:
    input lane `i` has `g = i % RED_GROUPS` and `s = i // RED_GROUPS`, and
    lands at leaf `g * RED_GROUP_SIZE + s`. It is deliberately not lane-major
    whenever RED_GROUPS < RED_LANES, so it reassociates the sum -- which under
    EXACT arithmetic is a permutation the model may accept and under ROUNDING
    is an obligation it must report.
    """
    p = params or ReduceParams()
    leaves = leaves or (f"(i % {p.RED_GROUPS}) * {p.RED_GROUP_SIZE} "
                        f"+ (i // {p.RED_GROUPS})")
    feed = Unit("dot_feed", ports=(Port("a.read"), Port("b.read"),
                                   Port("mul", physical=p.RED_LANES),
                                   Port("red_in")))
    tree = Unit("reduce_tree", ports=(Port("red_in"),
                                      Port("adder", physical=p.RED_LANES - 1),
                                      Port("red_full"), Port("red_group")))
    sink = Unit("dot_sink", ports=(Port("red_full"), Port("red_group"),
                                   Port("c.write"), Port("g.write")))
    return Machine(
        name="dot_tree",
        units=(feed, tree, sink),
        states=(State("A", rows="DOT_MAX * DOT_MAX", owner="dot_feed",
                      lanes="1"),
                State("B", rows="DOT_MAX * DOT_MAX", owner="dot_feed",
                      lanes="1"),
                State("C", rows="DOT_MAX * DOT_MAX", owner="dot_sink",
                      lanes="1"),
                State("G", rows="DOT_MAX * DOT_MAX * RED_GROUPS",
                      owner="dot_sink", lanes="1"),
                # THE ONE PLACE THE MODEL HAD TO BEND. A lane map is checked
                # against a lane COUNT, and the model reads that count off the
                # action's `state`. MiniTPU's fold reads a vector register
                # file, so its width is a property of addressed state; ours
                # arrives on a CHANNEL, and a channel is not a state. Without
                # something here the composition is refused with "lane map
                # without a width ... None declares no lane count", so the
                # packed word is declared as the state it momentarily is --
                # one row, RED_LANES lanes wide. It is the honest reading of a
                # FIFO word and it is still a bend: see
                # `test_the_lane_width_had_to_be_hung_on_a_declared_channel`.
                State("red_in", rows="1", owner="reduce_tree",
                      lanes="RED_LANES")),
        parameters={"DOT_MAX": p.DOT_MAX, "RED_LANES": p.RED_LANES,
                    "RED_GROUPS": p.RED_GROUPS,
                    "RED_GROUP_SIZE": p.RED_GROUP_SIZE,
                    "RED_DEPTH": p.RED_DEPTH},
        arithmetic=arithmetic,
        contracts=(Contract(
            "accumulator_is_wide_enough",
            "RED_ACC_BITS >= RED_IN_BITS + RED_DEPTH, so no level rounds and "
            "the leaf order is unobservable",
            discharged_by="reduction_tree_legality, at composition time"),),
        instructions=(
            Instruction("dot", rows="m * n", actions=(
                Action("dot_feed", "read", "a.read", state="A", base="a_s",
                       into="a", role="one row of A"),
                Action("dot_feed", "read", "b.read", state="B", base="b_s",
                       into="b", role="one column of B"),
                Action("dot_feed", "compute", "mul", compute="mul",
                       args=("a", "b"), into="prod",
                       role="RED_LANES lane products, packed"),
                Action("dot_feed", "emit", "red_in", args=("prod",),
                       into="word"),
                Action("reduce_tree", "receive", "red_in", into="word"),
                # ONE fold, TWO taps off it. The tree is not run twice: the
                # group sums are the level the root is built from, which is
                # why the tap is worth four wires and why MiniTPU's separate
                # lane-reduce network cost 19,198 LUT against the tree's own
                # 20,216.
                Action("reduce_tree", "compute", "adder",
                       compute="reduce_add", args=("word",), into="full",
                       state="red_in", lanes=leaves,
                       at=MEASURED["red_full"],
                       role="the root: every lane"),
                Action("reduce_tree", "compute", "adder",
                       compute="reduce_add_group", args=("word",),
                       into="groups", state="red_in", lanes=leaves,
                       at=MEASURED["red_group"],
                       role="the tap: RED_GROUPS subtree roots"),
                Action("reduce_tree", "emit", "red_full", args=("full",),
                       into="full_w", at=MEASURED["red_full"]),
                Action("reduce_tree", "emit", "red_group", args=("groups",),
                       into="group_w", at=MEASURED["red_group"]),
                Action("dot_sink", "receive", "red_full", into="full_w"),
                Action("dot_sink", "receive", "red_group", into="group_w"),
                Action("dot_sink", "write", "c.write", state="C",
                       base="c_d", args=("full_w",)),
                Action("dot_sink", "write", "g.write", state="G",
                       base="g_d", args=("group_w",)),
            )),
        ),
    )


ENV = {"m": 4, "n": 4, "a_s": 0, "b_s": 0, "c_d": 0, "g_d": 0}


def test_the_tree_is_expressible_as_actions_on_the_real_unit():
    """The headline: a reduce IS expressible as Actions, on the unit that
    synthesises, with its parameters rather than an example's."""
    m = tree_machine()
    assert set(m.units_of("dot")) == {"dot_feed", "reduce_tree", "dot_sink"}
    assert m.work("reduce_tree", "dot", ENV) == 16


def test_the_leaf_order_is_the_units_own_and_is_a_permutation():
    p = ReduceParams()
    m = tree_machine(p)
    fold = [e for e in m.effects("dot", ENV) if e.compute == "reduce_add"]
    # RED_GROUPS=2, RED_GROUP_SIZE=4: lanes 0,2,4,6 -> leaves 0..3.
    assert fold[0].lanes[:4] == (0, 4, 1, 5)
    assert sorted(fold[0].lanes) == list(range(p.RED_LANES))


def test_a_lane_order_that_loses_a_term_is_refused():
    """The model's one-sided rule, on our mapping rather than the example's."""
    with pytest.raises(actions.ActionError) as e:
        tree_machine(leaves="(i // 2) * 4 + (i // 4)")
    assert "lane map is not a permutation" in str(e.value)


def test_both_output_depths_are_in_the_interface():
    """Lesson 1 from MiniTPU, as a property of the composition: a unit with
    two output depths declares BOTH, and each is the measured number."""
    m = tree_machine()
    emits = {a.port: a.at for a in m.instruction("dot").actions
             if a.kind == "emit" and a.unit == "reduce_tree"}
    assert emits == {"red_full": MEASURED["red_full"],
                     "red_group": MEASURED["red_group"]}


def test_the_measured_tap_slack_contradicts_the_structural_one():
    """The result worth reporting. `red_group` is taken RED_DEPTH -
    RED_TAP_LEVEL adder levels above the root, so a latency derived from the
    structure would make the tap strictly earlier; the probe measures them
    arriving in the SAME cycle. The Action model carries the measured number,
    and this test is what stops the structural one from creeping back in."""
    p = ReduceParams()
    structural = p.RED_DEPTH - p.RED_TAP_LEVEL
    assert structural == 1
    measured = MEASURED["red_full"] - MEASURED["red_group"]
    assert measured == 0, (
        "if this ever becomes non-zero the tap has started to pay, and the "
        "gap row that says Allo cannot express two output LATENCIES needs "
        "re-reading before the number is believed")


def test_a_reassociating_fold_leaves_an_obligation_under_rounding():
    """Exactness is a property of the two ENDS here, not of the levels, and
    that is the whole architectural difference from MiniTPU's tree. Under
    EXACT the reassociation is discharged; under ROUNDING it is not, and the
    model says so rather than the docstring saying so."""
    exact = [o for o in check(tree_machine()) if "lane-major" in o.claim]
    rounding = [o for o in check(tree_machine(arithmetic=actions.ROUNDING))
                if "lane-major" in o.claim]
    assert len(exact) == 1 and exact[0].discharged_by
    assert len(rounding) == 1 and rounding[0].discharged_by is None


def test_the_lane_width_had_to_be_hung_on_a_declared_channel():
    """The friction, recorded so it is not mistaken for a clean fit.

    `_lanes_of` takes the fold's lane count from `self.state(action.state)`,
    so a lane map is only checkable against ADDRESSED STATE. A unit whose
    operand arrives on a channel has no such state, and the composition is
    refused until one is invented. Declaring the packed word as a one-row
    state is a fair reading of a FIFO word, but it is the model meeting a
    dataflow unit half way, and the repair the error suggests ("give the state
    a `lanes` expression") names a thing that does not exist for this unit.
    """
    p = ReduceParams()
    m = tree_machine(p)
    word = m.state("red_in")
    assert word is not None and word.rows == "1"
    # Its width is the tree's, not an example's.
    fold = [e for e in m.effects("dot", ENV) if e.compute == "reduce_add"]
    assert len(fold[0].lanes) == p.RED_LANES
