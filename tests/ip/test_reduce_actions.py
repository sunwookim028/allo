# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adder tree, described as Actions -- the Action layer's second machine.

The Action hypothesis concluded *not yet as the default, but yes for this
class*, and the stated reason not to promote it was that **only one machine
had exercised it**. This one is built from `ReduceParams` -- the same object
that parametrizes the unit that goes through csynth and Design Compiler -- so
every number here is a number the hardware also has, and both declared
latencies are what `reduce_latency_probe.py` MEASURED rather than what the
geometry suggests.

The machine is no longer written here. It is
`examples.tinytpu.ip.reduce.machine()`, and its units, ports, states and the
lane count of the packed word are `structure(architecture())` -- read off the
composed region, not declared a second time. What that file adds is the
arithmetic, the two measured latencies, the leaf order and the contract.

Three things this file used to record as the Action layer's limits, and what
happened to each:

* **the work count was wrong by 5x** -- `Machine.work` charged a declared
  latency as occupancy and returned 80 steps where the hardware does 16.
  REPAIRED, by distinguishing a row's LATENCY from the rate the unit
  sustains, which is derived from the ports and not declared:
  `test_the_latency_and_the_rate_are_now_two_numbers`.
* **the lane width hung on addressed state** -- the packed word had to be
  declared a one-row `State` because a lane map was only checkable against a
  memory. REPAIRED, by giving the channel the lane count that the width is
  derived from: `test_the_lane_width_is_the_channels_own`.
* **one fold read at two taps cannot be said** -- STILL TRUE, and reported as
  two obligations for one reassociation:
  `test_a_reassociating_fold_leaves_an_obligation_under_rounding`.
"""

import pytest

import allo.actions as actions  # noqa: E402

from allo.actions import EXACT, check  # noqa: E402

from examples.tinytpu.ip.reduce import (  # noqa: E402
    MEASURED, ReduceParams, machine)


def tree_machine(params=None, arithmetic=EXACT, leaves=None):
    """The DotTree Machine, as `ip/reduce.py` composes it."""
    return machine(params=params, arithmetic=arithmetic, leaves=leaves)


ENV = {"m": 4, "n": 4, "a_s": 0, "b_s": 0, "c_d": 0, "g_d": 0}


def test_the_tree_is_expressible_as_actions_on_the_real_unit():
    """The headline, and it is a qualified yes: a reduce IS expressible as
    Actions, on the unit that synthesises, with its parameters rather than an
    example's. Every structural claim the unit makes survives the trip --
    which units the opcode reaches, the leaf order, the two output depths,
    the reassociation obligation. What does NOT survive is the unit's work
    count; see `test_the_declared_latency_is_charged_as_occupancy`."""
    m = tree_machine()
    assert set(m.units_of("dot")) == {"dot_feed", "reduce_tree", "dot_sink"}
    # The feeder is the unit whose Actions carry no declared latency, and it
    # is the one whose work count comes out right: one step per row.
    assert m.work("dot_feed", "dot", ENV) == 16
    assert m.work("dot_sink", "dot", ENV) == 16


def test_the_latency_and_the_rate_are_now_two_numbers():
    """THE REPAIRED NEGATIVE, and how it was repaired matters more than that
    it was.

    `Machine.work` documents itself as "the number a per-unit work count in an
    instruction-memory header carries". For `reduce_tree` it used to be 80
    where the hardware does 16, because `_book` took a row's cost to be the
    SPAN of its placement, so a measured `at=2` was paid once per row instead
    of once per stream -- the model drew no distinction between a latency and
    an initiation interval.

    It draws one now, and NOTHING NEW IS DECLARED to buy it. The calendar is
    unchanged -- the fold still lands two cycles after the word arrives, and
    both emits two after that -- but the cycles between one row and the next
    are the BUSIEST RESOURCE the row books, which is read off the ports the
    unit already declares. `Unit.ii` is untouched: buying the number with
    `ii=5` would have been declaring something false.

    Both numbers survive and they are different questions: 16 steps of work,
    20 cycles end to end for 16 rows whose last one finishes 5 cycles after
    it starts."""
    m = tree_machine()
    cost = m.profile("dot", ENV)["reduce_tree"]
    # receive at 0, both folds at 2 (`at`), both emits at 4: unchanged.
    assert [e.cycle for e in cost.row] == [0, 2, 2, 4, 4]
    assert cost.row_latency == 5 and cost.row_steps == 1
    assert m.work("reduce_tree", "dot", ENV) == 16 == m.work("dot_feed",
                                                             "dot", ENV)
    assert cost.latency == 20
    # And the rate is the ports': one word a cycle, because `red_in`,
    # `red_full` and `red_group` are one item a cycle each and the adder is
    # RED_LANES - 1 wide. Halve the tree's input port and the rate halves
    # with it, with no other edit.
    assert m.unit("reduce_tree").ii == 1


def test_the_lane_width_is_the_channels_own():
    """THE OTHER REPAIRED NEGATIVE. A lane map used to be checkable only
    against ADDRESSED STATE, so the packed word was declared a one-row
    `State` -- a memory nothing addresses -- to get the composition accepted.

    What a fold actually needs is a WIDTH, and `compose.Channel` can carry
    one: `red_in` declares `lanes="RED_LANES"` and `lane_bits="RED_IN_BITS"`,
    and the bit width of the word is DERIVED from the pair rather than
    declared beside them. The Action layer reads the count off the channel
    the operand arrived on, which it finds by following the value flow it
    already checks. No shadow state, and no second place the lane count can
    be written."""
    p = ReduceParams()
    m = tree_machine(p)
    assert m.state("red_in") is None, "the shadow state should be gone"
    assert m.channel("red_in").lanes == "RED_LANES"
    fold = [e for e in m.effects("dot", ENV) if e.compute == "reduce_add"]
    assert len(fold[0].lanes) == p.RED_LANES
    # What a channel does NOT have is rows, a bank map or a collision rule --
    # and a fold needs none of them. That is the finding: what the model was
    # missing was a lane count, not addressed state.
    assert m.channel("red_in").lane_bits == "RED_IN_BITS"


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
    # TWO obligations, not one, and the count is the same missing distinction
    # as the latency: the model has no way to say "one fold, two taps off it",
    # so the single physical fold is written as two `compute` Actions and its
    # one reassociation is reported once per Action. The direction is right
    # (discharged under EXACT, open under ROUNDING) and the arithmetic is the
    # tree's own; the multiplicity is the model's.
    assert len(exact) == 2 and all(o.discharged_by for o in exact)
    assert len(rounding) == 2 and all(o.discharged_by is None
                                      for o in rounding)
    assert {o.claim.split()[0] for o in exact} == {"reduce_add",
                                                   "reduce_add_group"}


def test_the_structure_is_the_composed_regions_and_not_a_second_copy():
    """The positive half of the claim, on the architecture where the two
    models' grain agrees. Every unit, every channel-carrying port, every memory port and
    every state of this machine is `structure(architecture())`; the only
    ports declared in `ip/reduce.py` are the two COMPUTE ports, which are
    arithmetic and which no structural declaration carries."""
    from examples.tinytpu.ip.reduce import architecture  # noqa: PLC0415
    from allo.actions import projection  # noqa: PLC0415
    report = projection(architecture(), tree_machine())
    extra = {u: r["only_declared"] for u, r in report.items()
             if r["only_declared"]}
    assert extra == {"dot_feed": ["mul"], "reduce_tree": ["adder"]}
    assert all(r["grain"] is None for r in report.values())
    # The unit's own local array is state the composition knows about and the
    # hand-written machine never had: `node`, the tree itself.
    assert tree_machine().state("node").rows == "2 * RED_LANES - 1"
