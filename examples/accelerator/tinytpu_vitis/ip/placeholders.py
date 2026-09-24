# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOT IMPLEMENTED. Declarations for capabilities this library does not have.

**Nothing here builds, and nothing here is instantiated by any architecture.**
Every name in this module raises ``NotImplemented`` when it is used. A
placeholder that lies about being implemented is worse than none, so these are
declarations and legality conditions only -- an interface written down, with
the one thing that would make it real stated beside it.

The gap each one stands for, the evidence that we cannot do it today and the
ranking are in ``docs/source/designs/ip_gaps.rst``. ``tests/ip/test_ip_gaps.py``
runs one failing or skipping test per row, so an unfilled gap is visible from a
test run and not only from prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class NotBuilt(NotImplementedError):
    """Raised by every placeholder in this module.

    Carries what the thing would be and what would make it real, because a
    bare ``NotImplementedError`` in a traceback two years from now says
    neither."""

    def __init__(self, gap):
        super().__init__(
            f"{gap.name}: NOT IMPLEMENTED. {gap.what}\n"
            f"  needs: {gap.needs}\n"
            f"  evidence that we cannot do it today: {gap.evidence}\n"
            f"  real when: {gap.real_when}")


@dataclass(frozen=True)
class Gap:
    """One missing capability, declared rather than implemented."""

    name: str
    what: str
    target: str
    needs: str
    evidence: str
    real_when: str
    kind: str = "cannot express"     # or "cannot refuse"
    interface: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    legality: tuple = ()

    def __call__(self, *a, **k):
        raise NotBuilt(self)

    def unit(self, *a, **k):
        raise NotBuilt(self)


SECOND_INSTANCE = Gap(
    name="second_instance",
    what=("one unit instantiated N times in one region, against different "
          "channels AND at different parameter sets -- N slices, each with "
          "its own local memory, its own sizes and its own wiring"),
    target="Jalapeno (per-slice local memory, no global L2); Groq LPU",
    needs=("compose.py to emit `@df.unit` instantiations instead of nesting "
           "kernel source, AND `@df.unit` to accept parameters at the "
           "instantiation site. Each half exists without the other: "
           "`@df.unit` binds ports positionally but takes its sizes from its "
           "own module, and compose.py binds any parameter set but reaches "
           "channels by lexical name, so two instances collide on the name"),
    evidence=("in THIS library, Architecture(units=(copy_unit, copy_unit)) "
              "fails Architecture._check with \"channel 'b' writes in both "
              "copy_unit and copy_unit\", because the composed source would "
              "define the same kernel twice "
              "(tests/ip/test_ip_gaps.py::test_second_instance_of_a_unit). "
              "The front end no longer has that limit -- "
              "tests/dataflow/test_stream_ports.py::"
              "test_a_unit_is_instantiated_twice_against_different_streams "
              "passes -- so the remaining half is adoption plus "
              "instantiation-site parameters, which "
              "docs/source/developer/stream_ports.rst names as not attempted"),
    real_when=("compose.py emits `@df.unit` instances and `@df.unit` joins "
               "tests/dataflow/test_hierachical.py's `inner[P0, P1]` type "
               "parameters. This is the highest-value structural change "
               "available to this library: it closes this row, and PLACEMENT "
               "and COLLECTIVE both need it first"),
    interface={"instances": "an instance NAME, not only a mapping shape",
               "reads": "bound positionally by the architecture",
               "writes": "bound positionally by the architecture",
               "parameters": "per instance, not per architecture"},
)

PLACEMENT = Gap(
    name="placement",
    what=("which physical slice a unit instance and the memory it addresses "
          "land on, decided at compile time and carried in the design rather "
          "than in a comment"),
    target=("Jalapeno: TensorInfo encodes logical layout AND physical "
            "placement, and the compiler decides which HBM slice every weight "
            "shard, KV and expert belongs to"),
    needs=("a placement concept in compose.py: Unit grows a placement "
           "expression over the architecture's parameters, Memory grows an "
           "owning slice, and Architecture._check holds a channel to units "
           "that are placed where the channel can reach"),
    evidence=("compose.py contains none of the words place, slice, location, "
              "affinity, topology or bank. Unit's fields are body, instances, "
              "memories, reads, writes, parameters, isa, directives, "
              "legality. tests/ip/test_ip_gaps.py::test_no_placement_concept"),
    real_when=("a second architecture needs it -- placement with one slice is "
               "a field nothing reads, and this library has never composed a "
               "machine with more than one"),
    parameters={"SLICES": "how many slices the machine has",
                "placement": "an expression over the instance's pid giving "
                             "its slice"},
)

COLLECTIVE = Gap(
    name="collective",
    what=("a channel whose topology is declared -- broadcast, all-reduce, "
          "all-gather -- rather than hand-built as a chain of "
          "point-to-point streams"),
    target=("Jalapeno: an independent 8x8 two-stage collective network "
            "alongside a general mesh, so a collective is not a route through "
            "the compute fabric; Groq LPU"),
    needs=("Channel grows a topology kind, and compose.py emits the chain or "
           "the tree that implements it. Allo's streams are one-reader "
           "one-writer, so every collective is SOME arrangement of them; what "
           "is missing is saying which one you meant"),
    evidence=("ip/tinytpu.py's own docstring: \"every distribution is a CHAIN "
              "carrying packed words, never a T-way fan-out\" -- the topology "
              "is the sixteen Channel declarations, and changing it means "
              "rewriting the unit bodies that index them. "
              "tests/ip/test_ip_gaps.py::test_no_collective_topology"),
    real_when=("reduce_tree generalises: a tree over UNITS is the same shape "
               "as the tree inside reduce_tree, one level per hop instead of "
               "one level per adder, so the first real collective should be "
               "built by lifting that rather than by a new mechanism"),
    parameters={"kind": "point_to_point | chain | broadcast | all_reduce",
                "arity": "fan-in or fan-out per stage"},
)

UNIT_LATENCY = Gap(
    name="unit_latency",
    what=("a unit's latency declared as a fact and checked from both ends: "
          "the declaration against the scheduled RTL, and the declaration "
          "against whatever books it"),
    target="both, and MiniTPU's own strongest recommendation to us",
    needs=("a latency contract beside ip/isa.py, generated-from and checked-"
           "against like the encoding is; and a two-sided probe -- one that "
           "scans DOWNWARD for the smallest offset from which every later "
           "offset also matches, so it fails when the RTL is FASTER than "
           "declared and not only when it is slower"),
    evidence=("Unit has no latency field, and reduce_tree's two outputs sit "
              "RED_DEPTH - RED_TAP_LEVEL adder levels apart with nothing "
              "stating what that is in cycles. MiniTPU shipped a measured "
              "lane-tap latency that NOTHING USED for a whole commit -- the "
              "probe was right, the scheduler booked every reduce at the root "
              "latency, and wiring the two together took a span-64 softmax "
              "from 150 bundles to 142. "
              "tests/ip/test_ip_gaps.py::test_unit_cannot_declare_latency"),
    real_when=("the `unit-actions` branch lands its per-unit effects: a "
               "latency is a property of an action, not of a unit, and "
               "building it twice is the failure this is here to avoid. "
               "FLAGGED TO THAT BRANCH, NOT BUILT HERE"),
    kind="cannot refuse",
    parameters={"occupancy": "cycles the unit is busy", "result": "cycles to "
                "first result -- NOT the same number, and MiniTPU's "
                "vmatpush is 5 and 82"},
)

DECLARED_ARITHMETIC = Gap(
    name="declared_arithmetic",
    what=("a unit declaring the types it computes in, so that composing it "
          "into an architecture whose arithmetic it does not implement is a "
          "composition error"),
    target="all three: MXFP4/FP8 on Jalapeno, BF16 on MiniTPU, int8 here",
    needs=("int8, int16, int32 leave FRONTEND_NAMES and become declarable "
           "parameters like any other, so Unit.check sees them"),
    evidence=("FRONTEND_NAMES exempts int8/int16/int32 from declaration, so "
              "pe declares parameters=('T','VW','AW') and says nothing about "
              "computing an int8 x int8 product into int32. A unit's "
              "interface does not state its arithmetic. reduce_tree declares "
              "RED_IN and RED_ACC instead and is the shape the other eight "
              "would take. "
              "tests/ip/test_ip_gaps.py::test_units_do_not_declare_arithmetic"),
    real_when=("retrofitting the eight shipped units, which is a change to "
               "every declaration in the library for no capability today -- "
               "deliberately not done, see the ranking"),
    kind="cannot refuse",
)

ASIC_LEGAL_MEMORY = Gap(
    name="asic_legal_memory",
    what=("refusing, at composition time, a unit whose local array has more "
           "than one write site -- the structure an FPGA gives away and "
           "standard cells refuse"),
    target="the ASIC flow itself, which is constraint 1 on any unit here",
    needs=("a write-site count per local array in Unit.check, or a partition "
           "directive required of any array with more than one; plus the RTL "
           "audit, which today exists only as the memory of one refusal"),
    evidence=("a unit whose local buffer is written in two loops composes AND "
              "builds with nothing remarked "
              "(tests/ip/test_ip_gaps.py::test_two_write_sites_not_refused); "
              "asic_synthesis/README.md records DC refusing exactly that "
              "shape with ELAB-366 after Vitis emitted a true dual-write-port "
              "RAM still named _1R1W"),
    real_when=("the audit in reduce_csynth.py --audit runs over every "
               "configuration rather than over the one it synthesises, and "
               "Unit.check can see a body's array write sites -- the AST is "
               "already parsed for free_names, so this is the cheapest "
               "unfilled row"),
    kind="cannot refuse",
)

CHAIN_OWNERSHIP = Gap(
    name="chain_ownership",
    what=("holding a stream ARRAY to one writer and one reader per element, "
          "and to indices inside its declared shape"),
    target="none -- this is our own soundness, not a target's requirement",
    needs=("either an abstract interpretation of the index expressions in a "
           "unit body, or a declared per-element ownership rule "
           "(\"stage i reads ch[i] and writes ch[i+1]\") that compose.py can "
           "check the body against"),
    evidence=("two units that both write the same stream array compose "
              "cleanly, while the same pair on a SCALAR channel is refused "
              "(test_two_writers_on_a_chain_not_refused); an index 1000 past "
              "a stream array's declared shape composes, and df.build then "
              "says \"Variable narrow_1000 not defined in current scope\" "
              "(test_chain_index_outside_shape_not_refused). "
              "Architecture._check says why in a comment: the owner is per "
              "element and the element is a runtime index"),
    real_when=("a declared chain rule exists to check against; the exemption "
               "is honest today and a wrong guess here would refuse the "
               "shipped design"),
    kind="cannot refuse",
)

GAPS = (SECOND_INSTANCE, PLACEMENT, COLLECTIVE, UNIT_LATENCY,
        DECLARED_ARITHMETIC, ASIC_LEGAL_MEMORY, CHAIN_OWNERSHIP)

__all__ = ["ASIC_LEGAL_MEMORY", "CHAIN_OWNERSHIP", "COLLECTIVE",
           "DECLARED_ARITHMETIC", "GAPS", "Gap", "NotBuilt", "PLACEMENT",
           "SECOND_INSTANCE", "UNIT_LATENCY"]
