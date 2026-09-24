# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does the ASIC manifest still describe the design it claims to describe?

`allo/backend/asic` plans a chip from `asic_manifest`: it places macros from
the PID deltas in `semantic_id`, weights the wires between them by the stream
widths in `channels[].type`, and decides which RTL modules are the same macro
from the equivalence records. None of that is checkable by looking at the
emitted Verilog, so nothing downstream can catch a manifest that quietly stops
matching the design -- it just plans a different chip.

So the assertions here are of two kinds, and both matter:

* **against the design**: every expectation is computed from
  `TPU.architecture` -- its units, their `instances=`, its channels and their
  shapes -- rather than typed in, so changing TinyTPU changes what this file
  demands and a manifest that did not follow fails;
* **against the consumer**: the flow's own `plan_macros.graph_pin_sides` and
  `plan_physical_intent.semantic_kernel`/`semantic_pid` are loaded from
  `allo/backend/asic/nodes/` and run on the emitted manifest, so "the consumer
  accepts it" is executed rather than asserted.

The geometry test is the sharp one. The flow is told nothing about systolic
arrays; it is given PIDs and channel endpoints. If it reads back `a_fwd` going
east and `p_fwd` going south, the manifest carried the array.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import allo
import allo.dataflow as df
from allo.backend import asic_manifest
from allo.ir.types import int32, Stream
from allo.passes import analyze_arg_load_store

from examples.tinytpu.microarch_isa import TPU, tinytpu_isa

NODES = Path(allo.__file__).parent / "backend" / "asic" / "nodes"
MANIFEST_CONFIG = {
    "enabled": True,
    "path": "asic-manifest.json",
    "debug_artifacts": True,
    "debug_dir": "asic-debug",
}


def _load(name, relative):
    """Load one of the flow's node scripts as a module, by path.

    They are not a package -- several share a basename and several assume
    their own directory is the working directory -- so importing one is by
    file, which is also how the node itself runs them.
    """
    path = NODES / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan_macros = _load(
    "allo_asic_plan_macros",
    "allo-macro-generation/allo-asic-macro-plan/plan_macros.py",
)
physical_intent = _load(
    "allo_asic_physical_intent",
    "allo-full-chip/allo-asic-physical-intent/plan_physical_intent.py",
)


@pytest.fixture(scope="module")
def tinytpu():
    """The shipped TinyTPU, compiled once, as its pre-HLS manifest."""
    schedule = df.customize(tinytpu_isa)
    arguments = df._build_manifest_top_arguments(
        schedule.func_args[schedule.top_func_name],
        [a.top_name for a in schedule.func_args[schedule.top_func_name]],
        analyze_arg_load_store(schedule.module)[schedule.top_func_name],
        asic_manifest.DIRECTION_MAP,
    )
    return asic_manifest.collect(
        schedule.module,
        schedule.top_func_name,
        top_arguments=arguments,
        mappings=getattr(tinytpu_isa, "mappings", None),
        backend="vitis",
        clock_period_ns=3.33,
    )


def _expected_instances():
    """{kernel: [grid extent]} straight off the composed architecture."""
    parameters = TPU.architecture.parameters
    return {
        unit.name: [int(eval(dim, {}, dict(parameters))) for dim in unit.instances]
        for unit in TPU.architecture.units
    }


def _expected_channel_count():
    """One channel per element of every declared `Channel`, arrays included."""
    parameters = TPU.architecture.parameters
    total = 0
    for channel in TPU.architecture.channels:
        count = 1
        for dim in channel.shape:
            count *= int(eval(dim, {}, dict(parameters)))
        total += count
    return total


def test_manifest_describes_the_composed_architecture(tinytpu):
    assert tinytpu["stage"] == "pre_hls"
    assert tinytpu["schema_version"] == asic_manifest.SCHEMA_VERSION
    assert tinytpu["top"] == TPU.architecture.name

    expected = _expected_instances()
    assert tinytpu["kernel_grids"] == expected
    assert len(tinytpu["pe_instances"]) == sum(
        _product(grid) for grid in expected.values()
    )
    assert len(tinytpu["channels"]) == _expected_channel_count()
    assert [memory["name"] for memory in tinytpu["memories"]] == [
        memory.name for memory in TPU.architecture.memories
    ]


def _product(values):
    total = 1
    for value in values:
        total *= value
    return total


def test_every_wired_channel_has_one_producer_and_one_consumer(tinytpu):
    """`Architecture._check` demands it of the design; the manifest must say it.

    A channel that reaches the flow with two producers, or half-wired, is a
    netlist the placer will happily lay out and the hardware cannot implement.

    The exception is real and is kept rather than hidden: a stream ARRAY
    declares an element per grid point, and each of TinyTPU's three chains
    leaves its terminal element unwired -- the last column forwards no weights
    or activations east, and the bottom row sends `cw` rather than a partial
    sum south. That is `3 * T` elements, and the count moving is how a wiring
    change would show up here.
    """
    unconnected = []
    for channel in tinytpu["channels"]:
        roles = [endpoint["role"] for endpoint in channel["endpoints"]]
        if not roles:
            unconnected.append(channel["channel_id"])
            continue
        assert sorted(roles) == [
            "consumer",
            "producer",
        ], f"{channel['channel_id']} has endpoints {roles}"
    assert len(unconnected) == 3 * TPU.architecture.parameters["T"]
    assert tinytpu["summary"]["unconnected_channels"] == len(unconnected)


def test_memories_name_the_axi_adapters_the_backend_will_emit(tinytpu):
    """The `gmem<i>_m_axi` adapters are `bundle=gmem{i}` on argument `i`.

    `postprocess_hls_code` pragmas them that way, and the shipped synthesis
    hierarchy report names `gmem0_m_axi_U` .. `gmem3_m_axi_U`.
    """
    top = tinytpu["top"]
    for ordinal, memory in enumerate(tinytpu["memories"]):
        assert memory["bundle"] == f"gmem{ordinal}"
        assert memory["rtl_adapter_module"] == f"{top}_gmem{ordinal}_m_axi"
        assert memory["pe"] in {pe["semantic_id"] for pe in tinytpu["pe_instances"]}


def test_semantic_ids_round_trip_through_the_flows_own_parsers(tinytpu):
    for pe in tinytpu["pe_instances"]:
        semantic_id = pe["semantic_id"]
        assert physical_intent.semantic_pid(semantic_id) == tuple(pe["pid"])
        assert physical_intent.semantic_kernel(semantic_id) == (
            f"{tinytpu['top']}/{pe['kernel']}"
        )


def test_the_flows_placer_recovers_the_systolic_geometry(tinytpu):
    """Run the consumer, and check it reads the array back out.

    `graph_pin_sides` is `plan_macros`'s own function: it joins
    `channels[].endpoints[].accesses[].port_ordinal` to `pe_instances[].ports`
    and turns PID deltas into compass sides. TinyTPU sends activations east
    (`a_fwd`) and partial sums south (`p_fwd`), and nothing in the manifest
    says so in words.
    """
    sides = plan_macros.graph_pin_sides(tinytpu)
    assert sides, "the consumer resolved no stream pins at all"

    by_pe = {pe["semantic_id"]: pe for pe in tinytpu["pe_instances"]}
    interior = f"{tinytpu['top']}/pe/pid=1,1"
    observed = {}
    for port in by_pe[interior]["ports"]:
        decision = sides.get((interior, port["ordinal"]))
        if decision is None:
            continue
        observed[(port["stream"].rsplit("_", 2)[0], port["direction"])] = decision[
            "side"
        ]
    assert observed[("a_fwd", "in")] == "W"
    assert observed[("a_fwd", "out")] == "E"
    assert observed[("p_fwd", "in")] == "N"
    assert observed[("p_fwd", "out")] == "S"

    methods = {decision["method"] for decision in sides.values()}
    assert "same_kernel_neighbor" in methods


def test_the_compilation_nodes_pre_hls_gate_accepts_it(tinytpu):
    """The checks `nodes/allo-asic-compilation/validate_build.py` runs."""
    assert tinytpu["stage"] == "pre_hls"
    assert tinytpu["pe_instances"]
    assert tinytpu["summary"]["unmatched_or_ambiguous"] == 0
    assert tinytpu["summary"]["unjoined_post_hls_records"] == 0


def test_a_manifest_that_stops_describing_the_design_is_refused(tinytpu):
    """The emitter checks what the consumers assume and never state.

    Each of these is a `KeyError`, a `StopIteration` or a silently misplaced
    macro several nodes downstream, so it is caught where it is written.
    """
    broken = json.loads(json.dumps(tinytpu))
    broken["channels"][0]["endpoints"][0]["pe"] = "nowhere/at/pid=0"
    with pytest.raises(RuntimeError, match="unknown PE"):
        asic_manifest._check(broken)

    broken = json.loads(json.dumps(tinytpu))
    broken["channels"][0]["endpoints"][0]["accesses"][0]["port_ordinal"] = 99
    with pytest.raises(RuntimeError, match="does not have"):
        asic_manifest._check(broken)

    broken = json.loads(json.dumps(tinytpu))
    broken["pe_instances"][1]["semantic_id"] = broken["pe_instances"][0]["semantic_id"]
    with pytest.raises(RuntimeError, match="duplicate"):
        asic_manifest._check(broken)


def test_build_manifest_top_arguments_uses_the_realized_order():
    """A region's arguments are not emitted in the order they were declared.

    The flow binds the frozen workload's vectors to ports by this order, so
    getting it from the declaration rather than from the build is a silent
    wrong answer. Same contract as the flow's own
    `nodes/allo-asic-compilation/test_catapult_manifest.py`.
    """
    from types import SimpleNamespace

    source = [
        SimpleNamespace(name="A", top_name="A", shape=(4,), dtype="i16"),
        SimpleNamespace(name="B", top_name="B", shape=(8,), dtype="i32"),
        SimpleNamespace(name="C", top_name="C", shape=(2,), dtype="f16"),
    ]
    arguments = df._build_manifest_top_arguments(
        source,
        ["A", "C", "B"],
        ["in", "out", "both"],
        {"in": "input", "out": "output", "both": "inout"},
    )
    assert [item["name"] for item in arguments] == ["A", "C", "B"]
    assert [item["shape"] for item in arguments] == [[4], [2], [8]]
    assert [item["type"] for item in arguments] == ["i16", "f16", "i32"]
    assert [item["direction"] for item in arguments] == [
        "input",
        "output",
        "inout",
    ]
    assert [item["ordinal"] for item in arguments] == [0, 1, 2]


# ---------------------------------------------------------------------------
# A second design, so nothing here is TinyTPU-shaped by accident
# ---------------------------------------------------------------------------


def _chain_region():
    Ty = int32
    M = 8

    @df.region()
    def chain(A: Ty[M], B: Ty[M]):
        pipe: Stream[Ty, 4]

        @df.kernel(mapping=[1], args=[A])
        def source(local_a: Ty[M]):
            for i in allo.grid(M):
                pipe.put(local_a[i])

        @df.kernel(mapping=[1], args=[B])
        def sink(local_b: Ty[M]):
            for i in allo.grid(M):
                local_b[i] = pipe.get() + 1

    return chain


def test_build_writes_what_the_compilation_node_asks_for(tmp_path):
    """`configs["asic_manifest"]` is the whole interface the node uses.

    It calls `build(project, target, mode, configs)` and then expects the
    manifest beside the project -- this is that contract, on a design that is
    not TinyTPU.
    """
    project = tmp_path / "top.prj"
    df.build(
        _chain_region(),
        target="vitis_hls",
        mode="csyn",
        project=str(project),
        configs={
            "frequency": 300.0,
            "device": "u280",
            "asic_manifest": dict(MANIFEST_CONFIG),
        },
    )
    manifest = json.loads((project / "asic-manifest.json").read_text())
    assert (project / "asic-manifest.tcl").is_file()
    assert (project / "asic-debug" / "pe-instances.tsv").is_file()
    assert manifest["stage"] == "pre_hls"
    assert manifest["backend"] == "vitis"
    assert {pe["kernel"] for pe in manifest["pe_instances"]} == {"source", "sink"}
    assert len(manifest["channels"]) == 1
    assert manifest["channels"][0]["type"] == "!allo.stream<i32, 4>"
    assert [memory["bundle"] for memory in manifest["memories"]] == [
        "gmem0",
        "gmem1",
    ]
    assert manifest["top_interface"]["clock"]["name"] == "ap_clk"


def test_enrichment_joins_the_design_to_the_rtl_it_became(tmp_path):
    """The post-HLS half: which module is which PE, and which are one macro.

    Vitis names a kernel's module `<top>_<instance>`, sometimes with a
    disambiguating suffix, and splits pipelined loops into `<root>_...`
    children. Two PEs are one macro class when their root modules are the same
    text once the instance's own name is spelled out of them -- which is what
    makes macro reuse a claim about hardware and not about naming.
    """
    project = tmp_path / "top.prj"
    df.build(
        _chain_region(),
        target="vitis_hls",
        mode="csyn",
        project=str(project),
        configs={
            "frequency": 300.0,
            "device": "u280",
            "asic_manifest": dict(MANIFEST_CONFIG),
        },
    )
    pre = json.loads((project / "asic-manifest.json").read_text())
    top = pre["top"]

    rtl = tmp_path / "verilog"
    rtl.mkdir()
    body = "module {name} (input ap_clk);\n  wire {name}_w;\nendmodule\n"
    for pe in pre["pe_instances"]:
        instance = pe["rtl_instance"]
        root = f"{top}_{instance}"
        # `sink` gets the disambiguated spelling Vitis uses for a cloned
        # function, so the join is exercised on both forms.
        if pe["kernel"] == "sink":
            root = f"{root}_1"
        (rtl / f"{root}.v").write_text(body.format(name=root))
        (rtl / f"{root}_Pipeline_LOOP_1.v").write_text(
            body.format(name=f"{root}_Pipeline_LOOP_1")
        )
    (rtl / f"{top}.v").write_text(body.format(name=top))
    (rtl / f"{top}_gmem0_m_axi.v").write_text(body.format(name=f"{top}_gmem0_m_axi"))

    final = asic_manifest.enrich(pre, str(rtl))
    assert final["stage"] == "post_hls_enriched"
    assert final["summary"]["unmatched_or_ambiguous"] == 0
    assert final["summary"]["unjoined_post_hls_records"] == 0

    roots = {
        pe["semantic_id"]: pe["post_hls_records"][0]["rtl_root_module"]
        for pe in final["pe_instances"]
    }
    assert roots == {
        f"{top}/source/pid=0": f"{top}_source_0",
        f"{top}/sink/pid=0": f"{top}_sink_0_1",
    }
    for pe in final["pe_instances"]:
        names = [module["name"] for module in pe["post_hls_records"][0]["rtl_modules"]]
        assert len(names) == 2 and names[1].endswith("_Pipeline_LOOP_1")

    # The two roots are the same text once each instance's name is removed, so
    # they hash alike -- but they are different kernels, so they are different
    # macro classes. Equivalence is per kernel by construction.
    assert len(final["macro_groups"]) == 2
    for group in final["macro_groups"]:
        assert group["proof"]["status"] == "proven"
        assert group["member_count"] == 1
        assert group["representative"] in roots
    assert f"{top}_gmem0_m_axi" in final["support_modules"]

    # The macro planner reads the enriched records; it must not choke on them.
    assert plan_macros.graph_pin_sides(final)


def test_enrichment_reports_a_pe_it_could_not_join(tmp_path):
    """Silence is the failure mode worth guarding: the count must move."""
    project = tmp_path / "top.prj"
    df.build(
        _chain_region(),
        target="vitis_hls",
        mode="csyn",
        project=str(project),
        configs={
            "frequency": 300.0,
            "device": "u280",
            "asic_manifest": dict(MANIFEST_CONFIG),
        },
    )
    pre = json.loads((project / "asic-manifest.json").read_text())
    top = pre["top"]
    rtl = tmp_path / "verilog"
    rtl.mkdir()
    (rtl / f"{top}_source_0.v").write_text(
        f"module {top}_source_0 (input ap_clk);\nendmodule\n"
    )
    # `sink`'s module is absent, and a module named after it is left over.
    (rtl / f"{top}_sink_0_stray.v").write_text(
        f"module {top}_sink_0_stray (input ap_clk);\nendmodule\n"
    )
    final = asic_manifest.enrich(pre, str(rtl))
    assert final["summary"]["unmatched_or_ambiguous"] == 1
    assert final["summary"]["ambiguous_pe_instances"] == [f"{top}/sink/pid=0"]
    assert final["summary"]["unjoined_post_hls_records"] == 1


def test_split_instance_prefers_the_regions_own_kernel_names():
    """A kernel whose name ends in a number is only safe with the mapping.

    `stage_2` replicated over a 1-D grid is spelled `stage_2_0`, which is
    indistinguishable from a kernel `stage` at pid `(2, 0)` unless the region
    says which kernels it declared -- so the region's `mappings` is passed in
    and preferred, and the fallback is documented rather than silent.
    """
    mappings = {"stage_2": [4], "pe": [4, 4]}
    assert asic_manifest.split_instance("stage_2_3", mappings) == ("stage_2", (3,))
    assert asic_manifest.split_instance("pe_1_2", mappings) == ("pe", (1, 2))
    assert asic_manifest.split_instance("stage_2_3") == ("stage", (2, 3))


if __name__ == "__main__":
    pytest.main([__file__])
