"""Fast geometry tests for physical-intent planning."""

import importlib.util
import json
import math
from pathlib import Path


PATH = Path(__file__).with_name("plan_physical_intent.py")
SPEC = importlib.util.spec_from_file_location("plan_physical_intent", PATH)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def write_area_report(inputs, total=1000, macro=0):
    reports = inputs / "synthesis-reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "top.mapped.area.rpt").write_text(
        f"Macro/Black Box area: {macro}\nTotal cell area: {total}\n"
    )


def test_optional_srams_and_sequential_perimeter_pack():
    placed, bands = PLAN.edge_pack([], 200, 200, 10, 5)
    assert placed == []
    assert not any(bands.values())
    srams = [
        {"name": "s0", "cell": "S", "width": 40, "height": 20, "symmetry": []},
        {"name": "s1", "cell": "S", "width": 40, "height": 20, "symmetry": []},
    ]
    placed, bands = PLAN.edge_pack(srams, 200, 200, 10, 5)
    assert [item["edge"] for item in placed] == ["bottom", "bottom"]
    assert placed[1]["x"] == placed[0]["x"] + 45
    assert bands["bottom"] == 20


def test_rotated_dimensions():
    assert PLAN.oriented_dimensions(10, 20, "R0") == (10, 20)
    assert PLAN.oriented_dimensions(10, 20, "R90") == (20, 10)


def test_density_sanity_check_rejects_underprovisioned_floorplan():
    predicted, limit = PLAN.validate_predicted_density(70, 100, 0.70)
    assert (predicted, limit) == (0.70, 0.77)
    try:
        PLAN.validate_predicted_density(82, 100, 0.70)
    except ValueError as error:
        assert "under-provisioned" in str(error)
    else:
        raise AssertionError("82% predicted density passed a 77% sanity limit")


def test_variable_kernel_slots_do_not_pad_small_kernel_to_largest_width():
    clusters = {
        "compute": {"width": 100, "height": 40},
        "loader": {"width": 20, "height": 10},
    }
    columns, rows, width, height = PLAN.variable_slot_extents(
        clusters,
        {"compute": (0, 0), "loader": (1, 0)},
        separation_x=30,
        separation_y=15,
    )
    assert columns == {0: 0.0, 1: 130.0}
    assert rows == {0: 0.0}
    assert width == 150
    assert height == 40


def interleave_member(kernel, row, col, name):
    return {
        "kernel": kernel,
        "pid": (row, col),
        "name": name,
        "width": 10,
        "height": 8,
        "orientation": "R0",
        "lef_symmetry": ["X", "Y", "R90"],
    }


def test_exact_repeated_cross_kernel_stencil_is_interleaved_when_enabled():
    grouped = {
        "top/compute": [
            interleave_member("top/compute", 1, 0, "c1"),
            interleave_member("top/compute", 2, 0, "c2"),
        ],
        "top/router": [
            interleave_member("top/router", 1, 2, "r12"),
            interleave_member("top/router", 1, 3, "r13"),
            interleave_member("top/router", 2, 2, "r22"),
            interleave_member("top/router", 2, 3, "r23"),
        ],
    }
    channels = []
    for row in (1, 2):
        for col in (2, 3):
            channels.append({"endpoints": [
                {"pe": f"top/compute/pid={row},0"},
                {"pe": f"top/router/pid={row},{col}"},
            ]})
    accepted, decisions = PLAN.infer_interleave_pairs(
        grouped, {"whole_region_connections": channels}
    )
    assert len(accepted) == 1
    assert accepted[0]["anchor_kernel"] == "top/compute"
    assert accepted[0]["target_kernel"] == "top/router"
    assert accepted[0]["transform"] == "R0"
    assert accepted[0]["stencil"] == [[0, 2], [0, 3]]
    assert accepted[0]["coverage"] == 1.0
    assert decisions[-1]["accepted"] is True
    cluster = PLAN.build_interleaved_cluster(accepted[0], grouped, 4, 5)
    assert [item["name"] for item in cluster["members"]] == [
        "c1", "r12", "r13", "c2", "r22", "r23"
    ]
    assert cluster["width"] == 38
    assert cluster["height"] == 21


def test_irregular_or_partial_cross_kernel_pattern_falls_back():
    grouped = {
        "top/a": [
            interleave_member("top/a", 0, 0, "a0"),
            interleave_member("top/a", 1, 0, "a1"),
        ],
        "top/b": [
            interleave_member("top/b", 0, 1, "b0"),
            interleave_member("top/b", 1, 1, "b1"),
            interleave_member("top/b", 2, 1, "b2"),
        ],
    }
    plan = {"whole_region_connections": [
        {"endpoints": [{"pe": "top/a/pid=0,0"}, {"pe": "top/b/pid=0,1"}]},
        {"endpoints": [{"pe": "top/a/pid=1,0"}, {"pe": "top/b/pid=1,1"}]},
    ]}
    accepted, decisions = PLAN.infer_interleave_pairs(grouped, plan)
    assert accepted == []
    assert decisions == [{
        "kernels": ["top/a", "top/b"],
        "accepted": False,
        "reason": "no exact complete nonoverlapping repeated PID stencil",
    }]


def test_disabled_path_uses_original_rigid_kernel_cluster_geometry():
    members = [
        interleave_member("top/compute", 0, 0, "c00"),
        interleave_member("top/compute", 0, 1, "c01"),
        interleave_member("top/compute", 1, 0, "c10"),
        interleave_member("top/compute", 1, 1, "c11"),
    ]
    cluster = PLAN.build_kernel_cluster(members, 4, 5)
    assert (cluster["width"], cluster["height"]) == (24, 21)
    by_name = {item["name"]: (item["local_x"], item["local_y"]) for item in cluster["members"]}
    assert by_name == {
        "c00": (0.0, 13.0), "c01": (14.0, 13.0),
        "c10": (0.0, 0.0), "c11": (14.0, 0.0),
    }


def test_connection_weights_are_remapped_around_composite_cluster():
    weights = {
        ("top/a", "top/b"): 32,
        ("top/a", "top/c"): 8,
        ("top/b", "top/c"): 16,
    }
    mapping = {"top/a": "interleave:a+b", "top/b": "interleave:a+b", "top/c": "top/c"}
    assert PLAN.remap_connection_weights(weights, mapping) == {
        ("interleave:a+b", "top/c"): 24,
    }


def test_systolic_pid_rows_are_placed_north_to_south(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    macro_dir = inputs / "macro-registry" / "macro_alpha_test"
    macro_dir.mkdir(parents=True)
    (macro_dir / "m.lef").write_text(
        "MACRO m\n  SIZE 10 BY 20 ;\n  SYMMETRY X Y R90 ;\nEND m\n"
    )
    replacements = [
        {
            "stable_instance_name": f"allo_pe_{row}",
            "semantic_id": f"top/kernel/pid={row},0",
            "macro_class_id": "macro_alpha_test",
            "canonical_module": "m",
            "desired_orientation": "R0",
        }
        for row in (0, 1)
    ]
    (inputs / "assembly-plan.json").write_text(json.dumps({
        "top_module": "top",
        "elaborated_macro_instance_count": 2,
        "replacements": replacements,
        "whole_region_connections": [],
    }))
    (inputs / "macro-registry.json").write_text(json.dumps({"macros": [{
        "macro_class_id": "macro_alpha_test",
        "top_module": "m",
        "views": {"lef": {"path": "macro_alpha_test/m.lef"}},
    }]}))
    (inputs / "macro-collateral.json").write_text("{}")
    (inputs / "design.v").write_text(
        "module top; m allo_pe_0(); m allo_pe_1(); endmodule\n"
    )
    (inputs / "macro-link.rpt").write_text("m 2\nTOTAL 2\n")
    write_area_report(inputs, total=1000, macro=200)
    monkeypatch.chdir(tmp_path)
    PLAN.main()
    intent = json.loads((tmp_path / "outputs/physical-intent.json").read_text())
    by_pid = {tuple(item["pid"]): item for item in intent["placements"]}
    assert by_pid[(0, 0)]["y"] > by_pid[(1, 0)]["y"]
    assert intent["pe_stream_facing_policy"]["pid_row_direction"] == "south"


def test_whole_kernel_rotation_transforms_hls_children_as_one_block():
    member = {
        "name": "pipeline_0",
        "local_x": 0,
        "local_y": 0,
        "width": 60,
        "height": 10,
        "orientation": "R0",
        "lef_symmetry": ["X", "Y", "R90"],
        "candidate_kind": "repeated_hls_submodule",
    }
    rotated = PLAN.rotate_cluster(
        {"width": 60, "height": 10, "members": [member]}, "R90"
    )
    assert (rotated["width"], rotated["height"]) == (10, 60)
    assert rotated["members"][0]["orientation"] == "R90"
    assert rotated["members"][0]["candidate_kind"] == "repeated_hls_submodule"


def test_rotation_optimizer_can_turn_thin_owner_kernel_cluster():
    symmetry = ["X", "Y", "R90"]
    clusters = {
        "top/compute": {
            "width": 100,
            "height": 100,
            "members": [{"local_x": 0, "local_y": 0, "width": 100, "height": 100, "orientation": "R0", "lef_symmetry": symmetry}],
        },
        "top/load": {
            "width": 120,
            "height": 10,
            "members": [{"local_x": 0, "local_y": 0, "width": 120, "height": 10, "orientation": "R0", "lef_symmetry": symmetry}],
        },
    }
    result = PLAN.choose_cluster_rotations(
        clusters,
        {"top/compute": (0, 0), "top/load": (1, 0)},
        {("top/compute", "top/load"): 32},
        30,
        30,
        4,
    )
    assert result["top/load"]["orientation"] in {"R90", "R270"}
    assert (result["top/load"]["width"], result["top/load"]["height"]) == (10, 120)


def test_short_row_cleanup_preserves_large_gaps_and_cuts_only_slivers():
    placements = [
        {"x": 10, "y": 10, "width": 20, "height": 20},
        {"x": 40, "y": 10, "width": 20, "height": 20},
        {"x": 80, "y": 10, "width": 15, "height": 20},
    ]
    cuts = PLAN.short_row_fragment_cuts(placements, 100, 50, 2, 12)
    horizontal = [cut for cut in cuts if cut["y"] == 8 and cut["height"] == 24]
    # 8 um at the left edge, 6 um between the first two inflated macros, and
    # 3 um at the right edge are removed. The 16 um middle gap and all
    # unobstructed area survive.
    assert {(cut["x"], cut["width"]) for cut in horizontal} == {
        (0.0, 8.0),
        (32, 6),
        (97, 3),
    }


def test_synthesis_area_keeps_dc_abstract_and_physical_areas_distinct(tmp_path):
    write_area_report(tmp_path, total=1000, macro=200)
    area = PLAN.synthesis_area(tmp_path, require_macro_area=True)
    assert area == {
        "dc_total_cell_area_um2": 1000,
        "dc_macro_abstract_area_um2": 200,
        "estimated_standard_cell_area_um2": 800,
    }
    physical_macro_area = 350
    assert math.isclose(
        area["estimated_standard_cell_area_um2"] / 0.70 + physical_macro_area,
        (1000 - 200) / 0.70 + 350,
    )


def test_synthesis_area_requires_macro_area_for_hierarchical_floorplan(tmp_path):
    reports = tmp_path / "synthesis-reports"
    reports.mkdir()
    (reports / "top.mapped.area.rpt").write_text("Total cell area: 1000\n")
    try:
        PLAN.synthesis_area(reports, require_macro_area=True)
    except ValueError as error:
        assert "requires Macro/Black Box area" in str(error)
    else:
        raise AssertionError("missing DC macro area was accepted")


def test_planner_main_without_optional_srams(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    macro_dir = inputs / "macro-registry" / "macro_alpha_test"
    macro_dir.mkdir(parents=True)
    (macro_dir / "m.lef").write_text(
        "MACRO m\n  SIZE 10 BY 20 ;\n  SYMMETRY X Y R90 ;\nEND m\n"
    )
    plan = {
        "top_module": "top",
        "elaborated_macro_instance_count": 1,
        "replacements": [{
            "stable_instance_name": "allo_pe_test",
            "semantic_id": "top/kernel/pid=0,0",
            "macro_class_id": "macro_alpha_test",
            "canonical_module": "m",
            "desired_orientation": "R0",
        }],
        "whole_region_connections": [],
    }
    registry = {"macros": [{
        "macro_class_id": "macro_alpha_test",
        "top_module": "m",
        "views": {"lef": {"path": "macro_alpha_test/m.lef"}},
    }]}
    (inputs / "assembly-plan.json").write_text(json.dumps(plan))
    (inputs / "macro-registry.json").write_text(json.dumps(registry))
    (inputs / "macro-collateral.json").write_text("{}")
    (inputs / "design.v").write_text("module top; m allo_pe_test(); endmodule\n")
    (inputs / "macro-link.rpt").write_text("m 1\nTOTAL 1\n")
    write_area_report(inputs, total=1000, macro=200)
    monkeypatch.chdir(tmp_path)
    PLAN.main()
    intent = json.loads((tmp_path / "outputs/physical-intent.json").read_text())
    assert len(intent["placements"]) == 1
    assert intent["sram_support"] == {
        "enabled": False,
        "instance_count": 0,
        "policy": "sequential_perimeter",
    }
    generated_tcl = (tmp_path / "outputs/physical-intent.tcl").read_text()
    assert "proc cut_allo_short_row_fragments" in generated_tcl
    assert "proc create_allo_cluster_density_limits" in generated_tcl
    assert "createPlaceBlockage" not in generated_tcl
    assert intent["cluster_placement_policy"] == {
        "type": "none",
        "reason": "cluster-wide partial placement blockages disabled",
        "region_count": 0,
    }
    assert intent["row_fragment_policy"]["cut_count"] == 0
    assert intent["area_budget"]["dc_macro_abstract_area_um2"] == 200
    assert intent["area_budget"]["estimated_standard_cell_area_um2"] == 800
    assert intent["area_budget"]["physical_pe_macro_area_um2"] == 200
    assert intent["area_budget"]["target_standard_cell_density"] == 0.70


def test_planner_main_interleaves_only_with_explicit_flag(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    macro_dir = inputs / "macro-registry" / "macro_alpha_test"
    macro_dir.mkdir(parents=True)
    (macro_dir / "m.lef").write_text(
        "MACRO m\n  SIZE 10 BY 8 ;\n  SYMMETRY X Y R90 ;\nEND m\n"
    )
    replacements = []
    names = []
    for kernel, pids in {
        "compute": [(1, 0), (2, 0)],
        "router": [(1, 2), (1, 3), (2, 2), (2, 3)],
    }.items():
        for row, col in pids:
            name = f"allo_{kernel}_{row}_{col}"
            names.append(name)
            replacements.append({
                "stable_instance_name": name,
                "semantic_id": f"top/{kernel}/pid={row},{col}",
                "macro_class_id": "macro_alpha_test",
                "canonical_module": "m",
                "desired_orientation": "R0",
            })
    channels = []
    for row in (1, 2):
        for col in (2, 3):
            channels.append({"type": "i32", "endpoints": [
                {"pe": f"top/compute/pid={row},0"},
                {"pe": f"top/router/pid={row},{col}"},
            ]})
    (inputs / "assembly-plan.json").write_text(json.dumps({
        "top_module": "top",
        "elaborated_macro_instance_count": len(replacements),
        "replacements": replacements,
        "whole_region_connections": channels,
    }))
    (inputs / "macro-registry.json").write_text(json.dumps({"macros": [{
        "macro_class_id": "macro_alpha_test",
        "top_module": "m",
        "views": {"lef": {"path": "macro_alpha_test/m.lef"}},
    }]}))
    (inputs / "macro-collateral.json").write_text("{}")
    (inputs / "design.v").write_text(
        "module top; " + " ".join(f"m {name}();" for name in names) + " endmodule\n"
    )
    (inputs / "macro-link.rpt").write_text(f"m {len(names)}\nTOTAL {len(names)}\n")
    write_area_report(inputs, total=1000, macro=480)
    monkeypatch.setenv("interleave_macros", "True")
    monkeypatch.setenv("enable_kernel_rotation", "False")
    monkeypatch.chdir(tmp_path)
    PLAN.main()
    intent = json.loads((tmp_path / "outputs/physical-intent.json").read_text())
    policy = intent["cross_kernel_interleaving"]
    assert policy["enabled"] is True
    assert policy["accepted_pair_count"] == 1
    assert len(intent["kernel_clusters"]) == 1
    assert {item["kernel"] for item in intent["placements"]} == {
        "top/compute", "top/router"
    }
    assert {item["spatial_cluster"] for item in intent["placements"]} == {
        "interleave:top/compute+top/router"
    }
    assert "Accepted interleave pairs: 1" in (
        tmp_path / "outputs/physical-intent-report.txt"
    ).read_text()


def test_planner_main_for_flat_bypass(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    (inputs / "macro-registry").mkdir(parents=True)
    (inputs / "assembly-plan.json").write_text(json.dumps({
        "top_module": "top",
        "bypass_macro_generation": True,
        "implementation_style": "flat",
        "elaborated_macro_instance_count": 0,
        "replacements": [],
        "whole_region_connections": [],
    }))
    (inputs / "macro-registry.json").write_text(json.dumps({
        "bypass_macro_generation": True,
        "implementation_style": "flat",
        "macros": [],
    }))
    (inputs / "macro-collateral.json").write_text("{}")
    (inputs / "design.v").write_text("module top; endmodule\n")
    (inputs / "macro-link.rpt").write_text("TOTAL 0\n")
    write_area_report(inputs, total=1000, macro=0)
    monkeypatch.chdir(tmp_path)
    PLAN.main()
    intent = json.loads((tmp_path / "outputs/physical-intent.json").read_text())
    assert intent["placements"] == []
    assert intent["kernel_clusters"] == []
    assert intent["sram_support"]["enabled"] is False
