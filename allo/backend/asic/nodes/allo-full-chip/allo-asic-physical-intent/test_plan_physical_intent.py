"""Fast geometry tests for physical-intent planning."""

import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).with_name("plan_physical_intent.py")
SPEC = importlib.util.spec_from_file_location("plan_physical_intent", PATH)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


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
    monkeypatch.chdir(tmp_path)
    PLAN.main()
    intent = json.loads((tmp_path / "outputs/physical-intent.json").read_text())
    assert len(intent["placements"]) == 1
    assert intent["sram_support"] == {
        "enabled": False,
        "instance_count": 0,
        "policy": "sequential_perimeter",
    }
