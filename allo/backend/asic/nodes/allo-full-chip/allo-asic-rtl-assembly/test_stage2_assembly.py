"""End-to-end test of planning and canonical RTL substitution."""

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).parent
PLANNER_PATH = HERE.parent / "allo-asic-assembly-plan" / "plan_assembly.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLANNER = load("plan_assembly", PLANNER_PATH)
ASSEMBLER = load("assemble_rtl", HERE / "assemble_rtl.py")


def test_member_instances_become_one_canonical_macro(tmp_path, monkeypatch):
    rtl = """
module top();
  wrap_a wa();
  wrap_b wb();
endmodule
module wrap_a();
  wire x;
  canonical u_a(.a(x), .z(x));
endmodule
module wrap_b();
  wire x;
  member_b u_b(.b(x), .y(x));
endmodule
module canonical(a, z);
  input a;
  output z;
  assign z = a;
endmodule
module member_b(b, y);
  input b;
  output y;
  assign y = b;
endmodule
"""
    members = [
        {"semantic_id": "top/k/pid=0", "rtl_module": "canonical", "orientation": "unassigned"},
        {"semantic_id": "top/k/pid=1", "rtl_module": "member_b", "orientation": "unassigned"},
    ]
    manifest = {
        "top": "top",
        "pe_instances": [
            {"semantic_id": "top/k/pid=0", "kernel": "k"},
            {"semantic_id": "top/k/pid=1", "kernel": "k"},
        ],
        "channels": [],
        "macro_groups": [{"macro_class_id": "macro_alpha_test", "members": members}],
    }
    view_names = {"verilog": "macro.v", "liberty": "macro.lib", "db": "macro.db", "lef": "macro.lef", "gds": "macro.gds"}
    views = {name: {"path": f"macro_alpha_test/{filename}"} for name, filename in view_names.items()}
    registry = {
        "macros": [{
            "macro_class_id": "macro_alpha_test",
            "top_module": "canonical",
            "reuse_count": 2,
            "members": members,
            "lef_symmetry": ["X", "Y", "R90"],
            "views": views,
            "port_maps": {
                "canonical": [{"canonical": "a", "member": "a"}, {"canonical": "z", "member": "z"}],
                "member_b": [{"canonical": "a", "member": "b"}, {"canonical": "z", "member": "y"}],
            },
        }]
    }

    planner_dir = tmp_path / "planner"
    planner_inputs = planner_dir / "inputs"
    (planner_inputs / "macro-registry" / "macro_alpha_test").mkdir(parents=True)
    (planner_inputs / "design.v").write_text(rtl)
    (planner_inputs / "asic-manifest-final.json").write_text(json.dumps(manifest))
    (planner_inputs / "macro-registry.json").write_text(json.dumps(registry))
    for filename in view_names.values():
        (planner_inputs / "macro-registry" / "macro_alpha_test" / filename).write_text("view\n")
    monkeypatch.chdir(planner_dir)
    PLANNER.main()
    plan = json.loads((planner_dir / "outputs/assembly-plan.json").read_text())
    assert plan["replacement_instance_count"] == 2
    assert plan["elaborated_macro_instance_count"] == 2
    assert {path for item in plan["replacements"] for path in item["hierarchical_paths"]} == {
        "top/wa/u_a",
        "top/wb/u_b",
    }

    assembler_dir = tmp_path / "assembler"
    assembler_inputs = assembler_dir / "inputs"
    assembler_inputs.mkdir(parents=True)
    (assembler_inputs / "design.v").write_text(rtl)
    (assembler_inputs / "assembly-plan.json").write_text(json.dumps(plan))
    (assembler_inputs / "macro-registry.json").write_text(json.dumps(registry))
    (assembler_inputs / "macro-registry").symlink_to(
        planner_inputs / "macro-registry", target_is_directory=True
    )
    monkeypatch.chdir(assembler_dir)
    ASSEMBLER.main()
    assembled = (assembler_dir / "outputs/assembled-design.v").read_text()
    assert "module canonical" not in assembled
    assert "module member_b" not in assembled
    assert "canonical allo_pe_top_k_pid_0_" in assembled
    assert "canonical allo_pe_top_k_pid_1_" in assembled
    assert ".a(x), .z(x)" in assembled
    collateral = json.loads((assembler_dir / "outputs/macro-collateral.json").read_text())
    assert len(collateral["rewritten_instances"]) == 2
    assert "allo_asic_macro_modules" in (
        assembler_dir / "outputs/macro-collateral.tcl"
    ).read_text()
    assert "-period 10" in (assembler_dir / "outputs/constraints.tcl").read_text()
