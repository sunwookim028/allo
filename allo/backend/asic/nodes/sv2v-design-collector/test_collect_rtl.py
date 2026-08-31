import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("collect_rtl.py")


def test_bypass_collects_ordered_verilog_and_headers(tmp_path):
    design = tmp_path / "rtl"
    design.mkdir()
    (design / "pkg.v").write_text("module helper; endmodule\n")
    (design / "top.v").write_text("module top; helper h(); endmodule\n")
    (design / "unused.v").write_text("module unused; endmodule\n")
    (design / "defs.vh").write_text("`define VALUE 1\n")
    project = tmp_path / "design"
    project.mkdir()
    construct = project / "construct-commercial.py"
    construct.write_text("# test\n")
    manifest = project / "sources.f"
    manifest.write_text(f"{design / 'pkg.v'}\n{design}/\n!{design / 'unused.v'}\n")
    work = tmp_path / "work"
    work.mkdir()
    env = os.environ | {
        "construct_path": str(construct), "design_path": str(design),
        "manifest": str(manifest), "top_module": "top",
        "sv2v_include_dirs": ".", "normalize_rtl": "False",
        "sv2v_defines": "",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=work, env=env, check=True)
    metadata = json.loads((work / "outputs/rtl-collection.json").read_text())
    assert metadata["top_module"] == "top"
    assert metadata["normalize_rtl"] is False
    assert [Path(item["path"]).name for item in metadata["sources"]] == ["pkg.v", "top.v"]
    assert (work / "outputs/source-rtl/defs.vh").is_file()
    assert (work / "outputs/rtl-source-package/defs.vh").is_file()
    assert (work / "outputs/normalized-rtl/design.v").is_file()
    filelist = (work / "outputs/rtl-sources.f").read_text()
    assert "+incdir+inputs/rtl-source-package" in filelist
    assert "inputs/rtl-source-package/pkg.v" in filelist
    tcl = (work / "outputs/rtl-sources.tcl").read_text()
    assert "set rtl_source_files" in tcl
    assert "set rtl_include_dirs" in tcl


def test_bypass_packages_systemverilog_without_invoking_sv2v(tmp_path):
    design = tmp_path / "rtl"
    design.mkdir()
    (design / "top.sv").write_text(
        'module top #(parameter string MODE = "ASIC"); logic value; endmodule\n'
    )
    construct = tmp_path / "construct.py"
    construct.write_text("# test\n")
    manifest = tmp_path / "sources.f"
    manifest.write_text(f"{design / 'top.sv'}\n")
    work = tmp_path / "work"
    work.mkdir()
    env = os.environ | {
        "construct_path": str(construct), "design_path": str(design),
        "manifest": str(manifest), "top_module": "top",
        "sv2v_include_dirs": ".", "normalize_rtl": "False",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=work, env=env, check=True)
    assert "inputs/rtl-source-package/top.sv" in (
        work / "outputs/rtl-sources.f"
    ).read_text()
    packaged = (work / "outputs/rtl-source-package/top.sv").read_text()
    assert 'parameter MODE = "ASIC"' in packaged
    assert "logic value" in packaged
    assert "parameter string" not in packaged
