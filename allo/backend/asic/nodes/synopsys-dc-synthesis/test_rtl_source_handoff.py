from pathlib import Path


def test_packaged_rtl_handoff_preserves_sources_includes_and_defines():
    script = Path(__file__).parent / "scripts/read-design.tcl"
    text = script.read_text()
    assert "source inputs/rtl-sources.tcl" in text
    assert "concat $rtl_include_dirs $search_path" in text
    assert "analyze -format sverilog -define $rtl_defines $rtl_source_files" in text
