import json
from pathlib import Path

import pytest

from resolve_sram_contract import resolved_views, write_tcl


def make_contract(tmp_path: Path):
    root = tmp_path / "srams"
    macro = root / "mem"
    macro.mkdir(parents=True)
    views = {}
    for view, suffix in {
        "verilog": "v", "liberty": "lib", "database": "db",
        "lef": "lef", "gds": "gds", "spice": "sp",
    }.items():
        path = macro / f"mem.{suffix}"
        path.write_text(view)
        views[view] = [str(path.relative_to(root))]
    contract = tmp_path / "sram-contract.json"
    contract.write_text(json.dumps({
        "schema": "sram-collateral-contract", "schema_version": 1,
        "num_srams": 1, "srams": [{"name": "mem", "views": views}],
    }))
    return contract, root


def test_resolves_registered_views_and_writes_tcl(tmp_path):
    contract, root = make_contract(tmp_path)
    count, views = resolved_views(contract, root)
    output = tmp_path / "sram-views.tcl"
    write_tcl(output, count, views)

    assert count == 1
    assert views["database"] == [str((root / "mem/mem.db").resolve())]
    assert "set sram_contract_num_srams 1" in output.read_text()
    assert "set sram_database_files [list" in output.read_text()


def test_accepts_empty_contract(tmp_path):
    root = tmp_path / "srams"
    root.mkdir()
    contract = tmp_path / "sram-contract.json"
    contract.write_text(json.dumps({
        "schema": "sram-collateral-contract", "schema_version": 1,
        "num_srams": 0, "srams": [],
    }))

    count, views = resolved_views(contract, root)
    assert count == 0
    assert all(not paths for paths in views.values())


def test_rejects_count_mismatch_and_package_escape(tmp_path):
    contract, root = make_contract(tmp_path)
    data = json.loads(contract.read_text())
    data["num_srams"] = 2
    contract.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="num_srams"):
        resolved_views(contract, root)

    data["num_srams"] = 1
    outside = tmp_path / "outside.db"
    outside.write_text("db")
    data["srams"][0]["views"]["database"] = ["../outside.db"]
    contract.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        resolved_views(contract, root)

