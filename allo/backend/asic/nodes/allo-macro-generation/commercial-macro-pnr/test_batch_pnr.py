"""Fast tests for macro-PNR batch boundary processing."""

import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("batch_pnr.py")
SPEC = importlib.util.spec_from_file_location("batch_pnr", PATH)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def test_add_macro_power_ground_ports(tmp_path):
    netlist = tmp_path / "macro.lvs.v"
    netlist.write_text(
        "module pe (A, Z);\n"
        "  input A;\n"
        "  output Z;\n"
        "  INV_X1 u0 (.A(A), .ZN(Z), .VDD(VDD), .VSS(VSS));\n"
        "endmodule\n"
    )
    BATCH.add_macro_power_ground_ports(netlist, "pe")
    text = netlist.read_text()
    assert "module pe (A, Z, VDD, VSS);" in text
    assert "inout VDD;" in text
    assert "inout VSS;" in text
    BATCH.add_macro_power_ground_ports(netlist, "pe")
    assert netlist.read_text() == text
