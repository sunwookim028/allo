import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("export_workload.py")
SPEC = importlib.util.spec_from_file_location("allo_workload_export", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def test_export_workload_freezes_arrays_as_hex(tmp_path):
    design = tmp_path / "design.py"
    design.write_text(
        "import numpy as np\n"
        "def testbench_workload():\n"
        "    return {\n"
        "      'top_function': 'top',\n"
        "      'call_signature': ['A', 'C'],\n"
        "      'calls': [{'arguments': {\n"
        "        'A': np.array([1, -2], dtype=np.int32),\n"
        "        'C': np.array([-7, -7], dtype=np.int32)},\n"
        "        'expected': {'C': np.array([3, -4], dtype=np.int32)}}]}\n"
    )
    manifest_path = tmp_path / "workload-manifest.json"
    vectors = tmp_path / "workload-vectors"
    EXPORTER.export_workload(
        design,
        True,
        "testbench_workload",
        "top",
        manifest_path,
        vectors,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["call_signature"] == ["A", "C"]
    assert (vectors / "call_000/A.initial.hex").read_text().splitlines() == [
        "00000001",
        "fffffffe",
    ]
    assert (vectors / "call_000/C.expected.hex").read_text().splitlines() == [
        "00000003",
        "fffffffc",
    ]
