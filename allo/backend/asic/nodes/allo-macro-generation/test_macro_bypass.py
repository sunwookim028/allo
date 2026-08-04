"""Exercise the zero-entry contract without requiring commercial tools."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run_empty_stage(tmp_path: Path, node: str, input_name: str, output_name: str) -> dict:
    work = tmp_path / node
    batch = work / "inputs" / input_name
    batch.mkdir(parents=True)
    (batch / "index.json").write_text(
        json.dumps({"schema_version": 1, "bypass_macro_generation": True, "entries": []})
    )
    script = ROOT / node / {
        "commercial-batch-synthesis": "batch_synthesis.py",
        "commercial-macro-pnr": "batch_pnr.py",
        "commercial-batch-physical-verify": "batch_physical_verify.py",
        "commercial-batch-signoff": "batch_signoff.py",
    }[node]
    subprocess.run([sys.executable, str(script)], cwd=work, check=True)
    index = json.loads((work / "outputs" / output_name / "index.json").read_text())
    assert index["entries"] == []
    assert index["bypass_macro_generation"] is True
    return index


def test_all_commercial_batch_nodes_bypass_empty_batches(tmp_path):
    cases = [
        ("commercial-batch-synthesis", "macro-batch", "synthesis-batch"),
        ("commercial-macro-pnr", "synthesis-batch", "pnr-batch"),
        ("commercial-batch-physical-verify", "pnr-batch", "verified-pnr-batch"),
        ("commercial-batch-signoff", "verified-pnr-batch", "signoff-batch"),
    ]
    for node, input_name, output_name in cases:
        run_empty_stage(tmp_path, node, input_name, output_name)
