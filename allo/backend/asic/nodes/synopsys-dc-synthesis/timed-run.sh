#!/usr/bin/env bash
set -uo pipefail

start_epoch=$(date +%s)
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
bash run.sh
returncode=$?
set -e
end_epoch=$(date +%s)
finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p outputs
export SYNTHESIS_RC="$returncode"
export SYNTHESIS_SECONDS="$((end_epoch - start_epoch))"
export SYNTHESIS_STARTED="$started_at"
export SYNTHESIS_FINISHED="$finished_at"
python3 <<'PY'
import json
import os
from pathlib import Path

rc = int(os.environ["SYNTHESIS_RC"])

Path("outputs/synthesis-metrics.json").write_text(
    json.dumps({
        "schema_version": 1,
        "node": "synopsys-dc-synthesis",
        "status": "passed" if rc == 0 else "failed",
        "returncode": rc,
        "wall_seconds": float(os.environ["SYNTHESIS_SECONDS"]),
        "started_at": os.environ["SYNTHESIS_STARTED"],
        "finished_at": os.environ["SYNTHESIS_FINISHED"],
    }, indent=2) + "\n"
)
PY
exit "$returncode"
