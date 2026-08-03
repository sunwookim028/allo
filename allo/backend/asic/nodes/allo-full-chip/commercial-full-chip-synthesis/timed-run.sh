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
export FULL_SYNTH_RC="$returncode"
export FULL_SYNTH_SECONDS="$((end_epoch - start_epoch))"
export FULL_SYNTH_STARTED="$started_at"
export FULL_SYNTH_FINISHED="$finished_at"
python -c '
import json
import os
from pathlib import Path
rc = int(os.environ["FULL_SYNTH_RC"])
Path("outputs/full-chip-synthesis-metrics.json").write_text(json.dumps({
    "schema_version": 1,
    "node": "commercial-full-chip-synthesis",
    "status": "passed" if rc == 0 else "failed",
    "returncode": rc,
    "wall_seconds": float(os.environ["FULL_SYNTH_SECONDS"]),
    "started_at": os.environ["FULL_SYNTH_STARTED"],
    "finished_at": os.environ["FULL_SYNTH_FINISHED"],
}, indent=2) + "\n")
'
exit "$returncode"
