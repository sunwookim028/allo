#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: substitute.sh <minitpu-clone> <vitis-prj> <gen_mxu_vhls.py> [fifo-depth]
# In a LOCAL MiniTPU clone (never the pinned read-only one): replace src/core/mxu/mxu.sv by the generated
# wrapper (module name `mxu`, file name kept), copy the Vitis HLS RTL (<prj>/out.prj/solution1/syn/verilog/*.v)
# to src/core/mxu/allo_vhls/, and list those files in src/minitpu.f after `-f src/core/core.f`. Refuses to touch
# a clone with a remote, or with uncommitted changes. Does not commit (the caller commits, with the record's message).
set -euo pipefail
C=$(readlink -f "$1"); P=$(readlink -f "$2"); GEN=$(readlink -f "$3"); FD=${4:-1024}
test -z "$(git -C "$C" remote)" || { echo "refusing: $C has a remote (substitute only in a local scratch clone)"; exit 2; }
git -C "$C" diff --quiet && git -C "$C" diff --cached --quiet || { echo "refusing: $C is dirty"; exit 2; }
V=$P/out.prj/solution1/syn/verilog
test -d "$V" || { echo "no Vitis RTL at $V"; exit 2; }
python3 "$GEN" "$P" "$C/src/core/mxu/mxu.sv" --fifo-depth "$FD"
rm -rf "$C/src/core/mxu/allo_vhls"; mkdir -p "$C/src/core/mxu/allo_vhls"
cp "$V"/*.v "$C/src/core/mxu/allo_vhls/"
grep -q '^-f src/core/core.f$' "$C/src/minitpu.f"
python3 - "$C" <<'PY'
import os, sys
c = sys.argv[1]
f = os.path.join(c, "src/minitpu.f")
lines = open(f).read().splitlines()
lines = [l for l in lines if not l.startswith("src/core/mxu/allo_vhls/")]
i = lines.index("-f src/core/core.f")
rtl = sorted(x for x in os.listdir(os.path.join(c, "src/core/mxu/allo_vhls")) if x.endswith(".v"))
lines[i + 1:i + 1] = [f"src/core/mxu/allo_vhls/{x}" for x in rtl]
open(f, "w").write("\n".join(lines) + "\n")
print(f"minitpu.f: {len(rtl)} Vitis RTL files after -f src/core/core.f")
PY
git -C "$C" status --short | head -5
echo "substituted: wrapper fifo depth $FD, $(ls "$C/src/core/mxu/allo_vhls" | wc -l) files, sha256(kernel.cpp)=$(sha256sum "$P/kernel.cpp" | cut -c1-16)"
