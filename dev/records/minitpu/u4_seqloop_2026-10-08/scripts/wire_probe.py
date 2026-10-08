# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: the sequencer loop's same-cycle exchanges as the RTL has them -- Wires.

    $ALLO_PYTHON dev/records/minitpu/u4_seqloop_2026-10-08/scripts/wire_probe.py [--n 300] [--kind wire|comb|wire-x]

``template/sequencer.py`` with every channel of ``EXCHANGES`` declared
``kind="wire"`` (or ``"comb"``), built through ``Architecture.region("systemc")``
for csim (the simulator emits every link as a Stream and is not asked). The
``iv_by_level`` vectors stay Streams (an array of Wire links is refused by
``Channel``), the D-23 commands stay Streams. Prints what the tool does: a
refusal (where and why), a build failure, a hang, or the values against the
reference.
"""

import argparse
import dataclasses
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)


def exchanges_only(arch, names):
    """``wire-x``: the memories' servers as the simulator lowering has them
    (Stream links), only the exchanges as ``Wire`` -- the region text of
    ``source("simulator")`` with those declarations rewritten, executed as
    ``Architecture.region`` does."""
    import linecache
    import re

    from allo.compose import FRONTEND_NAMES

    src = arch.source("simulator", {"iram": "registers", "lb": "registers"})
    for c in names:
        src, k = re.subn(rf"\n(\s+){c}: Stream\[([^,]+(?:\([^)]*\))?), XD\]", rf"\n\1{c}: Wire[\2]", src)
        assert k == 1, (c, k)
    path = "<composed seqloop wire-x>"
    linecache.cache[path] = (len(src), None, src.splitlines(True), path)
    ns = dict(FRONTEND_NAMES)
    ns.update(arch.parameters)
    ns.update(arch._engine_namespace())  # pylint: disable=protected-access
    ns.update(arch._call_namespace())  # pylint: disable=protected-access
    exec(compile(src, path, "exec"), ns)  # pylint: disable=exec-used
    return ns[arch.name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--kind", default="wire")
    ap.add_argument("--prj", default="/tmp/u4_seqloop_wire")
    a = ap.parse_args()
    import allo.dataflow as df
    from allo.compose import Architecture
    from examples.minitpu.harness import check, rtl
    from examples.minitpu.template import sequencer as T
    from examples.minitpu.units import sequencer as S

    cmd, _ = check._trace_all(S, "loop", a.n, "loop")
    n = len(next(iter(cmd.values())))
    base = T.architecture(n)
    names = {c for c, _, _ in T.EXCHANGES}
    kind = "wire" if a.kind == "wire-x" else a.kind
    ch = tuple(dataclasses.replace(c, kind=kind) if c.name in names else c for c in base.channels)
    arch = Architecture(name="seqloop_wire", parameters=base.parameters, memories=base.memories, channels=ch,
                        units=base.units)
    print(f"WIRE-PROBE kind={a.kind} n={n}: {len(names)} exchanges as {a.kind}")
    try:
        if a.kind == "wire-x":
            top = exchanges_only(arch, names)
        else:
            top = arch.region("systemc", {"iram": "registers", "lb": "registers"})
    except Exception as e:  # noqa: BLE001
        print(f"  REFUSED at composition: {type(e).__name__}: {str(e)[:800]}")
        return 2
    try:
        mod = df.build(top, target="systemc", mode="csim", project=os.path.join(a.prj, f"{a.kind}_n{n}"))
    except Exception as e:  # noqa: BLE001
        print(f"  REFUSED at build: {type(e).__name__}: {str(e)[:1500]}")
        traceback.print_exc(limit=3)
        return 3
    t0 = time.time()
    got = T.run(mod, cmd, n)
    packed = {p: rtl.pack(cmd[p], w) for p, w in S.LOOP_RTL.inputs}
    want, reason, _ = S.REF("loop", packed)
    bad = tot = 0
    first = []
    for p in want:
        r = rtl.unpack(want[p])
        for t in range(n):
            if not reason[p][t]:
                tot += 1
                if int(got[p][t]) != int(r[t]):
                    bad += 1
                    if len(first) < 6:
                        first.append(f"cycle {t} {p}: allo {int(got[p][t]):x} rtl {int(r[t]):x}")
    print(f"  RAN in {time.time() - t0:.1f}s: {tot - bad}/{tot} defined slots equal")
    for f in first:
        print(f"    e.g. {f}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
