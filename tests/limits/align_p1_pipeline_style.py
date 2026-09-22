# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item P1: a dataflow process that gets a response and puts the next request
in one pipelined loop deadlocks in RTL unless that loop is flushable.

Original claim (branch `tinytpu-align`, increments 3 and 4): under Vitis's
default *stall* pipeline (`stp`), a blocking stream read in a later iteration
freezes the whole pipeline, including an earlier iteration's pending put.
A process that gets a response and puts the next request in one pipelined
loop therefore waits forever. `style=flp` (flushable) keeps the older
iterations draining. (The register also says `frp` does not help; measured
here, `frp` passes this repro -- see the run log.)
`s.pipeline(axis, ..., style=)` was added to Allo for this (`0038833c`).

The repro: `drv` gets the previous iteration's response and then puts the next
request, in one pipelined loop whose style is the variable; `wrk` answers.
Nothing else is in the region. The same region is taken through Vitis csynth +
cosim once per style; a deadlock shows up as Vitis's own `DEADLOCK DETECTED` /
`HLS 200-742`, and a pass shows up as `mismatches = 0` twice (C, then RTL).

Notes:

* The dataflow simulator cannot see this at all -- it is checked first and
  passes under every style.
* A cyclic region also cannot be C-simulated at all (item P2), and cosim runs
  the C testbench before the RTL, so the emitted top has to be rewritten to
  run its processes on threads. That rewrite is inlined below (the tree's copy
  is `examples/accelerator/tinytpu_vitis/threaded_csim.py`) so this repro
  stands alone.
* Needs Vitis HLS 2023.2 and several minutes per style; `ALLO_P1_STYLES` (e.g.
  `stp` or `stp,flp,frp`) selects which. Default `stp,flp`. The pseudo-style
  `default` emits the pragma with no style at all, which is what a checkout
  without `0038833c` can express -- run `ALLO_P1_STYLES=default` there.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "P1"
import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32, Stream  # noqa: E402

VITIS = "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
LDFLAGS = "-B/usr/bin -lpthread"  # binutils 2.37 vs this glibc; threads for csim
N = 8
STYLES = os.environ.get("ALLO_P1_STYLES", "stp,flp").split(",")
PRJ = os.environ.get(
    "ALLO_P1_PRJ", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".p1")
)


def region():
    """`drv` gets the response for iteration x-1 and then puts the request for
    iteration x, in ONE pipelined loop -- the shape of TinyTPU-align's `vpu`,
    which pops the array's results and pushes its next rows in one loop. The
    get is scheduled ahead of the put it depends on, so under a STALL pipeline
    the blocked get freezes the pending put and neither side can move."""

    @df.region()
    def top(src: int32[N], dst: int32[N]):
        req: Stream[int32, 2]
        resp: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[src, dst])
        def drv(a: int32[N], o: int32[N]):
            for x in range(N + 1):
                if x > 0:
                    o[x - 1] = resp.get()
                if x < N:
                    req.put(a[x])

        @df.kernel(mapping=[1])
        def wrk():
            for y in range(N):
                v: int32 = req.get()
                resp.put(v * 2)

    return top


def thread_the_c_model(kernel_cpp, top):
    """Run each process of the emitted top on a std::thread, under
    `#ifndef __SYNTHESIS__` only (item P2's workaround, inlined)."""
    src = open(kernel_cpp).read()
    m = re.search(r"\nvoid %s\(.*?\n\}\n" % re.escape(top), src, re.S)
    assert m, "top function not found"
    body = m.group(0)
    call = re.compile(r"^(\s+)(\w+)\((.*)\);(\s*//.*)?$")
    lines = body.split("\n")
    idx = [i for i, l in enumerate(lines) if call.match(l) and not l.strip().startswith("#")]
    calls = [call.match(lines[i]) for i in idx]
    ind, n = calls[0].group(1), len(calls)
    thr = ["#ifndef __SYNTHESIS__",
           f"{ind}for (int _k = 0; _k < {n - 1}; _k++) hls::stream_globals::incr_task_counter();"]
    thr += [f"{ind}std::thread _t{k}([&]() {{ {c.group(2)}({c.group(3)}); }});"
            for k, c in enumerate(calls)]
    thr += [f"{ind}_t{k}.join();" for k in range(n)]
    thr += [f"{ind}for (int _k = 0; _k < {n - 1}; _k++) hls::stream_globals::decr_task_counter();",
            "#else"]
    new = lines[: idx[0]] + thr + [lines[i] for i in idx] + ["#endif"] + lines[idx[-1] + 1:]
    src = src.replace(body, "\n".join(new), 1).replace(
        "#include <hls_stream.h>", "#include <hls_stream.h>\n#include <thread>", 1
    )
    open(kernel_cpp, "w").write(src)


def cosim(style):
    prj = os.path.join(PRJ, style)
    shutil.rmtree(prj, ignore_errors=True)
    os.makedirs(prj, exist_ok=True)
    s = df.customize(region())
    if style == "default":
        # What a tree without `s.pipeline(style=)` can express at all: the
        # pragma with no style, i.e. Vitis's stall pipeline. Use this style
        # name to run the repro on a checkout that predates 0038833c.
        s.pipeline("drv_0:x")
    else:
        s.pipeline("drv_0:x", style=style)  # the loop on the cycle
    s.pipeline("wrk_0:y")                   # the responder: Vitis's default
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False)
    src = open(f"{prj}/kernel.cpp").read()
    dep = iter([N, N])  # cosim needs explicit m_axi depths; Allo emits none
    src = re.sub(
        r"(#pragma HLS interface m_axi port=\w+ offset=slave bundle=gmem\d+)",
        lambda m: m.group(1) + f" depth={next(dep)}",
        src,
    )
    open(f"{prj}/kernel.cpp", "w").write(src)
    if style != "default":
        assert f"style={style}" in src, f"the pragma did not carry style={style}"
    else:
        assert "style=" not in src, "expected no style on the pragma"
    thread_the_c_model(f"{prj}/kernel.cpp", "top")
    a = np.arange(1, N + 1, dtype=np.int32)
    g = a * 2
    open(f"{prj}/tb.cpp", "w").write(
        '#include <cstdio>\n#include <cstdint>\nextern "C" void top(int32_t*, int32_t*);\n'
        f"static int32_t A[{N}] = {{{', '.join(map(str, a))}}};\n"
        f"static int32_t G[{N}] = {{{', '.join(map(str, g))}}};\n"
        f"static int32_t O[{N}];\nint main(){{ top(A, O); int bad=0;"
        f" for(int i=0;i<{N};i++) bad += O[i]!=G[i];"
        ' printf("TB mismatches = %d\\n", bad); return bad!=0; }\n'
    )
    open(f"{prj}/run.tcl", "w").write(
        f"""open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
set_part {{xcu280-fsvh2892-2L-e}}
create_clock -period 3.33
csim_design -ldflags "{LDFLAGS}"
csynth_design
cosim_design -trace_level none -rtl verilog -ldflags "{LDFLAGS}"
exit
"""
    )
    rc = subprocess.call(
        ["bash", "-lc",
         f"source {VITIS} && cd {prj} && timeout 1200 vitis_hls -f run.tcl > vitis.log 2>&1"]
    )
    log = open(f"{prj}/vitis.log", errors="replace").read()
    tb = re.findall(r"TB mismatches = (\d+)", log)
    # Vitis names a deadlock DETECTOR in every cosim log, so match its verdict,
    # not the word.
    dead = [l.strip()[:120] for l in log.splitlines()
            if "DEADLOCK DETECTED" in l or "HLS 200-742" in l]
    if dead:
        return f"DEADLOCK ({dead[0]})"
    if rc == 124 or ("timeout" in log.lower() and not tb):
        return "TIMEOUT (1200s)"
    if len(tb) >= 2 and tb[-1] == "0":
        return "PASS (RTL mismatches = 0)"
    if tb and tb[-1] != "0":
        return f"WRONG (mismatches = {tb[-1]})"
    return f"NO RESULT (rc={rc}); see {prj}/vitis.log"


def main():
    # The simulator sees nothing: it passes whatever the style is.
    mod = df.build(region(), target="simulator")
    a = np.arange(1, N + 1, dtype=np.int32)
    o = np.zeros(N, dtype=np.int32)
    mod(a, o)
    print("  simulator:", "OK" if np.array_equal(o, a * 2) else f"WRONG {o.tolist()}", flush=True)
    out = {}
    for style in STYLES:
        out[style] = cosim(style)
        print(f"  style={style:3s} -> {out[style]}", flush=True)
    broken = [s for s, r in out.items() if not r.startswith("PASS")]
    verdict(
        ITEM,
        bool(broken),
        f"styles that do not run in RTL: {broken}; "
        + ", ".join(f"{s}={r}" for s, r in out.items()),
    )


if __name__ == "__main__":
    main()
