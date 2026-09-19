# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a CYCLIC Allo dataflow region runnable in Vitis csim and cosim.

Vitis runs the C model of a `#pragma HLS dataflow` region by calling its
processes one after another, in the order they appear. A process that reads a
stream fed only by a *later* process therefore reads an empty stream, and in a
design whose process graph has a cycle there is always such a process: csim
stops with

    ERROR [HLS SIM]: an hls::stream is read while empty

and cosim, which runs the C testbench first to record the top-level
transactions, stops with `COSIM 212-360 Aborting co-simulation: C TB simulation
failed`. Csynth accepts the same region and the RTL is concurrent, so this is a
limit of the C model, not of the hardware (probe: `align_probes/probe_cycle.py`, a
two-process ping-pong: Allo simulator OK, csynth OK, csim and cosim fail).

The MiniTPU-aligned machine is cyclic by construction -- one VREG file feeds the
array and receives its results -- so its cosim needs a C model that runs the
processes concurrently. Vitis's own `hls::stream` model is already thread-safe
and blocking (a read on an empty stream waits on a condition variable; the
cosim deadlock detector fires when more reads are blocked than there are
registered tasks). So this rewrites the emitted top function, under
`#ifndef __SYNTHESIS__` only, to start each process call on its own
`std::thread` and join them all. Synthesis sees the original calls unchanged.

The task counter is raised by (processes - 1): the detector fires only when
every process is blocked on a read, which in a region with no other thread is
exactly a deadlock.
"""

import re


def patch(kernel_cpp, top):
    src = open(kernel_cpp).read()
    m = re.search(r"\nvoid %s\(.*?\n\}\n" % re.escape(top), src, re.S)
    assert m, f"top function {top} not found in {kernel_cpp}"
    body = m.group(0)
    call = re.compile(r"^(\s+)(\w+)\((.*)\);(\s*//.*)?$")
    lines = body.split("\n")
    idx = [i for i, l in enumerate(lines) if call.match(l)
           and not l.strip().startswith("#")]
    assert idx, "no process calls in the top function"
    assert idx == list(range(idx[0], idx[-1] + 1)), (
        "process calls are not contiguous; refusing to guess")
    calls = [call.match(lines[i]) for i in idx]
    ind = calls[0].group(1)
    n = len(calls)
    thr = [f"#ifndef __SYNTHESIS__",
           f"{ind}for (int _k = 0; _k < {n - 1}; _k++) "
           f"hls::stream_globals::incr_task_counter();"]
    for k, c in enumerate(calls):
        thr.append(f"{ind}std::thread _t{k}([&]() {{ {c.group(2)}({c.group(3)}); }});")
    thr += [f"{ind}_t{k}.join();" for k in range(n)]
    thr += [f"{ind}for (int _k = 0; _k < {n - 1}; _k++) "
            f"hls::stream_globals::decr_task_counter();", "#else"]
    new = lines[:idx[0]] + thr + [lines[i] for i in idx] + ["#endif"] + lines[idx[-1] + 1:]
    src = src.replace(body, "\n".join(new), 1)
    if "#include <thread>" not in src:
        src = src.replace("#include <hls_stream.h>", "#include <hls_stream.h>\n#include <thread>", 1)
    open(kernel_cpp, "w").write(src)
    return n
