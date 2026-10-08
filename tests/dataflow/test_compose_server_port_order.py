# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12 (U4 sequencer loop, S-2): the ``registers`` server answers every
read port of an iteration before it waits on a write-only port, whatever the
order the ports are declared in.

``m`` has a write port ``w`` (owned by ``wr``) and an asynchronous read port
``r`` (latency 0, owned by ``rd``). Each iteration ``rd`` reads ``m[0]`` and
sends it to ``wr``, which writes ``m[0] = x + 1`` -- the loop buffer's replay
read feeding the capture's write, as in MiniTPU's ``sequencer.sv``. Before the
fix the server took the ports in declaration order, so with ``w`` declared
first it waited on the write's address, which waited on the read it had not
answered: the simulator hung with no message
(``tests/limits/new_d12_server_port_order_deadlock.py``). Now both orders
return, and each read sees the previous iteration's write (pre-edge state).
Each build runs in a child process with a timeout, so a regression fails
instead of hanging the suite."""
import multiprocessing as mp

import numpy as np
import pytest

N = 8
TIMEOUT = 120


def _child(q, write_first, target):
    from allo.compose import Architecture, Channel, Memory, Port, unit
    from allo.ir.types import int32  # noqa: F401  (the bodies' names)
    import allo.dataflow as df

    @unit(memories=("O", "m.r"), writes=("x",), parameters=("N",))
    def rd(o: int32[N], mem):
        for t in range(N):
            v: int32 = mem[0]
            o[t] = v
            x.put(v)

    @unit(memories=("m.w",), reads=("x",), parameters=("N",))
    def wr(mem):
        for _ in range(N):
            v: int32 = x.get()
            mem[0] = v + 1

    ports = (Port("w", "w", visible=1), Port("r", "r", latency=0))
    if not write_first:
        ports = ports[::-1]
    arch = Architecture(name="srvorder", parameters={"N": N},
                        memories=(Memory("O", "int32[N]"),
                                  Memory("m", "int32", rows="4", ports=ports,
                                         collision="refuse", reset=False)),
                        channels=(Channel("x", "int32", "2"),), units=(rd, wr))
    region = arch.region("simulator", {"m": "registers"})
    if target == "simulator":
        mod = df.build(region, target="simulator")
    else:
        import tempfile  # noqa: PLC0415

        mod = df.build(region, target="systemc", mode="csim", project=tempfile.mkdtemp())
    o = np.zeros(N, dtype=np.int32)
    mod(o)
    q.put(o.tolist())


def _run(write_first, target="simulator"):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, write_first, target))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join(5)
        pytest.fail(f"ports {'(w, r)' if write_first else '(r, w)'} on {target}: "
                    f"no return after {TIMEOUT} s (the S-2 deadlock)")
    assert p.exitcode == 0, f"child exit {p.exitcode}"
    return q.get()


@pytest.mark.parametrize("write_first", [True, False], ids=["w_first", "r_first"])
def test_server_port_order_simulator(write_first):
    o = _run(write_first)
    # word 0 is unreset (undefined at power-up); each read is the previous + 1
    assert all(o[t + 1] == o[t] + 1 for t in range(N - 1)), o
