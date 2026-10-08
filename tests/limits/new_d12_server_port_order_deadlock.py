# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 sequencer loop (dev/records/minitpu/u4_seqloop_2026-10-08.rst, S-2): the
D-12 memory server serves its ports in DECLARATION order each iteration, so a
composition in which a write port's address or enable depends -- through
another channel -- on the same iteration's read of a port declared after it
deadlocks. The simulator hangs with no diagnostic.

``m`` has a write port ``w`` (owned by ``wr``) and an asynchronous read port
``r`` (latency 0, owned by ``rd``). Each iteration ``rd`` reads ``m[0]`` and
sends it to ``wr``, which writes ``m[0] = x + 1`` (the loop buffer's replay
read feeding, through issue and loop control, the capture's write: MiniTPU's
``sequencer.sv``). Nothing in the declaration orders the two ports; the
lowered server does: with ``w`` declared first it ``get``s the write's address
before it answers the read, which the write is waiting on. Declaring ``r``
first runs (each read is the previous one + 1, from the unreset word's
initial value: the read sees the state before the iteration's write, as the
RTL's).

The runs are child processes with a timeout; REPRODUCES when the ``(w, r)``
order hangs and the ``(r, w)`` order returns. Expected after a fix: the server
serves every read port of an iteration before it waits on a write port (reads
see pre-edge state either way), or the composition refuses an order its
channels make cyclic, naming the ports."""
import multiprocessing as mp

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-d12-server-port-order-deadlock"
TIMEOUT = 90
N = 8


def child(q, write_first):
    from allo.compose import Architecture, Channel, Memory, Port, unit
    from allo.ir.types import int32  # noqa: F401  (the bodies' names)

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
                        memories=(Memory("O", "int32[N]"), Memory("m", "int32", rows="4", ports=ports,
                                                                    collision="refuse", reset=False)),
                        channels=(Channel("x", "int32", "2"),), units=(rd, wr))
    import allo.dataflow as df

    mod = df.build(arch.region("simulator", {"m": "registers"}), target="simulator")
    o = np.zeros(N, dtype=np.int32)
    mod(o)
    q.put(o.tolist())


def run(write_first):
    q = mp.Queue()
    p = mp.Process(target=child, args=(q, write_first))
    p.start()
    p.join(TIMEOUT)
    hung = p.is_alive()
    if hung:
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
    return hung, (None if hung else (q.get() if not q.empty() else f"exit {p.exitcode}"))


def main():
    hung_wr, out_wr = run(True)
    hung_rw, out_rw = run(False)
    detail = (f"ports (w, r): {'no return after %d s (killed)' % TIMEOUT if hung_wr else out_wr}; "
              f"ports (r, w): {'hung' if hung_rw else out_rw}")
    verdict(ITEM, hung_wr and not hung_rw, detail)


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
