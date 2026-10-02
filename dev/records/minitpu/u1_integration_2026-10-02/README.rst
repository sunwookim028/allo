U1 integration run, 2026-10-02
==============================

Every U1 unit, every Allo variant, simulator and SystemC csim, on the merged
``u1-pilot`` at ``1d49f23f`` (harness, seven units, the SystemC emitter fixes
S1-S6, C1, C2), on zhang-21. Bindings: a private snapshot of the
``systemc-u1-fixes`` build at ``e238e410`` (``mlir/build/SNAPSHOT``), so no
concurrent rebuild could touch the run. Command, per unit::

   source examples/minitpu/harness/env-zhang21.sh
   $ALLO_PYTHON -m examples.minitpu.harness.check <unit> --backend simulator --backend systemc

Result: every ``bits``-family variant (``bits``, ``bits_pipe``, ``stages``,
``staged``, ``fn``, ``bits_dispatch``, ``netlist``) is **UNIT-MATCH** on both
backends, on each unit's full stimulus (251,936 to 6,019,104 vectors). Every
``native``-family difference falls in a named rule; **no vector is
unexplained** (``grep unexplained *.log`` is empty). Exit status 1 per unit
is the ``native`` UNIT-DIFF. Per-unit logs: ``<unit>.log``; table:
``summary.txt``. Latency is ``untimed`` (simulator) / ``unchecked`` (csim):
it is checked only on RTL (``u1_pipe_2026-10-02.rst``).

An earlier attempt of the same run failed from ``bf16_mul_pipe`` on with
``ImportError: ... libAlloMLIRAggregateCAPI.so ...: file too short``: a
rebuild of the shared build it then symlinked landed mid-run. Hence the
private snapshot.
