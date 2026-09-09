# Minimal repro: a `mapping=` grid hangs when an argument is read by every instance

`a_passes_no_imem.py` is `tests/dataflow/test_tiled_systolic.py` with two
mechanical edits: `M, N, K = 8, 8, 8` and `int32 -> float32`. It **passes**
(`Dataflow Simulator Passed!`).

`b_hangs_with_imem.py` is that file plus one change -- a fourth region argument
`imem: int32[NI]` that every grid instance reads:

```python
def top(imem: int32[NI], A: float32[M, K], ...):
    @df.kernel(mapping=[P0, P1], args=[imem, A, B, C])
    def gemm(li: int32[NI], local_A: ..., ...):
        ...
        op: int32 = li[m * (N // Nt) + n]     # <-- the only addition
```

It **hangs**: every process blocks and the simulator never returns, with no
diagnostic. Reproduce with

    OMP_NUM_THREADS=8 python a_passes_no_imem.py     # passes in ~1 min
    OMP_NUM_THREADS=8 python b_hangs_with_imem.py    # hangs

## What is and is not established

Established, each by a single-variable run:

- It is **not** the shape. The reference passes at both 16x16x16 and 8x8x8.
- It is **not** `float32`. The reference passes with the dtype changed and
  nothing else.
- It is **not** the data-dependent branch. `b_hangs_with_imem.py` has the
  activation conditional written as `if op == 999999`, which is never taken, and
  it still hangs. (With `if op == 1` it also hangs.)
- It is **not** the command-broadcast network. An earlier version decoded in
  PE(0,0) and forwarded the opcode across the top row and down each column; that
  hangs too, for a *different* and understood reason -- the command chain shares
  a schedule with the operand streams, so PE(0,1) blocks pushing an operand to
  PE(1,1), which blocks pushing to PE(1,2), which is still waiting for a command
  from PE(0,2), which is waiting on PE(0,1). That cycle only closes at two or
  more instructions, which is why a single-tile program passes either way.

Not yet separated -- any of these could be the trigger:

- every instance reading the **same address** of the argument (a broadcast),
  where the reference's `A`/`B`/`C` are read at instance-dependent indices;
- the argument **count** rising from 3 to 4;
- an `int32` argument alongside `float32` ones.

The next step is one run each: read `li[i * Nt + j]` (instance-dependent index
rather than broadcast), then make the extra argument `float32`.

## Why it matters

This is the blocker for an instruction-programmable grid on `main`. The
`mapping=` construct gives a genuine spatial array -- a PE is a process, not a
loop iteration -- and it works well for a fixed-function GEMM. But an
accelerator with an ISA needs the decoded instruction to reach every PE, and the
two ways to do that are a shared instruction memory (this repro) or a broadcast
network (the deadlock above). Until one of them works, the grid form can express
the array but not the programmability, which is the opposite of the
`chia-codesign` branch, where the array had to be hand-unrolled but the ISA
works.

Both failures are silent hangs with no diagnostic, which is the same class of
complaint as the `[PREP]` segfaults recorded in that branch's
`FINDINGS_v2.md` section C.
