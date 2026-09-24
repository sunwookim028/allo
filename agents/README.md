# agents/

An experiment in **LLM/agent-driven interconnect design**: fix the compute blocks, and let an
agent choose only how they are *wired together*.

The premise is that block logic (a router's arbiter, a PE's ALU) is the part you want written
by a human and verified once, while the interconnect — which link primitive, which topology,
what FIFO depth — is a large, mechanical design space worth searching. So the blocks here are
stripped down to plain callable functions with a declared `PORT_SPEC`, and the agent emits
only the connections between them. A verifier then checks the emitted wiring against
[`INTERCONNECT.md`](INTERCONNECT.md) and each block's `PORT_SPEC`.

## The contract

**[`INTERCONNECT.md`](INTERCONNECT.md)** is the agent-facing reference: the three link
primitives, their methods, and the rules for combining them. Ground truth is
`allo/ir/types.py` (types and methods) and `allo/backend/hls.py` (the backend guard) — the
document exists so an agent does not invent API outside that table.

## Contents — measured state, 2026-08-15

| File | State |
|---|---|
| `INTERCONNECT.md` | **good** — a document, no dependencies to rot |
| `eva_blocks.py` | **good** — EVA blocks as plain functions (router, switch, PE) on abstract ports; imports cleanly |
| `pe_alu.py` | **good** — the simplest possible PE: `(op1, op2, opcode) -> result`, no state, no sequencer; imports cleanly |
| `eva_pe_router_split.py` | **stale, kept** — the EVA design split into router + PE. Unique (not a copy), but dies at `s.partition("node_{i}_{j}:...")` with `RuntimeError: Target function node_0_0 not found`: the kernel-instance naming it assumes no longer matches current Allo. Kept because nothing else holds this split. |

## What was deleted, and why

Six files were removed on 2026-08-15 after every file here was re-run. All are recoverable
from git history.

**Five callability experiments** — `confirm_blocks.py`, `exp_block_callable.py`,
`exp_block_callable2.py`, `exp_err.py`, `cosim_split_sim.py`. Every one failed at import on
`router_XY_stripped` / `router_XY_PEs_bp_stripped`. One cause: those stripped router blocks
lived in `agents/noc/`, which commit `779e435` deleted from this branch — that cleanup moved
the "still-useful pieces" out and left these dependents dangling.

They tested the load-bearing question for this whole approach: whether a `@df.kernel` can
*call* a stripped block function — one that mutates arrays and declares locals — without
losing anything at lowering. To revive them, restore their dependency first:

```bash
git checkout 779e435^ -- agents/noc     # all 53 files
```

**One duplicate** — `eva_sb_syscredit_rtprime.py`, byte-identical (1,785 lines, 88 KB) to
`examples/systemc/eva_example/eva_sb_syscredit_rtprime.py`. That directory is the maintained
home: it has the build and cosim drivers, workloads and a README.

The generator that would consume `INTERCONNECT.md` was never built.

**The one finding worth keeping, because it is not recorded anywhere else:** block
parameters must be annotated — an unannotated parameter silently changes how the block
lowers. The experiments that established this no longer run, and they never wrote their
results down, so this line is the surviving record.

See [`../notes/STATE.md`](../notes/STATE.md) for where this sits relative to the rest of
the work.
