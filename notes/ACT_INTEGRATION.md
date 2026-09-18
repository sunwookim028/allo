# ACT integration: what exists, and the one change that would connect it

Audited 2026-09-18 against `chia-codesign` (`allo/exp/dsa/`, 11 files) and
`main` (`examples/accelerator/tinytpu_vitis/`). `main` has **no ACT sources at
all** -- `allo/exp/dsa/` here holds only stale `.pyc` files left by a branch
switch.

## The finding

**ACT's mapper is already a loop-nest mapper. Everything below it is defined
over a flat list of steps.** So the obstacle is ACT's *backend*, not its model,
and the cheap fix is on our side rather than in ACT.

`mapping.Loop(rank, factor, level, spatial)` and `mapspace.mapspace` enumerate
factorizations x permutations x spatial splits -- a real loop nest, and close to
what our `LOOP`/`ENDLOOP` + 3-term AGU encoding *is* a serialization of. Then
`mapping._lower` walks that nest into flat steps and **discards it**. From there
down everything assumes flatness: `OracleProgram.steps` is a flat list,
`build_main` walks it linearly, `epoch.epochs` makes one epoch per emit, and
`_Planner`'s liveness indexes steps by integer position.

### The minimum change, both directions

| | scope |
| --- | --- |
| **ACT side** | Large. Loop nodes in `steps`/`emits`; a loop-aware `build_main` (the allo dialect has no loop-of-emits op); per-iteration liveness / allocation / spill in `_Planner`; loop-aware `epochs` and `check`; address params as affine functions of induction variables; and an imem-capacity objective, since `Priced.cost` is `(makespan, emits)` with `emits` only a tie-break -- nothing models instruction memory. |
| **our side** | Moderate, localized. Write an ACT `isa.py` for our 7 data opcodes, take ACT's flat stream, and **re-roll it**. |

**Recommended: use ACT as a mapper only** -- read the chosen `Loop` nest out of
`Mapping` and have `isa_dsl.py` emit `LOOP`/`ENDLOOP` + AGU from it, bypassing
ACT's codegen. We already have the forward direction and, more usefully, the
equivalence check: `isa_dsl.py` derives AGU levels from `with k.loop(...)`
nesting and `bench_isa.py` asserts `expand(looped) == flat` at every shape, so
the re-roll is the missing inverse of an identity we already test.

**The caveat that has to travel with that recommendation:** `mapspace.py` is
**dead code**. Nothing imports it, no test exercises it, it is absent from
`__init__.py`'s `__all__`, and its docstring calls itself "refactor target 4".
Taking this route makes us its first user.

## Our 9 opcodes against ACT's ISA model

| opcode | expressible | blocker |
| --- | --- | --- |
| `DMA_LD`, `DMA_ST`, `VLD`, `MVOUT` | yes | movers: `contiguous`/`strided` + `primitive.identity` |
| `VADD`, `VRELU` | yes | -- |
| `MM` (plain) | yes | -- |
| `MM` (accumulating) | **no** | reads its own destination, so pattern matching returns `None` and it is "oracle-only". Workaround: model the accumulator as an explicit source. |
| `MM` (stationary weight latch) | partial | a constant-addressed read resolves only *inside one* `@expand`; no ISA-level stationary-register concept |
| `NOP` | no | needs both a compute and an access region, and has neither |
| **`LOOP` / `ENDLOOP`** | **no** | touch no buffer and compute no value; they mutate PC and the loop stack, and ACT has no machine state outside buffers and no step kind for them |

Our AGU terms are likewise unexpressible: our addresses are affine in loop
induction variables, while an ACT emit's address params are concrete integers
bound at call time.

## Two TinyTPUs, not one

They share no code. The branches' merge base (`76130c63`) predates both;
`tinytpu_vitis` does not exist on `chia-codesign` and `tinytpu` does not exist
on `main`; the two `microarch.py` files share only their SPDX header.

| | ours (`main`) | chia's (`chia-codesign`) |
| --- | --- | --- |
| data type | int8 / int32 | fp32 |
| instruction | 2 packed `UInt(64)` words | 4-5 `i32` words |
| control | PC + 4-deep loop stack + 3-term AGU | flat PC walk, no loop stack |
| ISA | 9 opcodes, hand-written | 11 ACT-declared instructions |

**Chia's TinyTPU is not a loop machine either**, which reframes the whole
question: ACT emitting flat streams is not an oversight, it matches the only
machine ACT has ever targeted. Ours would be the first.

## Corrections to earlier claims in this repo's history

* "A minimal ACT pilot was run, using a toy ISA." **No artifact exists.** There
  are no `pilot` hits in either repo's notes or git log. If it ran, it ran in
  `~/allo-act`, deleted 2026-09-07 (`CHIA_CHECKPOINT.md`). Nothing in
  `chia_runs/` is it: `20260905-060830/variants.jsonl` is a single baseline
  line, and `swarm-20260905-063857/` is a 6-worker **CHIA LLM-agent** search
  (best 126,432 -> 31,056 cycles on the `dram` hypothesis), which is not a
  mapspace search.
* "ACT has a mapspace search" -- it has a mapspace *file*. See above.
* "Flat needs 106 imem words vs our 56." The 106 is right; the 56 was
  `IMEM_SIZE`, not the program. Measured: `gemm.relu` at 16x16x16 is 49
  instructions = 106 words flat, 17 instructions = **42 words** looped. Flat
  still does not fit; the headroom is 14 words, not 0.
