# Search backend — algorithm analysis

`allo/exp/dsa/search.py` compiles a **source program** (a TOSA-dialect MLIR module,
supplied as text — we own no source generator) onto a user-described **ISA**, and
emits a `CompiledProgram` that the functional oracle (`oracle.py`) can run.

```
source TOSA text ──parse──▶ Catalog ─▶ match ─▶ solve ─▶ plan ─▶ CompiledProgram ─▶ run (JIT)
                           (index)    Stage 1  Stage 2  Stage 3
```

The pipeline is deliberately **dialect-agnostic**: source ops and ISA instructions
are both reduced to a small *prim-tag* vocabulary (`add/sub/mul/relu/matmul/identity`),
so the matcher never reasons about TOSA directly. TOSA appears only at two boundaries —
the source-op recognizer (`source_tag`) and codegen lowering — both op-general.

`compile_program` returns **plain data only** (offsets, shapes, `EmitRecord`s — no IR
handles), so the parse `ir.Context` is dropped on return.

---

## 0. Catalog — the prim-tag index

Two reductions to the common vocabulary:

- **Instruction → tag.** `instruction_pattern(instr)` traces the compute region to the
  root `TensorProxy` of its DAG; the tag is `root.kind`. Data-movement (`identity`) and
  multi-output instructions return `None` (not matched). `Catalog.patterns[tag]` lists
  `(instruction, root_pattern)` so 1:1 and multi-node instructions are looked up the same way.
- **Source op → tag.** `source_tag(op)` maps `tosa.add/sub/mul/matmul → add/sub/mul/matmul`
  and `tosa.clamp{min_val=0} → relu`. Unrecognized ops → `None`.

`_canon(value)` peels `tosa.reshape` chains to the underlying value (reshape is a pure
layout alias — TOSA wraps batched-matmul operands 2-D↔3-D at I/O). Everything downstream
keys on canonical values, so reshapes are transparent to both matching and allocation.

---

## 1. `match_program` — instruction selection (cost-aware tree-DP)

**Goal.** Cover the source compute DAG with instruction patterns at minimum total cost,
folding a multi-node subgraph (e.g. `relu(add)`) into a single instruction when one exists.

**Key structural fact.** A value used by more than one consumer is a **forced cut point**:
it must be materialized to a buffer (it cannot live *inside* one instruction's tile). Once
all multi-use values are cuts, every *foldable* subgraph is a **tree** — so a per-value DP
is globally optimal (no overlap between sibling tiles to coordinate).

**Pre-pass.**
- `def_op`: canonical value → the recognized op that defines it; `index`: → block position.
- `use`: use-count per canonical value, **skipping layout/const ops** (a reshape's read of
  its input is plumbing, not a real consumer).

**DP — `materialize(v)` (memoized).** For each candidate `(instr, root)` whose root tag
equals `source_tag(def_op[v])`:
- `_match_pattern(root, v, …)` does a **rooted structural isomorphism**: an `arg` leaf binds
  a source buffer (`buffer_index → value`); an internal node must align with a recognized
  source op of the same tag and recurse on its data operands (`_source_ins`, which drops
  `tosa.mul`'s shift and `tosa.matmul`'s zero-points). Commutative prims (`add`, `mul`) try
  both operand orders. A **non-root folded value must be single-use** (else it is a cut and
  cannot be absorbed). `bindings` is rolled back on a failed branch.
- Cost = `instr.cost + Σ materialize(operand).cost` over operands that are **single-use
  recognized defs only**. Multi-use operands are *not* billed here — they are materialized
  once at their own root, so charging every consumer would double-count. (This is exactly
  why the cut-point/tree structure makes the DP correct.)
- Keep the minimum-cost `_Choice(cost, instruction, operands)`.

**Reconstruction — `schedule(v)`.** From each terminator operand, walk the chosen tiles,
recursing into recognized operands (deduped by `visited`); emit one `Match` per cut point.
Finally sort matches by source block `index` (a valid topological order).

**Output.** `Selection(func, matches)`, each `Match(instruction, operand_values,
result_value)` with `operand_values` the *raw* source SSA values in source-buffer order.

**Complexity.** `O(V · C · P)` — `V` cut values, `C` candidate instructions per tag, `P`
pattern size; memoization makes each value's tile cost computed once.

**Notes / limits.** Single-output instructions only. Folding does not cross a `tosa.reshape`
(a reshape between two foldable ops makes the inner value unrecognized → treated as a leaf).

---

## 2. `solve` — shape inference as constraint unification

**Goal.** For each matched instruction, infer its symbolic shape parameters (or reject) by
unifying its **parametric visible shape** with the **concrete source shapes**. No tiling:
every dim must fit exactly.

**Where shapes come from.** Each access pattern models *its own* dims from *its own* params
(`PatternExpr.visible_shape()`: `strided`→counts, `expand`→output shape, `collapse`→product
of source dims, `transpose`→permutation). `trace_instruction` composes these into one
`arg_shapes` list per buffer. Dims are `int` (static) or `IndexExpr` (a symbolic affine
expression over access params).

**Param roles (`core.param_roles`).** Every access param is classified by the semantics it
appears under and the partition must be exact (disjoint + complete):
- `offset` — a strided/tiled `basis`: a buffer address, assigned by **allocation** (Stage 3).
- `shape` — a `counts` / `expand`-shape / `tiled`-size entry: a tensor dim, **solved here**.
- `stride` — a strided `stride`: pure addressing, no shape info → **unsupported, rejected**.

**Constraint system.** For each (operand+result) buffer, equate the visible shape with the
source shape `_static_shape(value)` dimension by dimension:
- a **param-free** dim → checked directly (`expects X but source is Y` → reject; no tiling);
- a **param-bearing** dim → an equation `visible_dim == source_dim` (`_to_sympy`).

**Solving (`sympy.linsolve`), three outcomes:**
- **nonlinear** (any equation has total degree > 1, e.g. a collapse of ≥2 symbolic dims —
  `p*q == N` has no unique factorization) → rejected up front by `_is_affine`;
- **empty solution** → the shapes are **inconsistent** (the source does not fit);
- **free symbol remains** → a param is **under-constrained** (no source dim pins it; a future
  explicit constraint could — for now reject and name the param);
- **unique** → each param must be a **non-negative integer** (else reject); record into
  `Match.shape_params{param_index → int}`.

This is the unification view: the instruction has a polymorphic shape signature `(α,β)→γ`;
matching a concrete op assigns the type variables, and the affine common case is a linear
system that is uniquely solvable or fails.

**Complexity.** Per match: `O(D)` constraints (`D` total dims) + one `linsolve` over the few
shape params. Trivial for realistic instructions.

**Notes / limits.** Per-match solving (no cross-instruction param coupling). Collapse with a
single free factor (the rest pinned by another operand) is *not* yet supported — any
multi-symbolic collapse is rejected. Explicit constraints (`M ≤ K`, `N % 4 == 0`) have no
frontend yet but would feed the same system.

---

## 3. `plan` — allocation, data movement, scheduling, emission

**Model — the *location*.** The unit of allocation is `_Loc(value, buffer, size)` with a
runtime `offset` and a live range described by `uses` / `last_use`. A value holds **several
locations over its life** — a `bram` copy and a `vreg` copy, or two `vreg` copies split
around a spill. This single abstraction unifies placement, data movement, and spilling:
a move/spill/reload is just *ending one location and opening another*. `_Loc` is identity-keyed
(`eq=False`) — equal-looking residences must stay distinct. Sizes are per buffer
(`prod(shape) // slot_size`), so the same value occupies 1 vreg slot but 8 bram words.

This is **scratchpad (SPM) allocation + DMA routing**, not uniform-register coloring: values
are variable-size runs, buffers are distinct address spaces, and crossing them costs an
explicit instruction.

### Pass 1 — schedule over locations

Walk the matches in order, lowering each to a linear stream of `_Move` / `_Compute` steps:
- `bring_to(value, target)`: if the value is not resident in `target`, route it there. When no
  direct move edge exists, `_route` runs **BFS over the move graph** (edges = single-src/dst
  `identity` instructions, `_movement_catalog`) for the **shortest hop chain** — `route_move`
  appends one `_Move` per hop, each intermediate buffer getting a short-lived location (**P-C
  multi-hop routing**). Among multiple residences it picks the shortest route.
- Program **inputs** start in the global/`io` buffer; **outputs** are brought back to `io`.
- A `_Compute` step carries the data emit needs *without* the spec: `offset_of` (param→buffer),
  `shape_params`, `n_addr`, and `in_place` (`_in_place_safe`: the result may reuse a dying
  operand's slot iff the compute is purely element-wise).

### Feasibility precondition

For every `_Compute`, the **distinct operand sizes per buffer must fit that buffer**. Operands
are all live at the instruction and none can be spilled (they are in use), so a buffer too
small for one instruction's operands is infeasible → reject. This *also* guarantees the spill
loop terminates: every remaining overflow then has a non-operand victim, so each spill resolves
the earliest overflow and pushes the frontier strictly later.

### Passes 2+3 — liveness + allocation, iterated to a fixpoint

```
loop:  liveness();  outcome = allocate();  if outcome is None: break;  spill(*outcome)
```

- **`liveness`** — over the current `steps`, set each location's `uses` / `last_use`; outputs
  live to a virtual final step. Re-run each iteration because spilling grows the schedule.
- **`allocate`** — one forward walk, **best-fit free-list per buffer** (`free[buf] = [(off,len)]`;
  `best_fit` picks the smallest fitting run → low fragmentation; `release` coalesces adjacent
  runs). At each step:
  - **in-place coalescing** — if the op is element-wise and an operand dies *this* step in the
    result's buffer at the same size, hand its slot to the result;
  - otherwise `best_fit` the result; on overflow return a **Belady victim** to spill;
  - free operands dying this step **after** placing the result — so plain best-fit can never
    silently alias a still-read operand (only the guarded in-place path may). This is what keeps
    a matmul (which re-reads its operands) from corrupting itself.
- **`_pick_victim`** — among locations resident in the overflowing buffer and **not used at this
  step**, choose the one whose *next* use is **farthest** (Belady). Belady is optimal here
  because the whole schedule is known offline. Empty candidate set → infeasible reject.
- **`spill`** — evict the victim to the backing store (`io`) over `[t, next_use)`: store it down
  before step `t`, reload it back before its next use `u`, and **repoint** every later step
  reading the victim onto the reloaded copy. Store/reload are themselves routed over the move
  graph (so spilling is multi-hop-capable). `io` is never spilled (it *is* the backing store).

### Emission

Per step build an `EmitRecord`: a `_Move` emits `[src_off, dst_off]`; a `_Compute` assembles its
`addr` in access-param order — an **offset** param gets its buffer's allocated offset, a **shape**
param gets its solved value. Inputs/outputs are reported as `(offset, shape[, label])` in `io`.

**Complexity.** Each `allocate` pass is `O(steps · buffers)`; fixpoint iterations ≤ number of
spills, each strictly advancing the overflow frontier. No tiling, no slot reuse beyond the
free-list.

**Notes / limits.** Spill target is always `io` (not the *nearest* lower level — capacity-aware
level selection is future). The schedule is the fixed source topological order (no
pressure-aware rescheduling). Exactly one global buffer is assumed.

---

## Driver & public API

- **`ISA.compile_program(source)`** — the public entry (sugar over the module-level
  `compile_program(source, isa)`): `parse → Catalog → match_program → solve → plan`. All IR
  access happens inside the parse context; the result holds only plain data.
- **`CompiledProgram(*inputs)`** (`__call__`) — marshal inputs into the `io` init array at their
  offsets, build an `OracleProgram` (the emit stream + an inspect per output), and JIT-execute via
  `oracle.simulate` (the same backbone hand-written assembly uses). Output slices are reshaped back.
- **`CompiledProgram.dump()`** — print the I/O map + emit stream (mnemonic + addr params) in
  program order, for inspection/debugging.

---

## Cross-cutting invariants

- **Canonical values everywhere** — reshape chains are peeled, so two reshapes of one value share
  its slot and never confuse matching, use-counting, or allocation.
- **Single-use ⇒ tree ⇒ optimal DP** — the cut-at-multi-use rule is what makes the per-value
  tile DP globally optimal and the operand-cost accounting non-double-counting.
- **In-place only for element-wise; operands freed after the result is placed** — the default
  free-list path is always alias-safe; reuse of a same-step operand is the explicit, guarded
  exception.
- **Exact-fit shapes (no tiling)** — every dim matches; mismatch is a hard reject.
- **Spill = backing-store eviction + reload, routed over the move graph; Belady-optimal** because
  the schedule is offline.

## Open items

- Tiling / fission (currently exact-fit only).
- Spill to the *nearest* reachable level rather than always `io`; pressure-aware rescheduling.
- Collapse with one free factor pinned elsewhere; explicit shape constraints (`M ≤ K`, `N%4==0`)
  feeding the same `linsolve` system; cross-instruction param coupling.
- `tiled` visible-shape solving (the access builder exists; `visible_shape` does not handle it).
- Multi-output instructions in the matcher.
