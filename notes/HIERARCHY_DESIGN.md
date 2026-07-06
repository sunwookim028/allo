# Allo hierarchical-region pitch: consolidate or fragment?

A research artifact for Sunwoo Kim, drafted 2026-05-12 before a future
"pitch the maintainers" session. The four items in scope are:

1. `fix/simulator-nested-call-streams` — simulator's `_process_function_streams` does not deep-scan sub-region calls in control flow.
2. `feature/region-scope-stateful` — region-body `@Stateful` shared across inner kernels (crashes on upstream `main`).
3. Per-kernel `static` emission of `@Stateful` globals in `EmitVivadoHLS.cpp` (no branch yet).
4. Bare-scalar auto-capture `s_axilite` (no branch; PR #577 explicitly rejects scalar in `args=[]`).

This file decides whether they go upstream as **four separate bugs/features** or as **one structural design discussion**.

---

## 0. Allo's emerging identity: a DSL for mesh-PE HLS accelerators

What this team is building, and what this file is ultimately about, is a domain-specific language and compiler for **HLS-targeted mesh-network-connected PE-system designs**. The relevant prior art is the *HLS coding pattern* literature — the libraries, reference designs, and compilers that show how a mesh-PE accelerator is actually written in HLS C++ / MLIR — not the architecture-spec literature (TPU papers, NVDLA whitepapers, ISCA microarchitecture descriptions are *not* the right reference set for this conversation).

Reference HLS codebases and pattern sources to cite when pitching upstream:

- **Xilinx Vitis_Libraries** (`github.com/Xilinx/Vitis_Libraries`) — comprehensive HLS reference library spanning vision, DSP, database, solver kernels. Canonical `hls::stream<T>` + DATAFLOW + sub-function patterns at production scale.
- **Vitis_Accel_Examples** (`github.com/Xilinx/Vitis_Accel_Examples`) — end-to-end Vitis kernels demonstrating AXI-Lite control + AXI-MM bulk-data shape (the exact `tpu(ctrl, d_addr, n, dma_buf)` interface).
- **FINN / finn-hlslib** (`github.com/Xilinx/finn-hlslib`) — quantized NN inference; hierarchical mesh of MAC-PE pipelines (parametrized `PE × SIMD`) with shared weight/threshold buffers at module scope.
- **HLS4ML** (`github.com/fastmachinelearning/hls4ml`) — open-source NN HLS; explicit hierarchical layer pipelines with module-scope buffers and per-layer sub-functions.
- **AutoSA** (`github.com/UCLA-VAST/AutoSA`) — automatic HLS systolic-array generation. One of the cleanest existing precedents for "mesh-PE compiled from a high-level description" — the generator output is essentially the ground-truth canonical mesh-PE HLS shape.
- **ScaleHLS** (`github.com/UIUC-ChenLab/ScaleHLS`) — MLIR-to-HLS compiler with hierarchical dataflow regions; Allo's immediate predecessor in the MLIR-to-HLS lineage.
- **Xilinx UG1399 (Vitis HLS User Guide)** — the official "DATAFLOW Optimization" and "Interface Configuration" chapters define the canonical structural pattern.

`allo-tpu` and `allo-npu` are application designs that emit code in this HLS-coding-pattern space. The four items in scope are gaps that would block *any* HLS-mesh-PE design lowered through Allo today — not just these two projects' specific architectures.

The question this file decides — should four items consolidate into one design discussion — reduces to: **is upstream Allo today capable of emitting the canonical HLS mesh-PE shape that Vitis_Libraries, FINN, AutoSA, and ScaleHLS all rely on, or are the four items four ways it cannot?** The verdict in §1 says: cannot, by the same root cause four times.

---

## 1. Verdict

**Consolidate**, with high confidence. The four items are not four bugs; they are four faces of one architectural mismatch — Allo's hierarchical-region machinery (PRs #487, #509, #518, #520, #522, #555, #557, #561) was grafted on top of a single-kernel, IsolatedFromAbove-per-kernel core in a 6-week window in early 2026, and never re-validated against the decoder-plus-driver mesh-connected pattern that allo-tpu and allo-npu need. Pitch a unified RFC, not four PRs.

---

## 2. Meta-narrative recap

Both real-hardware accelerators on this team (`allo-tpu` for U280/XRT, `allo-npu` sibling) implement an *identical* hierarchical pattern: a single `@df.region` declaring four AXI-Lite scalar ports (`ctrl`, `d_addr`, `n`) plus one AXI-MM buffer (`dma_buf`), containing a **decoder** kernel (drains imem, latches PRELOAD, emits commands on a `Stream`) and a **driver** kernel (drains commands, invokes a shared `mxu` sub-region, performs scratchpad reads/writes). The state — `spad`, `acc`, `imem`, latched addresses, per-row softmax statistics — must live at *region* scope so both kernels can read/write the same memory. This is the Gemmini topology and it's the canonical shape every host-driven mesh-connected accelerator needs.

Today, building this in Allo requires every single item in scope: items 1 and 2 to get the simulator running, item 3 to get the HLS C++ to even compile, item 4 to get clean XRT AXI-Lite control without `int32[1]` workarounds. If any one is missing, the others are useless — there's no value in a clean s_axilite if the simulator deadlocks, no value in fixing nested-call stream lowering if region-scope Stateful still crashes the AST builder.

`allo-npu/npu/npu.py:89-378` is the same shape as `allo-tpu`'s `tpu.py`. The pattern generalizes — that's the whole point.

### The canonical mesh-PE HLS pattern (one-paragraph definition)

The four-limb structure that industry HLS expresses natively:

1. **Top-scope shared state**: scratchpad / accumulator / instruction memory live in the top function or top module (not per-PE).
2. **Inter-PE streams**: top scope declares `hls::stream<T>` (Vitis) / `ac_channel<T>` (Catapult) / `sc_fifo` (SystemC) channels; PEs are sub-functions reading and writing those.
3. **Concurrent PE processes**: `#pragma HLS dataflow` on the top function fans out the PEs; each is its own region (function-call or class-instance style depending on backend).
4. **Mixed control / bulk-data interface**: scalar control regs bind to AXI-Lite (`s_axilite`); arrays bind to AXI-MM (`m_axi`).

Mapping each in-scope item to a specific limb this pattern fails to express in upstream Allo today:

| Limb | Item that breaks it today |
|---|---|
| (1) top-scope shared state | **Items 2 + 3** — region-scope `@Stateful` crashes the AST builder (front-end); even when fork-patched, vhls re-emits the static decl in each kernel body (back-end). |
| (2) inter-PE streams | **Item 1** — sub-region calls inside parent control flow leave the callee's streams un-lowered. |
| (3) concurrent PE processes | (lands separately) PR #577's recursive OMP injection makes inner kernels actually run concurrently in the simulator. |
| (4) mixed control / bulk-data | **Item 4** — auto-captured scalars have no `s_axilite` emission; PR #577's rejection of scalars in `args=[...]` pushes users onto a one-way street with no s_axilite path. |

The four items are *four limbs of the same canonical pattern*, not four unrelated bugs.

---

## 3. Evidence summary

### 3.1 Code in `cornell-zhang/allo`

**Timeline of the retrofit.** The fork branch carries the full upstream history; relevant landmark commits ordered by date:

| Date | SHA | Title |
|---|---|---|
| 2023-07 | `1492476` | repo root commit (`scalehls` origins, single-kernel HLS) |
| 2026-01-01 | `dcebc62` (#487) | `[Feature][IR] Add stateful type qualifier` |
| 2026-01-02 | `3a63cc2` (#509) | `[IR][Builder] Change stateful syntax` |
| 2026-01-11 | `448ee83` (#518) | `[dataflow] Support hierarchical modules` |
| 2026-01-12 | `1140acc` (#520) | `Add parameterized dataflow region` |
| 2026-01-12 | `5d4d5c6` (#522) | `[dataflow][simulator] Add hierarchical module support and fix deadlock` |
| 2026-03-05 | `a7508bd` (#557) | `[Fix] several issues related to test_hierachical.py` |
| 2026-03-06 | `83905ea` (#561 fix) | `Fix HLS codegen for hierarchical dataflow regions` |
| 2026-03-13 | `76130c6` (#555) | `Add Allo operations for SPMW` |
| 2026-04-30 | `5c4d1b5` | `feat(builder): support region-scope Stateful` (fork-local) |
| 2026-05-06 | `dbc60b4` | `feat(dataflow): support bare-scalar args` (fork-local, then reverted) |

Allo lived for **2.5 years as a single-kernel-per-region compiler**. Hierarchical regions and persistent state both landed in the *same* 11-day window in Jan 2026, and their interaction was never tested.

**Single-kernel assumptions still visible in the code:**

- `allo/backend/simulator.py:97` — `func_ops = func.body.blocks[0].operations`. The original walk pattern: walk only the top block of the top function. The fix at line 137 (the fork-local `recursive_collect_ops(func, func_d.CallOp, nested_calls)`) is a deep-scan retrofit that *only* matters once you have sub-region calls inside `affine.for` / `scf.if` — which only happens with the decoder pattern.
- `mlir/lib/Translation/EmitVivadoHLS.cpp:3005-3035, 3046-3105` — `emitFunction` walks each function for its own `GetGlobalOp` uses and emits a fresh `static T name[N] = {...};` *inside that function body* (line 3057: hard-coded `os << "static "`). The walker explicitly collects `statefulGlobals` *per function*, with no module-level book-keeping. This is fine for a single-kernel program (one decl, one use). It silently mis-compiles every multi-kernel-sharing-stateful program.
- `mlir/lib/Translation/EmitVivadoHLS.cpp:1536-1543` (`emitGetGlobal`): emits a `// placeholder for const ...` comment then calls `emitValue(..., op.getName().str())`, and `emitValue` routes through `addName` (`mlir/lib/Translation/Utils.cpp:13-35`). `addName` maintains a *module-wide* `nameConflictCnt` — so the *first* function uses the bare symbol name, but the *second* function uses `name1` because `nameConflictCnt[name]` was incremented. The static decl on line 3062 still uses the raw `getSymName()`. This is the SSA-suffix mismatch documented in `handoff.md:248-252`. The two bugs (per-function `static` re-emission + monotonic `nameConflictCnt`) are *two symptoms of the same assumption*: each function is independently emitted, no cross-function global registry.
- `allo/ir/builder.py:1890-1970` — `Stateful` declaration logic uses `ctx.stateful_var_map` and `ctx.global_op_cache`. Comment at `allo/ir/visitor.py:152-157` admits the cache needs careful per-function reset to support region-scope Stateful — the fork branch fixed this; upstream `main` still crashes per issue #565.
- `allo/dataflow.py:539` — `# FIXME: this 'call_op' is required for current 'simulator' backend, but is incompatible with sharding`. Maintainer-acknowledged structural seam between backends.

**What's actually tested today.** The upstream `tests/dataflow/test_hierachical.py:14-44` exercises hierarchical with `@df.region` containing only `@df.kernel` instances and *zero* `Stateful` declarations — only plain arrays passed via `args=[A,B,C]`. The upstream `tests/dataflow/test_hierachical_mesh.py` uses Stateful (51 occurrences), but every Stateful is *kernel-local*: each PE redeclares its own private `imem`, `data_mem`, `A_buf`, `B_buf`, etc. The decoder + driver topology — shared `imem` between two kernels — is not exercised anywhere upstream. The flagship mesh example `examples/feather/feather.py` uses **zero** `Stateful` declarations; all state lives in `Stream`s. The pattern allo-tpu / allo-npu need is *not represented in the upstream test corpus*.

### 3.2 HLS industry practice

**Reference HLS codebases.** The canonical mesh-PE pattern is best read directly from production / research HLS implementations:

- **Xilinx Vitis_Libraries** — Vision L1 / DSP L1 systolic and pipeline kernels. Top function declares all `hls::stream<T>` channels; sub-functions wrap per-PE compute; `#pragma HLS dataflow` on top; scratchpads as top-scope static arrays or passed-by-reference.
- **FINN-HLSLib** — hierarchical mesh of MAC PEs (parameterized `PE × SIMD`); shared weight / threshold buffers at module scope; `hls::stream<ap_uint<W>>` for inter-PE routing; per-layer sub-functions called from a top-level streaming wrapper.
- **HLS4ML** — NN layer pipelines with weight buffers at module scope; sub-functions per compute kernel (`dense_resource`, conv variants); DATAFLOW between layers.
- **AutoSA-generated kernels** — explicit mesh of PEs, all inter-PE FIFOs declared at top, on-chip buffers at top scope. Generator output is essentially "ground truth" for what canonical mesh-PE HLS C++ should look like coming out of a high-level compiler.
- **ScaleHLS hierarchical IR → HLS C++ lowering** — closest existing MLIR-to-HLS comparison point to Allo's pipeline; their region-to-function mapping is the natural reference for how hierarchical regions should lower.

In every one of those codebases, **shared accelerator state lives at the top scope of the hierarchy**, not inside each PE. Allo's vhls output structurally **diverges** from this: it puts `static T spad[N]` *inside* each kernel function (which becomes a per-function private static — wrong semantics — and triggers the redefinition/undeclared-name bugs detailed in §3.1).

**Toolchain conventions** the above codebases all conform to:

- **Vitis HLS DATAFLOW** (Xilinx UG1399, "Dataflow Optimization"): the canonical mesh pattern is a top function declaring inter-PE `hls::stream<T>` channels, sub-functions wrapping each PE's compute, and `#pragma HLS dataflow` on the top function (and any sub-region that fans out). State (scratchpads, ROMs, weight buffers) is declared at the **top function** as static C arrays, or passed in by reference. Control scalars get the `s_axilite` bundle via `#pragma HLS interface s_axilite port=N bundle=control`, arrays get `m_axi`.
- **Catapult HLS hierarchical**: arrays at module scope (`ac_channel<T>` for streams, `mc_array<T>` for shared buffers); each PE is a `class` with its own `run()`; state is class-member or top-scope. Single-static-decl-at-top is enforced by C++ scoping rules.
- **SystemC TLM**: same pattern — `sc_signal` / `sc_fifo` at the top module, `sc_module` per PE, state at the top module level. Hierarchical composition is first-class.

In all three industry patterns, the *natural place for shared accelerator state is the top scope of the hierarchy*, not inside each PE. Allo's vhls output structurally **diverges** from this: it puts `static T spad[N]` *inside* each kernel function (which becomes a per-function private static — wrong semantics — and triggers the redefinition/undeclared-name bugs).

Allo's mesh examples (`examples/feather`, `test_hierachical_mesh`) succeed only because they sidestep this: feather has no `Stateful`, the mesh test gives each PE *private* state. Neither expresses true shared state, so the bug never fires for them.

### 3.3 Maintainer signals

- **Issue #561** (Sunwoo, 2026-03-05) — `[Bug] Hierarchical regions with inter-kernel streams deadlock in simulator and have HLS codegen issues`. Maintainer `Fangtangtang` replied "fixed in #557" for *only the forward-declaration sub-issue*; the simulator deadlock and dataflow-pragma issues were left for the reporter's PR #577.
- **Issue #565** (Sunwoo, 2026-04) — `[Bug] Missing global_op_cache on ASTContext.copy()`. Direct admission that the `ASTContext` propagation logic introduced for hierarchical regions did not anticipate stateful-variable codegen state.
- **PR #577 review** (Fangtangtang, 2026-05-06): on the question "should scalars be allowed in `args=[...]`?", maintainer explicitly chose **option 1: reject**. This pushes users toward auto-capture, which item 4 then shows is *also* incomplete (no `s_axilite` emission). The maintainer made the right local call but unknowingly created a hole on the auto-capture side that item 4 fills.
- **`allo/dataflow.py:539`**: `# FIXME: this 'call_op' is required for current 'simulator' backend, but is incompatible with sharding`. Maintainer-marked structural inconsistency between simulator and other backends.
- **`allo/ir/builder.py:148, 206, 1123, 1146, 1158, 2269`**: six `FIXME (Shihan)` comments around stateful/global-op handling in the AST builder, several explicitly about workaround-needed-for-stateful-Globals. The hierarchical-stateful interaction was known-fragile when it shipped.
- No maintainer has said "this needs rethinking" in plain language, but the pattern of issues — `[Bug] Hierarchical… deadlock`, `[Bug] Missing global_op_cache`, `[Feature] combine features from almost the same framework` (#524) — converges on the same area. The four items in scope, plus the open issues, plus the unmerged feature branches, are *all in the same architectural seam*.

---

## 4. The four items, re-examined

| # | Item | Local cause | Real root |
|---|---|---|---|
| 1 | Sim nested-call stream lowering | `_process_function_streams` doesn't recurse into control flow | Sub-region calls inside `affine.for` only happen with the decoder pattern; the original walker was correct for the single-kernel-flat-call-graph world |
| 2 | Region-scope `@Stateful` | `ASTContext.copy()` doesn't propagate stateful state | Stateful was designed kernel-local before hierarchy existed; region-scope is a composition that the type-annotation builder (#509) doesn't handle |
| 3 | Per-kernel `static` emission | `emitFunction` emits stateful globals inside each function | The HLS emitter treats each function independently; module-level Stateful symbols have no module-level emission site |
| 4 | Bare-scalar auto-capture `s_axilite` | Auto-captured scalars in vitis_hls don't get `s_axilite` pragma | Auto-capture lowering was designed for arrays (memref) and never extended to bare scalars; the AXI-Lite mapping convention only exists in `postprocess_hls_code` for explicit `args=[]` scalars |

**Connections between items:**

- Item 1 ↔ Item 2: If a region-scope Stateful triggers a sub-region call inside `affine.for` (the decoder driver pattern), item 1 is what makes item 2 lowerable. They cannot be tested separately — fixing one without the other means the test case still fails for the other reason.
- Item 2 ↔ Item 3: Item 2 is the **front-end** of region-scope Stateful (Python → MLIR); item 3 is its **back-end** (MLIR → C++). They are literally the two halves of the same feature. Filing them as separate items is misleading; a user landing item 2 alone gets MLIR that the HLS emitter mis-compiles, and landing item 3 alone gets an HLS emitter that nothing produces valid input for.
- Item 4 ↔ PR #577: The maintainer's "reject scalar in `args=[]`" decision *creates* item 4 as a maintainer-acknowledged future-work item. PR #577's review correspondence is the explicit handoff to item 4.
- Item 4 ↔ Items 1/2/3: The `tpu(ctrl, d_addr, n, dma_buf)` signature is *the* trigger for all four items. Without item 4, every user falls back to `int32[1]` workaround, which masks the actual problem the maintainers should be solving.

The four items share four common ancestors: `dataflow.region`, `dataflow.kernel`, `@Stateful`, and the vhls postprocess pipeline. None can be cleanly fixed without touching at least one of the others' files.

---

## 5. Structural model assessment

**Single root assumption (named):** *A `@df.kernel` is an `IsolatedFromAbove` MLIR function, and all sub-region calls happen at the top of the top kernel's body.*

This assumption was correct for the original Allo (2023–2025), where a region was a thin grouping of kernels each of which was effectively a standalone HLS pipeline. Once `Stateful` (Jan 2026, #487/#509) and hierarchical regions (#518/#520/#522) landed within two weeks of each other, the assumption became inconsistent with the new feature surface:

- **Sub-region calls in control flow** (item 1) violate "all calls at top of body" → simulator walker breaks.
- **Region-scope state** (items 2, 3) violates "kernel is isolated from above" → AST builder cache breaks, HLS emitter per-function statics break.
- **Scalar AXI-Lite control plane** (item 4) violates "kernel arg lowering = memref boundary" → auto-capture path has no `s_axilite` channel.

All four are *the same incompatibility* between (a) the kernel-is-an-island model and (b) the host-driven mesh-connected accelerator architecture. Industry HLS handles (b) by making the top function the shared-state owner; Allo's hierarchy was designed before that pattern was in scope.

**The structural diagnosis: Allo's `@df.region` is a function with kernels as bullets, when what mesh accelerators need is a `@df.region` that is a hierarchical module with kernels as concurrently-running processes and the region body as the shared compute substrate.** The four items are the symptoms of asking the first model to do the second model's job.

---

## 6. Implications for allo-tpu / allo-npu / next projects

- Both projects are *already* using the same workaround stack: kernel-local Stateful hoisted to region-scope (forked Allo), int32[1] for control scalars (no s_axilite yet), and an OMP-injection patch for the simulator. The workaround stack is a *de facto* prototype of the upstream redesign.
- Whether item 4 is structurally absorbed (auto-capture → s_axilite) or stays as a `df.scalar` keyword changes the *user-facing* control-plane API for both projects. Pitch this now, before either project goes to tape-out and freezes the host API.
- Future mesh-connected projects on Allo (a transformer accelerator, a graph engine, a DPU) will hit *all four* of these the moment they write a decoder-plus-driver split. The fix is leverage: one good upstream RFC unblocks the next N projects.
- If the items are upstreamed as four PRs, the maintainers may merge 1 and 3 (mechanical), defer 2 (touches IR semantics), and reject 4 in favor of the `int32[1]` status quo. Then allo-tpu and allo-npu still need a fork — the worst outcome.

---

## 7. Open questions for the future session (Sunwoo, pre-pitch)

1. **What does `@df.region` *want* to be?** A grouping of independently-isolated kernels (today's model, with grafted state) or a hierarchical module owning state and channels (industry-standard mesh model)? Pick one explicitly and let the design follow. Without this, the maintainers will continue to merge local fixes that compound technical debt.
2. **Is region-scope Stateful actually compatible with `IsolatedFromAbove`?** The fork's `stateful_var_map` propagation makes it work via name-based symbol lookup, not direct SSA use. That's fine for MLIR, but does it generalize to AIE / Tapa / Catapult backends, or just LLVM-sim + vhls? Pitch should not promise something only the two backends support.
3. **Should the AXI-Lite control plane be a first-class concept, or a vhls-only postprocess?** Today `s_axilite` lives in `postprocess_hls_code` (a string-substitution layer). If it becomes first-class (a new `MemorySpace` / `Bundle` attribute on region args), every backend has to honor it. Worth it?
4. **Do upstream maintainers know about the decoder + driver HLS pattern?** Probably not in the form allo-tpu/allo-npu use it — feather and the hierarchical mesh tests both avoid it. Pitch should include a one-page motivation showing "this is the canonical HLS coding pattern, not an edge case". Cite **Vitis_Libraries** (DATAFLOW + `hls::stream` as the production canonical form), **FINN-HLSLib** (hierarchical PE mesh in production HLS), and **AutoSA** or **ScaleHLS** (precedents for high-level → HLS mesh-PE lowering) as evidence that the pattern is industry-standard *at the HLS code level*.
5. **What's the migration story?** If items 1–4 land as a unified design, existing code using `Stateful` at kernel scope still works (kernel-local Stateful is the only thing exercised upstream). What about `args=[]` with `int32[1]`? Does it stay as a legacy path, or get rewritten by a pass?
6. **Who owns the RFC review?** Fangtangtang reviewed #577 (regions/HLS), `chhzh123` is Allo's PI. The RFC should be addressed to both — and probably to `hc676` (Hongzheng Chen, who authored #518) as the architect of the current hierarchical model.

---

## 8. Suggested architectural alternatives to weigh

Three sketches, not full designs. The future session should pick one as the pitch's centerpiece.

### Alt A: Region-as-Module (industry-aligned)

`@df.region` becomes a SystemC-style hierarchical module:
- Region body owns Streams *and* Stateful arrays as module-scope state.
- Each `@df.kernel` is a concurrent process; access to region state is by name (no `args=` for state, only for top-level interface ports).
- Top-level interface ports get explicit binding attributes: `@s_axilite("ctrl")`, `@m_axi(bundle="gmem0")`. No more auto-capture-vs-args asymmetry.
- HLS emits one C++ function per kernel, one `static T X[N];` per Stateful at file scope, one top wrapper with the right pragmas.

Pros: matches Vitis HLS/Catapult/SystemC, all four items dissolve into a single redesign. Cons: bigger PR, breaks existing `args=[A,B,C]` ergonomics for new users.

### Alt B: Capture-by-name without isolation

Keep `@df.kernel` as a function, but drop `IsolatedFromAbove` for region-scope state symbols. Region-scope Stateful and scalars become *captured* automatically; kernels reference them by name. The MLIR side becomes a pre-pass that lowers captures into per-kernel arg lists *just before* the kernel functions are emitted.

Pros: minimal user-facing change; preserves the current kernel-as-function vocabulary. Cons: keeps the structural mismatch — every backend has to know how to lower captures, items 1/3 still need their own fixes (the items become smaller but don't dissolve).

### Alt C: Single-decl multiple-use Stateful + `df.scalar`

Smallest viable change: keep the current model; add (i) a module-level pass that ensures each `__stateful_*` symbol is emitted at file scope exactly once in vhls, (ii) a `df.scalar(int32)` type that explicitly binds to `s_axilite`. Items 1 and 2 remain as separate bugfixes; items 3 and 4 dissolve.

Pros: easiest to land. Cons: doesn't resolve the structural mismatch — the *next* hierarchical-region project hits the same class of bugs in a different corner. Buys 6 months at most.

Recommended path: pitch Alt A as the long-horizon design, propose Alt C as the bridge-the-gap interim if maintainers want to land things in two weeks instead of two months. Alt B is the worst of both worlds and probably not worth pitching.

---

## 9. What this section does NOT do

- Does **not** draft issue text, RFC text, or PR descriptions. The user will write those after deciding which alternative to pitch.
- Does **not** propose code patches. Item-level fixes already exist on fork branches; the question here is whether to upstream them as-is or rethink first.
- Does **not** recommend specific maintainers to ping or specific timing. The user has direct contact via PR #577.
- Does **not** include benchmarks / performance arguments. The pitch is a correctness-and-composability argument, not a perf argument.
- Does **not** extend the scope to NB streams, Tapa, Catapult, or AIE — only the four named items.

---

## Appendix: key file references

| File | Lines | What lives here |
|---|---|---|
| `allo/backend/simulator.py` | 78-150 | `_process_function_streams`, the recursive-scan retrofit (item 1) |
| `allo/dataflow.py` | 530-549, 578-604 | Region top-level function building, `# FIXME` on sharding |
| `allo/ir/builder.py` | 1890-1970 | `Stateful` AST handling (item 2) |
| `allo/ir/builder.py` | 2150-2278 | `build_FunctionDef` with `is_region` body; kernel call insertion |
| `allo/ir/visitor.py` | 124-177 | `ASTContext.global_op_cache` and `open_function_scope_for_stateful` (item 2) |
| `mlir/lib/Translation/EmitVivadoHLS.cpp` | 1536-1628 | `emitGetGlobal`, `emitGlobal` — Stateful HLS emission (item 3) |
| `mlir/lib/Translation/EmitVivadoHLS.cpp` | 2940-3123 | `emitFunction`, per-function `statefulGlobals` walk (item 3 root cause) |
| `mlir/lib/Translation/Utils.cpp` | 13-35 | `addName` with module-wide `nameConflictCnt` (item 3 mechanism) |
| `allo/dataflow.py` (fork) | _build_top scalar handling | Bare-scalar MLIR type emission (item 4 reference impl) |
| `allo/backend/vitis.py` | postprocess_hls_code | `s_axilite` pragma injection (item 4) |
| Issue #561 | — | Simulator deadlock + HLS forward-decl bug (covers items 1, 2) |
| Issue #565 | — | `global_op_cache` ASTContext.copy() (item 2) |
| PR #518 | — | Hierarchical modules (the retrofit) |
| PR #522 | — | Simulator hierarchical (the retrofit) |
| PR #557 | — | HLS hierarchical fixes (the retrofit) |
| PR #561 | — | Hierarchical HLS codegen fix (Sunwoo's contribution) |
| PR #577 | — | Hierarchical sim deadlock + HLS dataflow pragma (Sunwoo's contribution; review correspondence is the source of the "reject scalar in args" decision) |
| `tests/dataflow/test_hierachical_mesh.py` | 1-240 | The single upstream multi-kernel-Stateful test; uses *kernel-local* Stateful only — pattern allo-tpu/allo-npu need is not exercised |
| `examples/feather/feather.py` | 48-139 | The flagship mesh example; uses *zero* Stateful |
| Local `notes/PITFALLS_DATAFLOW_REGION.md` | 1-50 | The four-item exhibit list as the user found them |
| `ALLO_SHORTCOMINGS.md` | items 1-3 | Same three Allo gaps re-discovered from the TPU side |
