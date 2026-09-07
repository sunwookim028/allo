# CHIA co-design checkpoint — 2026-09-07

Maintenance record for the "Agentic Discovery of Domain-Specific Computing
Stack with CHIA" effort. Two purposes: pin what version of everything produced
the results, and track every claim we make (or want to make) against whether it
is push-button reproducible from a remote backup.

## 0. Backup status — resolved 2026-09-07

`chia-codesign` is pushed to `sunwookim028/allo`. Everything below traces to code
that exists on a remote.

`~/allo-act` has been removed. It was safe: its `HEAD` (`0b5fef7`) is reachable
from `chia-codesign` and present on `origin/chia-tinytpu-rtlgen`, its working
tree held only deletions, and its one untracked file (a `vadd_relu` fused
instruction on an older TinyTPU ancestor) was judged not worth keeping.

Still outstanding for an *Artifacts Available* badge: a permanent archive with a
DOI. A git branch alone does not satisfy it.

## 0b. Disk footprint

| Item | Size | Regenerable |
| --- | --- | --- |
| `~/allo-chia` built tree | 9.2 GB | all but `chia_runs` |
| ↳ `chia_runs` | 85 MB | **no — the evidence behind every result** |
| conda `allo` / `chia_env` | 1.5 GB / 0.6 GB | yes |
| `~/chia-tools` (CHIA + opencode) | 736 MB | yes |
| peak during a fresh build | ~19 GB | — |

Reclaimed 2026-09-07: **21 GB** — 16 GB of `~/.cache` (HuggingFace model blobs,
pip, conda tarballs, pre-commit) plus 2 GB of `~/allo-act`, and 5 GB of conda and
pip caches earlier. HuggingFace auth tokens were preserved; only the
re-downloadable model blobs were dropped.

## 1. Component inventory

| Component | Location | Version / pin | Remote | Backed up |
| --- | --- | --- | --- | --- |
| Allo fork (integration HEAD) | `~/allo` | `e78bf9b5` on `main` | `sunwookim028/allo` | yes |
| Allo + ACT (Kai) | *removed 2026-09-07* | `0b5fef7` — reachable from `chia-codesign` and on `origin/chia-tinytpu-rtlgen` | `kkkaishao/allo` | yes, via this fork |
| Allo + ACT + RTLGen + CHIA | `~/allo-chia` | `chia-codesign` | `sunwookim028/allo` | **yes — pushed** |
| ↳ base branch it forks | — | `882f7dd6` `chia-tinytpu-rtlgen` | `sunwookim028/allo` | yes |
| CHIA framework | `~/chia-tools/chia` | `16c35e9` | `ucb-bar/chia` | upstream only |
| opencode CLI | `~/chia-tools/opencode-cli` | `opencode-ai@1.18.25` | npm | pinned |

**Submodule pins** (`~/allo-chia/externals/`), all built and present:

| Submodule | Commit | Build artifact |
| --- | --- | --- |
| llvm-project | `040a641988f6` (LLVM 23.0.0) | `build/bin/mlir-opt` ✓ |
| circt | `af5369d7ea19` | `build/bin/circt-opt` ✓ |
| marl | `b8406ab0a825` | — |
| past-python-bindings | `65f989b86750` | — |
| OR-Tools 9.5 (fetched by CIRCT) | v9.5 tarball | `circt/ext/lib/cmake/ortools` ✓ |

**Toolchain**

| Item | Version | Notes |
| --- | --- | --- |
| Vitis HLS / Vitis / Vivado | 2023.2 | `/opt/xilinx/*/2023.2`; `settings64.sh` must be sourced |
| Target part | `xcu55c-fsvh2892-2L-e` | Alveo U55C, 300 MHz target |
| gcc / clang | 13.3.0 / 18.1.3 | **build with gcc** (see §4) |
| cmake / ninja | 4.2.1 / 1.11.1 | cmake 4 needs the OR-Tools policy shim |
| lld | `ld.lld-18` | shimmed to `ld.lld` in `~/.local/allo-bin` |
| conda env `allo` | Python 3.12.13 | editable install → `~/allo-chia` |
| conda env `chia_env` | Python 3.10.19, ray 2.54.0 | CHIA host |

**Cloud**

| Item | Value |
| --- | --- |
| GCP project | `test-adrs` |
| Billing (current) | `01BC37-C4DA99-068A2F` (free trial — burning the $300) |
| Billing (institutional) | `0148AB-064407-7A5F3C` — **relink here when trial credit is gone** |
| Model | `google-vertex/gemini-3.1-pro-preview`, location `global` |
| Auth | user ADC (`~/.config/gcloud/application_default_credentials.json`) |
| Spend to date | $186.44 |

## 2. Claims register

Every claim we make or want to make, with how it is reproduced and whether that
reproduction is currently push-button.

### Reproducible today (command given; needs the built tree)

| # | Claim | Reproduce | Status |
| --- | --- | --- | --- |
| C1 | The `chia-tinytpu-rtlgen` flow runs end to end: 8/8 oracle programs, direct-TOSA compiler checks, and CPU / Vitis-HLS / RTLGen backends all pass | `cd examples/accelerator/tinytpu && make oracle compiler cpu hls rtl` | ✅ |
| C2 | Vitis HLS C-synthesis of the composed design succeeds at 411 MHz, 8515 LUT / 9111 FF / 20 DSP / 26 BRAM18K on `xcu55c` | `make synth` | ✅ |
| C3 | Per-unit `(ii, depth)` derived from the csynth report reproduces the tool's measured call latency exactly for every statically-bounded unit | `pytest tests/dsa/test_tinytpu_synth.py` | ✅ (unit-tested against a recorded report) |
| C4 | The pre-existing frozen cost model was optimistic by 5.7× — 22,160 modeled vs 126,432 synthesis-grounded cycles | `python -m examples.accelerator.tinytpu.ppa --frozen` vs `ppa` | ✅ |
| C5 | `mxu` costs 72 cycles at II=2 (two sequential passes), not the declared 36 | `make synth`, read `latency_table.mxu` | ✅ |
| C6 | `dma_load` pays depth 75 — the real `m_axi` read latency — not the declared 8 | `make synth`, read `latency_table.dma_load` | ✅ |
| C7 | An agent-authored spec that rewrites files at import is refused, and the shipped spec satisfies the same policy | `pytest tests/dsa/test_tinytpu_agent_policy.py` | ✅ |
| C8 | One full evaluation (synthesize + re-measure + score) takes ~41 s | timed `make ppa` | ✅ |

### Reproducible, via replay

`verify_variant.py` rebuilds a recorded variant in a clean worktree, applies the
accepted diffs in order, re-synthesizes, and asserts the recorded cycle count.
Independently confirmed 2026-09-07: both replay bit-exact at zero tolerance.

| # | Claim | Reproduce | Status |
| --- | --- | --- | --- |
| C9 | An agent found **4.07×** (126,432 → 31,056), numerics exact | `verify_variant.py --run chia_runs/swarm-20260905-063857 --worker dram` | ✅ 42 s |
| C10 | A second angle found **1.98×** (63,864) independently | `... --worker granularity-retry` | ✅ 43 s |
| C11 | The 4× costs 4.7× BRAM (26 → 122) and +67% LUT at flat Fmax; per BRAM it is *worse* than baseline | printed by either replay | ✅ |
| C12 | Yield varies ~100× across hypotheses | `chia_runs/*/[worker]/variants.jsonl` | ⚠️ n=1 per angle; needs ≥3 seeds |
| C13 | Cost is ~$19/agent-hour | `chia_runs/opencode_sessions.db` | ✅ archived with the run |

The winning diffs are `chia_runs/<run>/swarm_best.diff` (best across workers) and
the `diff` field of each accepted entry in a worker's `variants.jsonl`.

### Observed once, not yet a defensible claim

| # | Observation | Why it is not yet a claim |
| --- | --- | --- |
| C14 | A live agent wrote self-modifying code into the spec, which the evaluator then imported and executed | Real and now guarded (C7), but the original artifact was reverted by `git checkout`. **Not captured.** Would need re-eliciting to demonstrate. |
| C15 | CHIA serializes all LLM calls on `opencode_creds`; one unit caps the whole cluster | Verified by observation (`4.0/4.0` after the fix). Not asserted by a test. |
| C16 | Vertex preview-model quota, not the harness, is the parallelism ceiling — 3 of 4 workers died on `RateLimitError` | Observed in worker logs; quota API does not expose the effective preview limit, so not independently confirmable. |

## 3. Maintenance gaps, by risk

1. **No remote backup** (§0). Everything else is downstream of this.
2. **Run artifacts partly in `/tmp`.** `chia_runs/` lives in the repo (good), but
   the swarm worktrees are at `/tmp/tinytpu_swarm_trees/worker-*` and are
   registered git worktrees. `/tmp` is wiped; this already destroyed the CHIA
   install and opencode CLI once this week (they were at `/tmp/chia` and
   `/tmp/chia-opencode-cli` per the original README). Stale worktree entries
   also accumulate: `git worktree list` currently shows two dead
   `/tmp/claude-*` paths.
3. **`~/allo-act` unrepresented.** 1 unpushed commit + 127 dirty files.
4. **Environments not captured.** Neither conda env is exported; rebuilding
   `allo` (3.12) and `chia_env` (3.10.19 + ray 2.54.0 + CHIA editable) is
   currently tribal knowledge.
5. **Cost telemetry is per-machine.** opencode's SQLite DB is the only record of
   spend and token usage; it is not in any backup.
6. **ADC is user-scoped and expires.** A service account would make unattended
   runs survivable.
7. **Billing is on the trial account.** Must be relinked to
   `0148AB-064407-7A5F3C` once the $300 is consumed, or awarded credit sits
   unused.
8. **No single bootstrap.** §4 is written out but not scripted.

## 4. Bootstrap from scratch (the three non-obvious fixes)

These are committed in `1f746005` but are the parts that break a naive rebuild:

```bash
# 1. lld must be reachable as `ld.lld` (Ubuntu ships `ld.lld-18`)
mkdir -p ~/.local/allo-bin && ln -sf /usr/lib/llvm-18/bin/ld.lld ~/.local/allo-bin/
export PATH=~/.local/allo-bin:$PATH

# 2. cmake 4 refuses OR-Tools 9.5's fetched deps; build-circt.sh exports the shim
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# 3. build Allo with gcc — clang's enable_if on llvm::StringLiteral breaks the
#    std::optional<StringLiteral> conversion the HLS emitters rely on
export SKBUILD_CMAKE_DEFINE="CMAKE_C_COMPILER=gcc;CMAKE_CXX_COMPILER=g++"
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PWD/externals/circt/ext" pip install -v -e .
```

Wall time on a 144-core host: LLVM ~12 min, CIRCT ~15 min, OR-Tools ~10 min,
Allo ~2 min.

## 5. Maintenance work

**Done**

- Pushed `chia-codesign` to `sunwookim028/allo`; removed the redundant
  `~/allo-act`. (§0)
- `verify_variant.py`: replays a recorded variant into a clean worktree,
  re-synthesizes, and asserts the recorded cycle count. Both headline results
  reproduce exactly in ~42 s each.
- `scripts/claims.sh`: runs the claims in three tiers (13 s / 133 s / 196 s),
  each step timing itself.
- `scripts/chia.env.example`: one home for the environment every script needs.
- The opencode session database is archived into `chia_runs/`, so cost telemetry
  is versioned with the results it explains.
- Reclaimed 21 GB (§0b).

**Outstanding, in priority order**

1. **Replicate each hypothesis ≥3×.** Every angle is n=1, which makes idea 4 —
   the project's most interesting claim — its least evidenced. This is the single
   highest-value use of further budget (~$300–600).
2. **Seed the search**, so a run is repeatable rather than merely re-runnable.
   Until then only the replay path is deterministic.
3. Fix `scripts/claims.sh`'s hardcoded `conda run -n allo`: a reader who follows
   the docs into a differently-named environment cannot run the claims.
4. `scripts/bootstrap_chia.sh` doing §4 end to end, plus `environment-*.yml`
   exports so the two conda environments are captured rather than tribal.
5. Move swarm worktrees off `/tmp` (they survive there only by luck; `/tmp` is a
   separate 15 GB filesystem and has already been wiped once this week, taking
   the CHIA install and opencode CLI with it). Have `verify_variant.py` prune its
   temporary worktree in a `finally`.
6. Deposit an archive with a DOI (Zenodo) for *Artifacts Available*.
7. Switch to a service account for unattended runs; user ADC expires.
