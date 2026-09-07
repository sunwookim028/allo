# CHIA co-design checkpoint — 2026-09-07

Maintenance record for the "Agentic Discovery of Domain-Specific Computing
Stack with CHIA" effort. Two purposes: pin what version of everything produced
the results, and track every claim we make (or want to make) against whether it
is push-button reproducible from a remote backup.

## 0. Backup status — resolved 2026-09-07

`chia-codesign` is pushed to `sunwookim028/allo`. Everything below traces to code
that exists on a remote.

`~/allo-act` has been removed. It was safe: its `HEAD` (`0b5fef7`) is reachable
from `chia-codesign` and present at tag `tinytpu-rtlgen-base`, its working
tree held only deletions, and its one untracked file (a `vadd_relu` fused
instruction on an older TinyTPU ancestor) was judged not worth keeping.

Still outstanding for an *Artifacts Available* badge: a permanent archive with a
DOI. A git branch alone does not satisfy it.

## 0b. Disk footprint

| Item | Size | Regenerable |
| --- | --- | --- |
| `~/allo` built tree (single checkout) | 9.2 GB | all but `chia_runs` |
| ↳ `chia_runs` | 85 MB | **no — the evidence behind every result** |
| conda `allo` / `chia_env` | 1.5 GB / 0.6 GB | yes |
| `~/chia-tools` (CHIA + opencode) | 736 MB | yes |
| peak during a fresh build | ~19 GB | — |

`~/allo-chia` was a git worktree of `~/allo`, not a second clone. Consolidated
2026-09-07: the build moved into `~/allo`, the worktree was dropped, and the
toolchain's hardcoded paths were rewritten (7 cmake exports, 2 CMakeCache.txt,
978 files under `build/`). `~/allo` is now the single checkout, on
`chia-codesign`. Switching it to `main` gives a different codebase (see §1) and
invalidates the built extension.

Reclaimed 2026-09-07: **21 GB** — 16 GB of `~/.cache` (HuggingFace model blobs,
pip, conda tarballs, pre-commit) plus 2 GB of `~/allo-act`, and 5 GB of conda and
pip caches earlier. HuggingFace auth tokens were preserved; only the
re-downloadable model blobs were dropped.

## 1. Component inventory

| Component | Location | Version / pin | Remote | Backed up |
| --- | --- | --- | --- | --- |
| Allo fork (integration HEAD) | `~/allo` | `e78bf9b5` on `main` | `sunwookim028/allo` | yes |
| Allo + ACT (Kai) | *removed 2026-09-07* | `0b5fef7` — reachable from `chia-codesign` and on `origin/chia-tinytpu-rtlgen` | `kkkaishao/allo` | yes, via this fork |
| Allo + ACT + RTLGen + CHIA | `~/allo` | `chia-codesign` | `sunwookim028/allo` | **yes — pushed** |
| ↳ commit it forks from | — | `882f7dd6`, tag `tinytpu-rtlgen-base` | `sunwookim028/allo` | yes |
| CHIA framework | `~/chia-tools/chia` | `16c35e9` | `ucb-bar/chia` | upstream only |
| opencode CLI | `~/chia-tools/opencode-cli` | `opencode-ai@1.18.25` | npm | pinned |

**Submodule pins** (`~/allo/externals/`), all built and present:

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
| conda env `allo` | Python 3.12.13 | editable install → `~/allo` |
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

## 2. Claims

[CODESIGN.md](../CODESIGN.md) is the canonical claim register: `C1`–`C8` for the
deterministic claims, each with the command that checks it, and `S1`–`S3` for
the stochastic ones a search cannot promise to repeat. Cite those identifiers.
This file deliberately does not restate them — two registers drift, and an
earlier draft of this one had already gone stale while `CODESIGN.md` was
correct.

What belongs here instead is the residue: things this effort observed once,
which are not claims because nothing reproduces them on demand.

| # | Observation | Why it is not a claim |
| --- | --- | --- |
| O1 | A live agent wrote self-modifying code into the spec, which the evaluator then imported and executed | Real, and now guarded (`C6`) — but the original artifact was reverted by `git checkout` and never captured. Demonstrating it again would mean re-eliciting it. |
| O2 | CHIA serializes every LLM call on `opencode_creds`; one unit caps the whole cluster | Verified by observation (`4.0/4.0` after raising it). No test asserts it. |
| O3 | Vertex preview-model quota, not the harness, is the parallelism ceiling — 3 of 4 workers died on `RateLimitError` before proposing anything | Seen in worker logs. The quota API does not expose the effective preview limit, so it cannot be confirmed independently. |

## 3. Bootstrap from scratch (the three non-obvious fixes)

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

## 4. Maintenance work

**Done**

- Pushed `chia-codesign` to `sunwookim028/allo`; removed the redundant
  `~/allo-act`. (§0)
- `verify_variant.py`: replays a recorded variant into a clean worktree,
  re-synthesizes, and asserts the recorded cycle count. Both headline results
  reproduce exactly in ~42 s each.
- `scripts/claims.sh`: runs the claims in three tiers (14 s / 141 s / 195 s on an
  idle 144-core host; they move with load),
  each step timing itself.
- `scripts/chia.env.example`: one home for the environment every script needs.
- The opencode session database is archived into `chia_runs/`, so cost telemetry
  is versioned with the results it explains.
- Reclaimed 21 GB (§0b).
- `scripts/claims.sh` takes the environment from `TINYTPU_ENV` and refuses to
  run when that environment imports `allo` from a different checkout, so a green
  result cannot come from someone else's tree.
- Worker scratch moved off `/tmp` to `~/.cache/tinytpu`, and `swarm.py` prunes
  stale registrations on entry and unregisters its worktrees on exit
  (`--keep-worktrees` opts out). `verify_variant.py` already removed its own in a
  `finally`.

**Outstanding, in priority order**

1. **Replicate each hypothesis ≥3×.** Every angle is n=1, which makes idea 4 —
   the project's most interesting claim — its least evidenced. This is the single
   highest-value use of further budget (~$300–600).
2. **Seed the search**, so a run is repeatable rather than merely re-runnable.
   Until then only the replay path is deterministic.
3. `scripts/bootstrap_chia.sh` doing §3 end to end, plus `environment-*.yml`
   exports so the two conda environments are captured rather than tribal.
4. Deposit an archive with a DOI (Zenodo) for *Artifacts Available*.
5. Switch to a service account for unattended runs; user ADC expires.
