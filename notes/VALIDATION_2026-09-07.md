# Independent build-and-reproduce validation — CHIA / TinyTPU co-design

**Tree under test:** `/home/sk3463/validate-chia`, detached `HEAD = 6d5380b6`
("Make every claim runnable, and give the environment one home"), the tip of
`origin/chia-tinytpu-synth-objective`.
**Env:** conda `allo_validate` (Python 3.12.14). **Host:** 144 cores, gcc 13.3.0,
clang 18.1.3, cmake 4.2.1, ninja 1.11.1, Vitis HLS 2023.2, `ld.lld` shimmed.
**Date:** 2026-09-07.

**Bottom line: every claim reproduces, exactly. The documentation does not.**
All 8 claim steps pass and both headline cycle counts replay bit-for-bit. But
three of the four documents I was told to treat as the specification are wrong,
missing, or silently point the reader at a different checkout on this machine.

---

## 0. The specification I was asked to test does not exist

| Named artifact | Status |
| --- | --- |
| `CODESIGN.md` | **Does not exist.** Not at `HEAD`, not in any commit on any ref. `git log --all --diff-filter=A -- '*CODESIGN*'` returns nothing; `find . -iname '*codesign*'` returns nothing. |
| branch `chia-codesign` | **Does not exist.** The only refs are `origin/main` and `origin/chia-tinytpu-synth-objective`. The clone is on a detached `HEAD` at the latter's tip. |
| `notes/CHIA_CHECKPOINT.md` | Exists. Stale against `HEAD` (see P6). |
| `scripts/chia.env.example` | Exists. Contains a path bug (see DEV-3). |
| `scripts/claims.sh` | Exists. Contains the env-name defect (see DEV-4). |

Because `CODESIGN.md` is absent, **the "project frontmatter" a newcomer is
supposed to start from does not exist in the repository.** Everything below
judges the documents that *are* there: `README.md`, `AGENTS.md`,
`ENVIRONMENT.md`, `notes/CHIA_CHECKPOINT.md`,
`examples/accelerator/tinytpu/README.md`, and
`examples/accelerator/tinytpu/chia_agent/README.md`.

The practical consequence is severe and is the theme of this report: **the four
artifacts that actually make the claims runnable — `scripts/claims.sh`,
`chia.env`, `verify_variant.py`, `chia_agent/smoke.py` — are mentioned by no
markdown file in the repository.** Verified:

```
$ grep -rn "claims.sh\|verify_variant\|smoke" --include=*.md . | grep -v externals
notes/CHIA_CHECKPOINT.md:90:  ... Needs a `verify_variant.py`.        # says it does NOT exist
notes/CHIA_CHECKPOINT.md:156: 4. Add `verify_variant.py`: ...         # proposes adding it
```

Nothing else. A reader following only the docs would never discover the entry
point, would never learn that the three tiers exist, and would have no expected
step counts or wall times to compare against — those numbers (13 s / 133 s /
196 s, 3 / 6 / 8 steps) live **only in the commit message of `6d5380b6`**.

---

## a. Deviations from the documentation

Six deviations. Two were forced by contradictions in the docs themselves; four
were forced by real defects.

### DEV-1 — Compiler choice for CIRCT: two docs disagree, I picked one

`README.md` (lines 63-72) builds CIRCT with clang:

```bash
cmake -G Ninja ../ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ...
```

`examples/accelerator/tinytpu/README.md` §"Clean build" — the only from-scratch
recipe specific to the work under test — uses gcc:

```bash
bash scripts/build-circt.sh externals/circt externals/llvm-project/build Release gcc g++
```

`scripts/build-circt.sh`'s own defaults are `clang`/`clang++`. I started the
clang path, then killed it and rebuilt with gcc to match the tinytpu recipe and
the compiler the inherited LLVM was built with. **Recorded as a deviation
because the docs do not agree on which is correct.** No error was produced —
this is an ambiguity, not a failure.

### DEV-2 — `README.md`'s manual build produces no OR-Tools, so Allo cannot be installed

Following `README.md` literally, the CIRCT block is a bare `cmake ../ && ninja`.
It never runs `externals/circt/utils/get-or-tools.sh`. The very next paragraph
then says:

> `CMAKE_PREFIX_PATH` points at the OR-Tools install that `build-circt.sh`
> places in `externals/circt/ext`; RTL scheduling does not build without it.

— referring to a script the reader was never told to run. The README's own
instructions therefore leave `externals/circt/ext` non-existent and the
subsequent `pip install -e .` fails with *"No OR-Tools cmake package was found"*
(the failure mode `scripts/build-circt.sh`'s own comment predicts).

**Deviation taken:** used `scripts/build-circt.sh`, which does run
`get-or-tools.sh`. Command actually run:

```bash
bash scripts/build-circt.sh externals/circt externals/llvm-project/build Release gcc g++
```

### DEV-3 — `chia.env` computes `ALLO_ROOT` as the repo's *parent*

`scripts/chia.env.example` says, in its first line, "Copy to `chia.env` at the
repo root", then computes:

```bash
export ALLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

That `/..` is correct for a file living in `scripts/`, and wrong for a file at
the repo root. Demonstrated verbatim, before any edit of mine:

```
$ cd /home/sk3463/validate-chia && bash -c 'source chia.env; echo "ALLO_ROOT=$ALLO_ROOT"; echo "PYTHONPATH=$PYTHONPATH"'
ALLO_ROOT=/home/sk3463
PYTHONPATH=/home/sk3463
SKBUILD_EDITABLE_SKIP=/home/sk3463/build
```

So sourcing the documented environment file puts `/home/sk3463` — a directory
containing three *other* Allo checkouts (`allo`, `allo-act`, `allo-chia`) — on
`PYTHONPATH`, and points `SKBUILD_EDITABLE_SKIP` at a path that does not exist.
This is latent rather than fatal only because `claims.sh` happens to `cd` to the
repo root first, so `''` precedes `/home/sk3463` on `sys.path`. Confirmed
present in a real run: `sys.path[:2] == ['', '/home/sk3463']`.

**Deviation taken:** changed my `chia.env` to
`export ALLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`, which yields
`/home/sk3463/validate-chia`.

### DEV-4 — the hardcoded conda env name, and why it is worse than a wrong name

Task hypothesis confirmed, and then some. `scripts/claims.sh:30` reads:

```bash
allo () { conda run -n allo "$@"; }
```

`chia.env.example` advertises itself as "the single place environment lives" but
carries **no variable for the environment name at all**. The name is hardcoded
in four places:

| File:line | Code |
| --- | --- |
| `scripts/claims.sh:30` | `allo () { conda run -n allo "$@"; }` |
| `scripts/claims.sh:62` | `conda run -n chia_env python .../smoke.py` |
| `examples/accelerator/tinytpu/verify_variant.py:75` | `"conda","run","-n","allo",` |
| `examples/accelerator/tinytpu/chia_agent/allo_tool.py:208` | `[self.conda_exe,"run","-n","allo",...]` |

plus `examples/accelerator/tinytpu/Makefile:3` (`ENV ?= allo`, overridable — but
`claims.sh:59` invokes `make` **without** passing `ENV`, so the override is
unreachable from the claim runner).

**This is not merely a wrong name. On this machine it fails silently and
produces a green result from a different repository.** Running the unmodified
script in the `allo_validate` env:

```
$ cd /home/sk3463/validate-chia && ./scripts/claims.sh --fast
  C2.3   derived (ii,depth) reproduces measured latency       PASS    4s
  C2.1   frozen cost model scores 22,160 cycles               PASS    8s
  C3.1   self-modifying spec refused; real spec accepted      PASS    3s
  3 passed, 0 failed, 15s total (tier: --fast)
```

Three green PASSes — from code I had not built. Proof:

```
$ source chia.env && conda run -n allo python -c "import allo; print(allo.__file__)"
allo.__file__ = /home/sk3463/allo-chia/allo/__init__.py
```

The `allo` env's editable install redirects `import allo` (via a
scikit-build-core meta-path finder, which outranks `sys.path`) to the **original
author's working tree** `~/allo-chia`. `make -n oracle` shows the same:

```
cd /home/sk3463/validate-chia && ... conda run -n allo python -m examples.accelerator.tinytpu.oracle
```

A validator who did not check `allo.__file__` would have reported a successful
reproduction while never once executing the tree they built. **This is the most
serious defect found.**

**Deviation taken:** made the name configurable (the minimal fix I would ship)
and set `TINYTPU_ENV=allo_validate` in `chia.env`:

```diff
-allo () { conda run -n allo "$@"; }
+allo () { conda run -n "${TINYTPU_ENV:-allo}" "$@"; }
-    make -C examples/accelerator/tinytpu oracle compiler cpu hls rtl
+    make -C examples/accelerator/tinytpu ENV="${TINYTPU_ENV:-allo}" oracle compiler cpu hls rtl
-    conda run -n chia_env python examples/.../smoke.py
+    conda run -n "${TINYTPU_CHIA_ENV:-chia_env}" python examples/.../smoke.py
```
```diff
# verify_variant.py:75
-        "allo",
+        __import__("os").environ.get("TINYTPU_ENV", "allo"),
# chia_agent/allo_tool.py:208
-        command = [self.conda_exe, "run", "-n", "allo", ...]
+        command = [self.conda_exe, "run", "-n",
+                   os.environ.get("TINYTPU_ENV", "allo"), ...]
```

### DEV-5 — the documented install does not install `pytest`, and two of three `--fast` steps are pytest

`examples/accelerator/tinytpu/README.md` §"Clean build" ends at
`CMAKE_ARGS=... pip install -v -e .`. That is the complete from-scratch recipe
and it installs no test runner. With that exact state, against the correct env:

```
$ ./scripts/claims.sh --fast
  C2.3   derived (ii,depth) reproduces measured latency       FAIL    4s
  C2.1   frozen cost model scores 22,160 cycles               PASS    7s
  C3.1   self-modifying spec refused; real spec accepted      FAIL    3s
  1 passed, 2 failed, 14s total (tier: --fast)

$ cat /tmp/claims_C2.3.log
/home/sk3463/miniconda3/envs/allo_validate/bin/python: No module named pytest
ERROR conda.cli.main_run:execute(148): `conda run python -m pytest tests/dsa/test_tinytpu_synth.py -q` failed.
```

**Deviation taken:** ran the root `README.md`'s separate developer step, which
the tinytpu clean-build recipe never cross-references:

```bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PWD/externals/circt/ext" pip install -e .[dev]   # 50 s, rc=0
```

### DEV-6 — `--full` depends on out-of-repo prerequisites that no build doc mentions

`claims.sh --full` step C5.0 requires the `chia_env` conda env, the CHIA
framework, and the opencode CLI. None of that appears in `README.md`,
`notes/CHIA_CHECKPOINT.md` §4 ("Bootstrap from scratch"), or the tinytpu clean
build. It is documented only in `chia_agent/README.md`, with **absolute paths
hardcoded to this user's home**:

```bash
git clone https://github.com/ucb-bar/chia.git /home/sk3463/chia-tools/chia
conda create -n chia_env python=3.10.19
npm install --prefix /home/sk3463/chia-tools/opencode-cli opencode-ai@1.18.25
```

**Deviation taken:** I used the pre-existing `chia_env` and
`~/chia-tools/opencode-cli` (read-only), rather than creating my own. I did not
re-derive that they can be built from scratch. `claims.sh` performs no
precondition check and would have reported C5.0 as a plain `FAIL` had they been
absent.

### Non-deviations, for the record

- `~/.local/allo-bin/ld.lld` (CHIA_CHECKPOINT §4 step 1) already existed on this
  machine; the documented `ln -sf` would have been a no-op.
- CHIA_CHECKPOINT §4 step 3 (`SKBUILD_CMAKE_DEFINE=...gcc`) is now redundant —
  `pyproject.toml` sets `CMAKE_C_COMPILER=gcc` / `CMAKE_CXX_COMPILER=g++`
  itself. Harmless, but stale.
- `git worktree`, `patch`, ADC, and Vitis all behaved as documented.

---

## b. Results per tier

All runs from `/home/sk3463/validate-chia` after DEV-1…DEV-5 were applied.
"Reported" is `claims.sh`'s own printed total; "wall" is `time(1)`.

| Tier | Steps | Result | Reported | `time` real | Expected | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `--fast` | 3 | **3 passed, 0 failed** | **14 s** | 0m14.225s | ~13 s | +1 s |
| (default) | 6 | **6 passed, 0 failed** | **145 s** | 2m25.473s | ~133 s | +12 s (+9 %) |
| `--full` | 8 | **8 passed, 0 failed** | **220 s** | 3m39.809s | ~196 s | +24 s (+12 %) |

Per-step times (`--full` run):

```
  C2.3   derived (ii,depth) reproduces measured latency       PASS    4s
  C2.1   frozen cost model scores 22,160 cycles               PASS    7s
  C3.1   self-modifying spec refused; real spec accepted      PASS    4s
  C2.2   synthesis measures mxu=72 II=2, dma_load depth 75    PASS   40s
  C4.1   replay the 4.07x variant and re-derive its score     PASS   45s
  C4.2   replay the 1.98x variant from another hypothesis     PASS   51s
  C1.1   same schedule lowers to CPU, Vitis HLS, and RTL      PASS   32s
  C5.0   CHIA round-trip: ADC -> Vertex -> opencode -> MCP -> allo  PASS  37s
```

The 9–12 % overrun on the two synthesis tiers is Vitis HLS variance, not a
regression; the tier totals are dominated by three ~40 s csynth invocations.

### Headline numbers — both reproduce EXACTLY

```
=== C4.1 (dram) ===
replaying 2 accepted candidate(s) from swarm-20260905-063857/dram
  recorded : 31,056 cycles
  replayed : 31,056 cycles
  baseline : 126,432 cycles  ->  4.07x
  area     : LUT 14244 FF 12253 DSP 25 BRAM18K 122  Fmax 411.02 MHz
  numerics : pass  (41s)
VERIFIED

=== C4.2 (granularity-retry) ===
replaying 1 accepted candidate(s) from swarm-20260905-063857/granularity-retry
  recorded : 63,864 cycles
  replayed : 63,864 cycles
  baseline : 126,432 cycles  ->  1.98x
  area     : LUT 9701 FF 9593 DSP 20 BRAM18K 26  Fmax 367.24 MHz
  numerics : pass  (48s)
VERIFIED
```

Both replayed twice (default and `--full` tiers), identical both times, with
`--tolerance 0.0` (the default — a bit-exact comparison, not a fuzzy one).
Area and Fmax also match the recorded `variants.jsonl` fields exactly.

### Every other reproduced number matched the record

| Claim | Recorded | Reproduced | Match |
| --- | --- | --- | --- |
| C4 / C2.1 frozen cost model | 22,160 cycles | 22,160 | exact |
| C9 / C4.1 dram variant | 31,056 cycles, 4.07× | 31,056, 4.07× | exact |
| C10 / C4.2 granularity-retry | 63,864 cycles, 1.98× | 63,864, 1.98× | exact |
| baseline | 126,432 cycles | 126,432 | exact |
| C2 csynth QoR | 411 MHz, 8515 LUT / 9111 FF / 20 DSP / 26 BRAM18K, `xcu55c-fsvh2892-2L-e` | 411.02 MHz, 8515 / 9111 / 20 / 26, same part | exact |
| C5 `mxu` | 72 cycles at II=2 (declared 36) | `call_cycles: 72, ii: 2` | exact |
| C6 `dma_load` | depth 75 (declared 8) | `depth: 75` | exact |
| C11 Pareto crossover | 4.7× BRAM (26 → 122), +67 % LUT (8515 → 14244), unchanged Fmax | 26 → 122 BRAM, 8515 → 14244 LUT (+67.3 %), 411.02 MHz both | exact |
| C1 backends | 8/8 oracle, CPU/HLS/RTL | 8/8 + SystemVerilog emitted for all 6 units | pass |
| C13 cost | ~$0.03 for the smoke round-trip | one billed Vertex round-trip, 31 s, correct tool answer | consistent (not independently priced) |

**No reproduced number differed from the recorded one.** Not one.

C5.0 output confirms the whole agent path is live, not stubbed:
`model replied: dma_load, dma_store, vload, vstore, vpu, mxu` →
`SMOKE OK (31s): ADC -> Vertex -> opencode -> CHIA -> MCP tool -> allo env`.

---

## c. Disk and build wall time

**Disk — fresh tree total: 19 GB** (`du -sh /home/sk3463/validate-chia`).

| Component | Size |
| --- | --- |
| `externals/` (llvm-project src+build, circt src+build, or-tools install, marl, past) | 11 GB |
| `allo/` (Python + generated) | 143 MB |
| `build/` (Allo native) | 104 MB |
| repo source, `chia_runs/`, tests, docs | ~180 MB |
| `.git` + submodule git dirs | balance (~7.5 GB) |

Free space never dropped below the stop threshold; it ended at 119 GB free.

**Build wall time — 641 s (10.7 min) total**, of which 317 s inherited:

| Stage | Wall | Source |
| --- | --- | --- |
| submodule checkout (~5.1 GB) | **unknown** | inherited; only its completion timestamp survives |
| LLVM/MLIR (`build-mlir.sh … gcc g++ --fresh`) | **317 s** (5.3 min) | inherited, timestamped `_t_mlir_start`/`_t_mlir_end` |
| CIRCT + OR-Tools 9.5 (`build-circt.sh … gcc g++`) | **200 s** (3.3 min) | mine |
| Allo editable (`pip install -v -e .`) | **74 s** | mine |
| dev extras (`pip install -e .[dev]`) | **50 s** | mine (DEV-5) |
| **my share** | **324 s** (5.4 min) | |

Against `notes/CHIA_CHECKPOINT.md` §4's "Wall time on a 144-core host: LLVM ~12
min, CIRCT ~15 min, OR-Tools ~10 min, Allo ~2 min" (≈39 min), the measured
figure on a 144-core host is **10.7 min — 3.6× faster than documented**. The
doc's estimates are pessimistic by a wide margin; CIRCT+OR-Tools took 200 s
against a claimed 25 min. Not a defect, but the numbers are not trustworthy as
planning figures.

---

## d. Prioritized defects, each with a minimal fix

**P0 — silent wrong-environment execution (DEV-4).** The claim runner can pass
green while executing a different checkout. This defeats the entire purpose of
`claims.sh`.
*Fix:* add `export TINYTPU_ENV=allo` to `scripts/chia.env.example`; replace the
four hardcoded `"allo"` literals with `${TINYTPU_ENV:-allo}` /
`os.environ.get("TINYTPU_ENV", "allo")`; pass `ENV="${TINYTPU_ENV:-allo}"` to
`make` at `claims.sh:59`. Then add one guard at the top of `claims.sh`:
`allo python -c "import allo,sys; sys.exit(0 if allo.__file__.startswith('$ROOT') else 1)"` —
abort loudly if the env's `allo` is not this tree.

**P1 — `CODESIGN.md` is missing (§0).** The named onboarding document does not
exist in any commit.
*Fix:* write it, or if it was meant to be one of the existing files, say which.
At minimum it must name `scripts/claims.sh`, the three tiers with their step
counts and expected times, the `chia.env` copy step, and the two headline
numbers a reader should see.

**P2 — nothing in any `.md` mentions `claims.sh`, `chia.env`,
`verify_variant.py`, or `smoke.py` (§0).** The reproduction entry point is
undiscoverable from the documentation.
*Fix:* add a "Reproduce the claims" section to `README.md` (or `CODESIGN.md`):
```bash
cp scripts/chia.env.example chia.env   # then set GOOGLE_CLOUD_PROJECT
./scripts/claims.sh --fast   # 3 steps, ~13 s, no synthesis, no API
./scripts/claims.sh          # 6 steps, ~133 s, adds Vitis HLS
./scripts/claims.sh --full   # 8 steps, ~196 s, ~$0.03 billed
```

**P3 — `README.md`'s manual build cannot succeed (DEV-2).** No OR-Tools step;
Allo install then fails.
*Fix:* replace the two hand-written cmake blocks with the two script
invocations from `examples/accelerator/tinytpu/README.md` §"Clean build", which
are correct and already exist. This also resolves the clang/gcc contradiction
(DEV-1) in one edit.

**P4 — `chia.env.example`'s `ALLO_ROOT` resolves to the repo's parent (DEV-3).**
*Fix:* `export ALLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`
(drop the `/..`), matching the file's own "copy to the repo root" instruction.

**P5 — the from-scratch recipe omits the test runner (DEV-5).** Two of three
`--fast` steps cannot run after the documented build.
*Fix:* change the last line of the tinytpu "Clean build" block to
`pip install -e .[dev]`, or add `pytest` to the base `dependencies` in
`pyproject.toml`.

**P6 — `notes/CHIA_CHECKPOINT.md` is stale against `HEAD`.**
Its §1 inventory pins the CHIA checkout at `56280ea1` while `HEAD` is `6d5380b6`
(two commits later); C9/C10 state the replay "is manual. Needs a
`verify_variant.py`" and §5 item 4 proposes writing one — but
`verify_variant.py` ships at `HEAD` and is exactly what makes C9/C10
push-button. A reader trusting §2 would conclude the headline claims are *not*
reproducible and stop.
*Fix:* move C9/C10/C11 from "Reproducible in principle" to "Reproducible
today" with the `verify_variant.py` commands; strike §5 item 4; re-pin §1 to
`6d5380b6`.

**P7 — broken evidence path in the claims register.** C9 cites
`chia_runs/swarm-20260905-063857/dram/best.diff`; that file does not exist (the
directory holds only `variants.jsonl`; the diff is
`chia_runs/swarm-20260905-063857/swarm_best.diff` at the run root).
`chia_agent/README.md` repeats the wrong shape ("the winning diff lands in
`<log-dir>/best.diff`").
*Fix:* cite `swarm_best.diff`, or have `swarm.py` write per-worker `best.diff`.

**P8 — `AGENTS.md` names a conda env that does not exist here.** Its very first
line is `Always run conda activate allo-rtlgen before building`, while
`README.md`, the tinytpu README, the checkpoint, and every script use `allo`.
Since `README.md` tells agents to import `AGENTS.md`, this is the first
instruction a coding agent reads, and it is wrong.
*Fix:* one-word change to `allo`, or to `${TINYTPU_ENV}` once P0 lands.

**P9 — `README.md` points at the wrong remote.** It says
`git clone https://github.com/kkkaishao/allo.git`; the checkpoint records this
work as living on `sunwookim028/allo`, and `kkkaishao/allo` carries none of the
CHIA/TinyTPU commits. A reader following the README clones a repo where nothing
in this report exists.
*Fix:* correct the URL and name the branch (`chia-tinytpu-synth-objective`).

**P10 — `ENVIRONMENT.md` contradicts every working script.** It says "**Always
use Docker** for Vitis commands" and gives host paths under
`/tools/Xilinx/Vitis/...`, while the actual, working path used by
`chia.env.example`, `claims.sh`, and both tinytpu READMEs is a direct host
source of `/opt/xilinx/Vitis_HLS/2023.2/settings64.sh`.
*Fix:* mark the Docker route as the fallback for incompatible hosts and state
the direct-source path first.

**P11 — `--full`'s out-of-repo prerequisites are undocumented in the build
path and hardcode `/home/sk3463` (DEV-6).**
*Fix:* add the three `chia_agent/README.md` prerequisite commands to the
bootstrap section with `${CHIA_TOOLS:-$HOME/chia-tools}`, and have `claims.sh`
precondition-check `conda env list | grep chia_env` and `command -v opencode`
with a clear skip message rather than a bare `FAIL`.

**P12 — `claims.sh` writes state into `/tmp`** (`/tmp/claims_*.log`,
`/tmp/claims_synth`), the exact practice `CHIA_CHECKPOINT.md` §3 item 2 flags as
having already destroyed artifacts once.
*Fix:* default to `${CLAIMS_OUT:-chia_runs/claims-$(date +%s)}`.

**P13 (minor) — checkpoint §4's wall-time table is off by ~3.6×** (§c above),
and step 3 (`SKBUILD_CMAKE_DEFINE=...`) is redundant with `pyproject.toml`.
*Fix:* re-measure, and delete step 3.

---

## e. Verdict

**Could a competent engineer with this repo, this machine, and no other context
reproduce these results? — Yes on the science, no on the documentation.**

The engineering underneath is genuinely reproducible, and better than most
work of this kind. `verify_variant.py` replays a recorded agent result by
applying its diffs into a throwaway git worktree, re-running real Vitis HLS
C-synthesis, re-deriving the latency table from the report, re-checking numerics
against NumPy, and comparing to the recorded score at **zero tolerance**. Both
headline numbers came back bit-exact, twice, on a tree I built myself, along
with every secondary number in the claims register. The design that scoring
cannot be gamed — the latency table is overwritten from synthesis before the
objective is computed — is real and holds up. There is no gap between what is
claimed and what the artifacts do.

**But the documentation would not get that engineer there.** Concretely, an
engineer with no other context would:

1. look for `CODESIGN.md` — the named starting point — and not find it;
2. follow `README.md`, clone the wrong GitHub repo (P9), build without OR-Tools
   (P3), and fail at `pip install -e .`;
3. read `AGENTS.md` first (as `README.md` instructs) and activate a conda env
   that does not exist (P8);
4. never learn that `scripts/claims.sh` exists at all (P2) — no markdown file
   names it — and so never run the claims;
5. reading `notes/CHIA_CHECKPOINT.md` §2 instead, conclude the two headline
   claims are *not* push-button and that a `verify_variant.py` still needs
   writing (P6) — when it ships in the very commit they have checked out;
6. and if they *did* stumble onto `claims.sh`, get three green PASSes on this
   machine that were produced by `~/allo-chia`, not by their build (P0).

Item 6 is the one that should worry the authors most. Every other defect fails
loudly. That one fails green.

**Caveats on my own result.** My run is not a clean-room reproduction:
`~/.local/allo-bin`, the `chia_env` conda env, `~/chia-tools/{chia,opencode-cli}`,
Vitis 2023.2, and Google ADC were all pre-existing on this host. I verified the
CHIA/opencode prerequisites *exist* and work; I did not verify they can be
installed from the docs. The LLVM/MLIR build was inherited from a prior attempt
(its 317 s is measured, its submodule-fetch time is unrecoverable). And I had to
apply six deviations before the claim runner would exercise the tree I built.

**Verdict: reproducible with caveats — but only by an engineer willing to read
the source when the docs run out, which is precisely the audience these docs are
supposed to spare.** Fix P0 through P3 and the answer becomes an unqualified
yes; P0 alone is the difference between a validation and the appearance of one.

---

### Appendix — files modified in the validation tree

`git diff` in `/home/sk3463/validate-chia` (workarounds only, all from DEV-4):

- `scripts/claims.sh` — env name via `${TINYTPU_ENV:-allo}` /
  `${TINYTPU_CHIA_ENV:-chia_env}`; `ENV=` passed to `make`
- `examples/accelerator/tinytpu/verify_variant.py:75` — env name from
  `TINYTPU_ENV`
- `examples/accelerator/tinytpu/chia_agent/allo_tool.py:208` — env name from
  `TINYTPU_ENV`
- `chia.env` (git-ignored) — `ALLO_ROOT` fix, `GOOGLE_CLOUD_PROJECT=test-adrs`,
  `TINYTPU_ENV=allo_validate`

Untracked helper scripts and timestamp files (`_build_*.sh`, `_build_*.log`,
`_t_*`, `_bin/`) are validation scaffolding, not repo changes.

`/home/sk3463/allo` (clean, `e78bf9b5`), `/home/sk3463/allo-chia`, and
`/home/sk3463/allo-act` were not modified; verified with `git status` and
`find -newermt`.
