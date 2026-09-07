# Agentic discovery of a domain-specific computing stack

An AI agent proposes changes to a TPU-style accelerator's **instruction set and
microarchitecture at once**, and every proposal is scored by running a real
commercial synthesis tool — not an estimate. One specification generates the
simulator, the hardware, and the compiler backend, so a single agent edit moves
all three together.

| | |
| --- | --- |
| Best agent result | **4.07×** fewer cycles (126,432 → 31,056), numerics exact |
| Second, independent | **1.98×** at **zero** extra BRAM |
| Cost model correction | prior hand-written model was **5.7× optimistic** |
| One evaluation | **41 s**, $0 — synthesis runs locally |
| One agent candidate | ~8–15 min, ~$2–3 |
| Verify both results | **~85 s**, two commands |

```bash
./scripts/claims.sh --fast     # 13s, no synthesis, no API
./scripts/claims.sh            # 133s, adds Vitis HLS
./scripts/claims.sh --full     # 196s, adds backends + one $0.03 agent round-trip
```

---

## The five ideas

1. **One specification, three artifacts.** The spec generates simulator,
   microarchitecture, and compiler backend. The agent edits one file and all
   three move — which is how a single candidate changed the ISA, the compiler's
   lowering, and the datapath together.
2. **The verifier must be the real tool.** The pre-existing hand-written cost
   model was optimistic by 5.7×: `mxu` costs 72 cycles, not 36 (it synthesizes
   as two sequential passes, II=2), and each DRAM access pays the real 75-cycle
   `m_axi` latency, not an assumed 8. Every decision taken against that model was
   taken against a fiction. Cheap proxies do not merely add noise — they point
   the wrong way.
3. **A cycles-only objective ranks designs backwards.** The 4.07× variant costs
   **4.7× the BRAM**, so per BRAM it is *worse* than the design it replaced. The
   1.98× variant costs none. Scored on cycles alone, the loop prefers the one
   that consumes the machine.
4. **Breadth beats depth.** Yield across hypotheses spanned ~100× (4.07× vs
   3.2%). Another hypothesis is worth far more than another iteration on the
   incumbent — but see *Not yet reproducible* before treating this as measured.
5. **Evaluation is cheap relative to proposal.** 41 s and $0 to evaluate; 8–60
   min and $2–3 to propose. That inverts the usual design-space-exploration cost
   model, where simulation dominates and effort goes into surrogates and caches.
   Here, caching evaluation would be optimizing the 1% side.

## Claims

Two tiers, because they are not equally strong and an artifact that blurs them
cannot be trusted.

### Deterministic — anyone can re-derive these

Reproduced by replay and re-synthesis; identical numbers every run.

| # | Claim | Command | Time |
| --- | --- | --- | --- |
| C1 | One schedule lowers to CPU, Vitis HLS, and CIRCT RTL | `make -C examples/accelerator/tinytpu oracle compiler cpu hls rtl` | 29 s |
| C2 | Frozen cost model scores 22,160 cycles; synthesis-grounded scores 126,432 (**5.7×**) | `./scripts/claims.sh --fast` | 7 s |
| C3 | Synthesis measures `mxu` = 72 cyc at II=2; `dma_load` depth = 75 | `python -m examples.accelerator.tinytpu.synth` | 35 s |
| C4 | Derived `(ii, depth)` reproduces the tool's measured latency exactly | `pytest tests/dsa/test_tinytpu_synth.py` | 4 s |
| C5 | A self-modifying spec is refused; the shipped spec passes the same policy | `pytest tests/dsa/test_tinytpu_agent_policy.py` | 3 s |
| C6 | The **4.07×** variant replays to 31,056 cycles, numerics exact | `verify_variant.py --run <run> --worker dram` | 42 s |
| C7 | The **1.98×** variant replays to 63,864 cycles at baseline BRAM | `verify_variant.py --run <run> --worker granularity-retry` | 43 s |
| C8 | Agent path is live end to end (ADC → Vertex → opencode → CHIA → MCP → allo) | `python chia_agent/smoke.py` | 32 s, $0.03 |

### Stochastic — a search, not a function

An agent search is not deterministic. Re-running it will not necessarily
rediscover the same design, and no honest artifact claims otherwise. What C6 and
C7 establish is that the designs the search *did* find are real and re-derivable
from the recorded trace. What is **not** yet established is the rate at which a
search finds them.

| # | Claim | Status |
| --- | --- | --- |
| S1 | A 4-worker × 5-iteration search finds ≥2× with probability *p* | **n=1.** Needs ≥3 seeds per hypothesis. |
| S2 | Yield varies ~100× across hypotheses (idea 4) | **n=1 per angle.** Suggestive, not measured. |
| S3 | Accept rate, and cycles-vs-iteration trajectory | **Not yet analysed.** Raw logs in `chia_runs/`. |

## Setup from zero

```bash
# 1. system packages (Ubuntu 24.04)
sudo apt install -y build-essential cmake ninja-build gcc g++ lld nodejs npm
mkdir -p ~/.local/allo-bin && ln -sf /usr/lib/llvm-18/bin/ld.lld ~/.local/allo-bin/
export PATH=~/.local/allo-bin:$PATH        # cmake wants `ld.lld`; Ubuntu ships `ld.lld-18`

# 2. Vitis HLS 2023.2 at /opt/xilinx/Vitis_HLS/2023.2 (C-synthesis needs no licence)

# 3. source and build  (~40 min on 144 cores, ~35 GB)
git clone git@github.com:sunwookim028/allo.git ~/allo
cd ~/allo && git checkout chia-codesign && git submodule update --init --recursive
conda create -y -n allo python=3.12 && conda activate allo
pip install "nanobind>=2.10,<3" "PyYAML<=6.0.1" typing_extensions \
            "scikit-build-core>=0.10" "setuptools_scm>=8" pytest numpy ml_dtypes rich sympy
export CMAKE_POLICY_VERSION_MINIMUM=3.5    # cmake 4 rejects OR-Tools 9.5's fetched deps
bash scripts/build-mlir.sh  externals/llvm-project Release gcc g++
bash scripts/build-circt.sh externals/circt externals/llvm-project/build Release gcc g++
SKBUILD_CMAKE_DEFINE="CMAKE_C_COMPILER=gcc;CMAKE_CXX_COMPILER=g++" \
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PWD/externals/circt/ext" pip install -v -e .

# 4. CHIA host (only needed to *run* agents; not needed to verify claims)
git clone https://github.com/ucb-bar/chia ~/chia-tools/chia
conda create -y -n chia_env python=3.10.19
conda run -n chia_env pip install -e ~/chia-tools/chia
npm install --prefix ~/chia-tools/opencode-cli opencode-ai@1.18.25

# 5. Google Cloud (only needed to run agents)
gcloud auth login && gcloud config set project <PROJECT_ID>
gcloud services enable aiplatform.googleapis.com
gcloud beta billing projects link <PROJECT_ID> --billing-account=<XXXXXX-XXXXXX-XXXXXX>
gcloud auth application-default login
gcloud auth application-default set-quota-project <PROJECT_ID>

# 6. environment, in one file
cp scripts/chia.env.example chia.env && $EDITOR chia.env
#    set GOOGLE_CLOUD_PROJECT, and TINYTPU_ENV if you named the conda env
#    something other than `allo`

# 7. check
./scripts/claims.sh --fast
```

`claims.sh` refuses to run if the conda environment imports `allo` from a
different checkout than the one you are standing in. An editable install
resolves through a meta-path finder that outranks `PYTHONPATH`, so without that
check every claim passes while exercising someone else's tree — a failure that
looks green. Set `TINYTPU_ENV` to the environment you built *this* checkout into.

**Build with gcc, not clang.** Under clang, the `enable_if` attribute on
`llvm::StringLiteral` makes it non-convertible for `std::optional`'s converting
constructor, so the HLS emitters fail to compile. `pyproject.toml` selects gcc.

### Budgets

| | |
| --- | --- |
| Build from scratch | 11–40 min, mostly LLVM; varies with machine load |
| Disk, built tree | **9.2 GB** — see the breakdown below |
| Disk, peak during build | ~19 GB (a fresh clone before `ninja` prunes intermediates) |
| Disk, conda envs | 1.5 GB (`allo`) + 0.6 GB (`chia_env`) + 0.7 GB (CHIA + opencode) |
| Verify all deterministic claims | 196 s, $0.03 |
| One agent candidate | ~8–15 min, ~$2–3 |
| One 4-worker × 5-iteration search | ~3 h, ~$100–190 |
| Rate | **$19 / agent-hour** (Vertex, `gemini-3.1-pro-preview`) |

Where the disk goes, once built:

| Path | Size | Regenerable? |
| --- | --- | --- |
| `externals/llvm-project/build` | 3.9 GB | yes — `scripts/build-mlir.sh` |
| `externals/circt/build` | 1.3 GB | yes — `scripts/build-circt.sh` |
| `externals/circt/ext` (OR-Tools) | 951 MB | yes — fetched by `build-circt.sh` |
| source + submodule checkouts | 3.1 GB | yes — `git submodule update` |
| `build` (allo extension) | 104 MB | yes — `pip install -e .` |
| `chia_runs` | **85 MB** | **no** — the evidence behind every result |

Everything except `chia_runs` is reproducible from the repository, so a machine
short on space can delete the build trees and rebuild in ~40 minutes. `chia_runs`
is the one directory that cannot be regenerated: it holds each candidate's diff,
synthesis report, and cost telemetry, and it is what `verify_variant.py` replays.

Preview-model **quota**, not credit, is the concurrency ceiling: an early
4-worker run lost 3 workers to `RateLimitError` before any proposed a candidate.

## Repository map

| Path | What it is |
| --- | --- |
| `examples/accelerator/tinytpu/isa.py` | ISA semantics — memories, instruction access/compute. **Agent-writable.** |
| `examples/accelerator/tinytpu/microarch.py` | Synthesizable units + fetch-decode-dispatch top + schedules. **Agent-writable.** |
| `examples/accelerator/tinytpu/synth.py` | Runs Vitis C-synthesis; rebuilds the per-unit latency table from the report |
| `examples/accelerator/tinytpu/ppa.py` | The objective: synthesize, re-measure, score cycles |
| `examples/accelerator/tinytpu/verify_variant.py` | Replays a recorded variant and re-derives its score |
| `examples/accelerator/tinytpu/chia_agent/` | The search: `loop.py` (one search), `swarm.py` (many), `spec_policy.py` (containment), `smoke.py` |
| `chia_runs/` | Every candidate ever scored, with diff, synthesis report, and cost telemetry |
| `scripts/claims.sh` | Runs the claims above |
| `notes/CHIA_CHECKPOINT.md` | Version pins for every component behind these numbers |

The agent may edit **only** `isa.py` and `microarch.py`. It cannot reach the
compiler, the runtime, the benchmarks, or the evaluator — and because those two
files are *imported* by the evaluator, `spec_policy.py` refuses edits that would
execute at import. That is a policy check, not a sandbox; real isolation means a
container.

## Not yet reproducible

Stated plainly, because an artifact that hides this is worse than one that
admits it.

- **S1–S3 need replication.** Every hypothesis has n=1. Idea 4 is the project's
  most interesting claim and currently its least evidenced.
- **The search is not seeded.** Re-running produces a different trajectory. Only
  the replay path (C6, C7) is deterministic.
- **No permanent archive.** For an *Artifacts Available* badge this needs a DOI
  (Zenodo), not just a git branch.
- **Agent runs need external services.** Vertex AI and a billing account; C1–C7
  need none of that, which is why the claim tiers are split the way they are.
