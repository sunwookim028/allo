# TinyTPU CHIA agent

A host-runnable CHIA co-design loop for TinyTPU. The agent can read and edit
only `isa.py` and `microarch.py`; it cannot alter the generic ACT compiler,
runtime, tests, benchmarks, or the evaluator. `microarch.py` stays explicitly
composed from named Allo-HLS blocks: `dma_load`/`dma_store`, `vload`/`vstore`,
`vpu`, `mxu`, and the top-level decoder/composition. The agent may connect or
refine those blocks (or add one focused `@tpu.unit`) while keeping ISA,
decoder, and schedules aligned.

The host uses `chia_env` (Python 3.10); checks run in the existing `allo`
environment through `conda run -n allo`. No Docker is required.

## The objective

Each candidate is scored by **synthesizing it**:

1. the composed `top_s` schedule is exported to Vitis HLS and C-synthesized,
2. every unit's `(ii, depth)` is re-measured from that report, replacing
   `microarch.py`'s frozen declarations (`../synth.py`),
3. the GEMM benchmarks are compiled, checked against NumPy, and costed under
   the measured table (`../ppa.py`).

The score is `sum(CompiledProgram.cycles())` over the 8x8x8 and 8x16x32 GEMMs.
Area and Fmax are recorded for every candidate but **not** scored. Because the
latency table is overwritten from synthesis before scoring, an optimistic
`ISA.latency` declaration cannot move the score.

One evaluation (synthesis included) takes about 50 seconds.

## What is here

| File | Role |
| --- | --- |
| `loop.py` | one search: baseline, propose from best, gate, score, keep or rewind |
| `swarm.py` | several searches at once, each in its own worktree and on its own hypothesis |
| `allo_tool.py` | the MCP surface the agent gets: read, patch, check, score |
| `spec_policy.py` | what an agent-authored spec may contain (see *Containment*) |
| `smoke.py` | smallest end-to-end check that the whole path is wired up |

## The search

`loop.py` runs generate -> gate -> score -> keep-or-rewind:

- it synthesizes the unmodified design first, as the baseline,
- each iteration proposes one candidate *from the best design so far*,
- a candidate must pass the direct-TOSA compiler check, then is synthesized
  and scored,
- it is kept only if it strictly beats the best score; otherwise the writable
  spec is rewound and the next iteration is told what failed,
- every candidate is appended to `<log-dir>/variants.jsonl` with its score,
  synthesis summary, and diff; the winning diff lands in `<log-dir>/best.diff`.

The repository is left holding the best design found.

## Running it

Prerequisites are installed under `/home/sk3463/chia-tools` (deliberately not
`/tmp`, which gets wiped):

```bash
git clone https://github.com/ucb-bar/chia.git /home/sk3463/chia-tools/chia
conda create -n chia_env python=3.10.19
conda run -n chia_env pip install -e /home/sk3463/chia-tools/chia
npm install --prefix /home/sk3463/chia-tools/opencode-cli opencode-ai@1.18.25
```

Then:

```bash
export GOOGLE_CLOUD_PROJECT=test-adrs
. /opt/xilinx/Vitis_HLS/2023.2/settings64.sh   # the score gate needs Vitis

source /home/sk3463/miniconda3/etc/profile.d/conda.sh
conda activate chia_env
export PATH=/home/sk3463/chia-tools/opencode-cli/node_modules/.bin:$PATH
export TINYTPU_CONDA="$(command -v conda)"

ray stop
ray start --head --resources='{"opencode_creds": 1}' --include-dashboard=false
python chia_agent/loop.py \
  --iterations 5 \
  --log-dir chia_runs/$(date +%Y%m%d-%H%M%S) \
  --task 'Reduce the synthesized cycle count of the tiled GEMMs. dma_load
          dominates: its measured depth is 75 cycles of m_axi read latency paid
          per instruction, so fewer, larger DRAM transfers or more on-chip reuse
          through VREG should pay off.'
ray stop
```

The default model is `google-vertex/gemini-3.1-pro-preview`; override with
`TINYTPU_OPENCODE_MODEL`. `cluster.yaml` remains available for multi-machine
CHIA use; it requires SSH connectivity between the declared workers.

## Vertex AI

Project `test-adrs` has `aiplatform.googleapis.com` enabled and billing active,
and `gemini-3.1-pro-preview` answers on location `global`. Application Default
Credentials are already present and refresh cleanly. If they lapse:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project test-adrs
```

## Environment knobs

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_CLOUD_PROJECT` | *(required)* | Vertex AI project |
| `TINYTPU_VERTEX_LOCATION` | `global` | Vertex AI location |
| `TINYTPU_OPENCODE_MODEL` | `google-vertex/gemini-3.1-pro-preview` | model |
| `TINYTPU_CONDA` | discovered | conda binary used to reach the build environment |
| `TINYTPU_ENV` | `allo` | conda environment holding the built `allo`. Set it if you built this checkout into a differently-named environment — otherwise the agent scores a different tree than the one you are editing. |
| `TINYTPU_VITIS_SETTINGS` | `/opt/xilinx/Vitis_HLS/2023.2/settings64.sh` | sourced for the synthesis gate |
| `TINYTPU_SYNTH_PROJECT` | `/tmp/tinytpu_chia_synth` | scratch HLS project; each tool instance appends its own name, so parallel workers never share one |
| `TINYTPU_CHIA_LOG_DIR` | `chia_runs/latest` | default variant log directory |

## Containment

`isa.py` and `microarch.py` are Python modules that the evaluator *imports*, so
whatever an agent writes there runs inside the process that scores it. A live
run demonstrated this: an agent that had given up on a candidate wrote
file-rewriting code into `isa.py` to revert itself, and that code executed at
import.

`spec_policy.py` decides what those files may contain — imports restricted to
what a spec needs, no `open`/`exec`/`eval`/`__import__`, no write or spawn
attributes, and no module-level `with`/`try`, which is how import-time side
effects hide. Both edit paths refuse a violating edit and leave the file
untouched. `tests/dsa/test_tinytpu_agent_policy.py` pins it in both directions:
the shipped spec satisfies the policy, and the module the agent actually wrote
is refused.

This is a policy check, not a sandbox. Real isolation means evaluating in a
container.
