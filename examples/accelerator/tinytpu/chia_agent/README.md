# TinyTPU CHIA agent

This is a minimal, host-runnable CHIA loop for TinyTPU co-design. Its agent can
read and edit only `isa.py` and `microarch.py`; it cannot alter generic compiler
code, runtime, tests, or benchmarks. `microarch.py` stays explicitly composed
from named Allo-HLS blocks: `dma_load`/`dma_store`, `vload`/`vstore`, `vpu`,
`mxu`, and the top-level decoder/composition. The agent may connect or refine
those blocks (or add one focused `@tpu.unit`) while keeping ISA, decoder, and
schedules aligned.

The host uses `chia_env` (Python 3.10); the check is run by the existing `allo`
environment through `conda run -n allo`. No Docker is required for the local
demo.

## First authenticated run

After Vertex AI, billing, and Application Default Credentials are configured:

```bash
export GOOGLE_CLOUD_PROJECT=<project-id>

source /home/sk3463/miniconda3/etc/profile.d/conda.sh
conda activate chia_env
npm install --prefix /tmp/chia-opencode-cli opencode-ai@1.18.25
export PATH=/tmp/chia-opencode-cli/node_modules/.bin:$PATH
export TINYTPU_CONDA="$(command -v conda)"

ray stop
ray start --head --resources='{"opencode_creds": 1}' --include-dashboard=false
python chia_agent/loop.py \
  --task 'Explore whether MXU and VPU can share tiled GEMM values through VREG rather than VMEM. Preserve results and minimize the VREG/VMEM traffic objective.'
ray stop
```

The default model is `google-vertex/gemini-3.1-pro-preview`; override it with
`TINYTPU_OPENCODE_MODEL`. Each candidate must pass the direct-TOSA compiler
check, then is evaluated on 8×8×8 and 8×16×32 GEMMs. The pre-synthesis
checkpoint has frozen costs: `VREG words × 1 + VMEM words × 4`; DRAM traffic
is reported but intentionally not scored. Allo-HLS export and synthesis are
deferred until the dedicated HLS tooling server is available.
`cluster.yaml` remains available for multi-machine CHIA use; it requires SSH
connectivity between the declared workers.

## Prerequisites

Install CHIA in a separate Python 3.10 environment, then install it editable:

```bash
git clone https://github.com/ucb-bar/chia.git /tmp/chia
conda create -n chia_env python=3.10.19
conda run -n chia_env pip install -e /tmp/chia
```

The parent Allo checkout must already have its `allo` environment built, as
documented by that repository. For Vertex AI, enable `aiplatform.googleapis.com`,
enable billing, then run `gcloud auth application-default login` and
`gcloud auth application-default set-quota-project <project-id>`.
