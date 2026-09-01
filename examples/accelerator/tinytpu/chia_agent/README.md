# TinyTPU CHIA agent

This is a minimal, host-runnable CHIA loop for ISA-only co-design. Its agent
can use only four MCP operations: read `isa.py`, apply a unified diff to
`isa.py`, insert text after a unique `isa.py` anchor, and run the direct-TOSA
compiler check. It cannot edit `microarch.py`, compiler code, or tests. The
hardware/compiler are therefore fixed contracts, as required.

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
  --task 'Add the ISA-level vabs instruction for static-shape TOSA absolute values. Keep the existing ISA unchanged otherwise.'
ray stop
```

The default model is `google-vertex/gemini-3.1-pro-preview`; override it with
`TINYTPU_OPENCODE_MODEL`. The agent’s final compiler check guards existing
direct-TOSA add and negate coverage. A new operation’s backend realization is
intentionally out of scope for this loop. `cluster.yaml` remains available for
multi-machine CHIA use; it requires SSH connectivity between the declared
workers.

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
