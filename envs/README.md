# envs

Pinned conda environments for this repo: one directory per env, each with a standard
`environment.yml` (conda packages, pinned) and `requirements.txt` (pip packages, pinned).
Generated 2026-10-31 from the live envs on ace-01 and verified by rebuilding each one.
The repo-root `requirements.txt` is different: it is the package's own (unpinned) install list.

| env | python | for |
|---|---|---|
| `allo` | 3.12.13 | allo development, tinytpu flows. Sets `LLVM_BUILD_DIR` |
| `chia_env` | 3.10.19 | the CHIA agent (`examples/tinytpu/chia_agent`) |

Create (from the repo root):

    conda env create -f envs/allo/environment.yml
    conda run -n allo python -m pip install -e . --no-build-isolation   # allo itself, from this checkout

Host-specific settings live in `variables:` in `environment.yml`. The value there is ace-01's.
Other hosts are listed as comments with the one command that sets their value; run it once
after creating the env, and check with `conda env config vars list -n allo`.
