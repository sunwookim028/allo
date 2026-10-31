# envs

The `allo` conda env, one file per server. Packages are identical everywhere; the files differ
only in `variables:` (server paths such as `LLVM_BUILD_DIR`).

    conda env create -f envs/$(hostname -s).yml                         # from the repo root
    conda run -n allo python -m pip install -e . --no-build-isolation   # allo itself, from this checkout

A server without a file fails right there (`EnvironmentFileNotFound`). To add one, copy an existing
`<host>.yml` to `envs/<hostname -s>.yml` and set its values. Check them with
`conda env config vars list -n allo`.

`requirements.txt` here pins every pip package (shared by all servers). The repo-root
`requirements.txt` is the package's own, unpinned install list.
The CHIA agent's env is separate: `examples/tinytpu/chia_agent/environment.yml`.
