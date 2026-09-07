# Vitis/Alveo U55C Environment

> **For the CHIA co-design flow, ignore the Docker path below.** Everything in
> [CODESIGN.md](CODESIGN.md) sources Vitis directly from the host:
> `. /opt/xilinx/Vitis_HLS/2023.2/settings64.sh`. C-synthesis needs no licence
> and no container. The `/tools/Xilinx/...` path below does not exist on this
> machine; the Docker flow is for `hw_emu`/`hw` builds, which this project does
> not run.

**Default version:** Vitis 2023.2 (best Alveo U55C compatibility).
**Always use Docker** for Vitis commands to avoid host library conflicts:
```bash
cd /path/to/project && /path/to/docker/run-vitis.sh command
```
Docker auto-configures Vitis 2023.2, XRT, and Alveo U55C. Platform files at `/opt/xilinx/platforms`.

Pull with `docker pull kkaish/vitis-runtime:2023.2` if the image is not locally available.

See `docker/run-vitis.sh` for environment variable overrides.

Direct host sourcing (non-Docker): `/tools/Xilinx/Vitis/{2022.1,2023.2,2024.2}/settings64.sh`.

# Makefile Targets (Allo-generated)

| Target        | Notes                                                |
| ------------- | ---------------------------------------------------- |
| `make csynth` | HLS only — no Docker needed, log at `hls_csynth.log` |
| `make xo`     | csynth + XO packaging - log at `hls_xo.log`          |
| `make host`   | host executable                                      |
| `make xclbin` | link XO → xclbin - log at `sys_link.log`             |
| `make all`    | xo + host + xclbin                                   |
| `make run`    | run host + xclbin  - log at `emu_run.log`            |
| `make clean`  | purge artifacts                                      |

`TARGET`: defaults to `hw_emu`. **Prefer `hw_emu`** unless explicitly requested — `hw` bitstream takes hours.

# Typical Workflow

```python
# In Allo (Python)
mod = s.export("vitis", device="u55c")
# 1. configure axi interfaces, etc.
mod.set_axi(0, bundle="gmem0", offset="slave") # configure m_axi for interface 0
mod.set_axilite(1) # configure s_axilite for interface 1
# 2. generate sample input with numpy for cosim (if cosim needed)
A = np.random.rand(16, 16).astype(np.float32)
B = 1024 # scalar input is also supported
mod.scaffold_project("/path/to/prj/", A, B)
```

```bash
# Then:
cd /path/to/prj/ &&
/path/to/docker/run-vitis.sh make run TARGET=hw_emu
```
