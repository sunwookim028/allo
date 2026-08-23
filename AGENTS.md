# Building
- Always run `conda activate allo-rtlgen` before building or running tests
- Run `ninja -C build [target]` to build specific targets when only using C++ side tools (e.g. `allo-opt`)
- Always run `pip install -e .` to synchronize python packages when modifying both Python side and C++ side
  (it also rebuilds C++ side automatically). `ninja -C build` **will not** synchronize Python packages.
- OR-Tools is a required dependency; set `CMAKE_PREFIX_PATH=$HOME/.local/share/or-tools-9.15`
  so CMake finds its packages.

# Testing
- Run `python -m pytest tests/` to run all tests
- Only run incremental cosim tests in `tests/rtl` to save time when developing.
  Avoid full suite run as far as possible.
- Set `XILINX_VITIS` to any invalid path to skip tests for synthesis with Vitis to save time
- Install the developer toolchain with `pip install -e .[dev]` to run the RTL cosim tests
- Run the RTL cosim tests in parallel with `pytest tests/rtl -n [jobs]` (pytest-xdist).

# Running
- Use `conda run -n allo-rtlgen <command>` to run commands in the conda environment
- When the host system is not compatible with a specific Vitis version,
  use `docker/run-vitis.sh <command>` to run commands in a docker container.

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise.
- If user explicitly requests a refactor, then larger diffs are acceptable,
  prefer cleaner code structure for future maintainability at this time.
- Use Modern C++ features and best practices in C++ code
- Use `assert` to enforce invariants and assumptions that should always hold by the design,
  and fail loudly during development instead of being silently tolerated.
- Always prefer systematic solutions over ad-hoc fixes when developing a new feature,
  even though it may take more effort and break some regression tests in the short term.
- Don't over encapsulate code in helper functions, e.g. encapsulating one or two lines
  of logic in a free function, which can make the code harder to read and understand.

# Comment Style
- DO NOT include any reference to intermediate design documents.
- DO NOT include too much reasoning or explanation in comments.
- DO NOT drafting design documents in comments.
- DO NOT continuously append comments when the code is modified, instead, update the comments
  to reflect the current state of the code.
- The comments should always be publicly understandable and self-contained. Always be concise.

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation

# Repository structure
- Place Python frontend code in `allo/`
- Place MLIR dialects and passes code in `mlir/`
- Use `drafts/` for temporary code when exploring new ideas

# Allo Usage
- This is not the upstream version of Allo.
- DO NOT assume the project structure, APIs, or compiler behavior is
  the same as the upstream Allo project.
- See [ALLO.md](ALLO.md) for a concise reference on writing Allo kernels and schedules

# Environment setup
- Follow instructions in [ENVIRONMENT.md](ENVIRONMENT.md) for setting up the Vivado/Vitis environment
