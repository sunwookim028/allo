# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a copy of a synthesized project with one program's testbench, so
`profile.sh`-style dataflow profiling can be run on it.

    python pyrun.py prof_job.py <base_prj> <job_dir> <M> <K> <N> [order]

The base must already have been synthesized by `parity_sweep.py`. Writes the
testbench and rewrites the solution's recorded project path; it does not run
Vitis.
"""

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis import cosim as C  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program  # noqa: E402

base, job = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
M, K, N = (int(x) for x in sys.argv[3:6])
order = sys.argv[6] if len(sys.argv) > 6 else "shipped"

shutil.rmtree(job, ignore_errors=True)
shutil.copytree(base, job, symlinks=True, ignore=shutil.ignore_patterns("sim"))
subprocess.call(["grep", "-rlZ", base, os.path.join(job, "out.prj")],
                stdout=open(os.path.join(job, "paths"), "w"))
for p in filter(None, open(os.path.join(job, "paths")).read().split("\0")):
    t = open(p, errors="surrogateescape").read()
    open(p, "w", errors="surrogateescape").write(t.replace(base, job))
open(os.path.join(job, "tb.cpp"), "w").write(
    C.testbench(M, K, N, prog=gemm_program(M, K, N, order=order)))
print(f"prepared {job} for {order} {M}x{K}x{N}")
