# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real cycle measurement: one csynth, then one cosim per submission.

Everything about the toolchain -- the Vitis path, the `-B/usr/bin` link flag
for Vitis 2023.2's binutils against this system's glibc, the `m_axi` depths
cosim needs and Allo does not emit, the `alignas(64)` the `align_value(64)`
promise requires -- is imported from `cosim.py` rather than restated. Prose:
docs/source/extensions/act_specs.rst."""

import os
import signal
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu import cosim, isa_ref  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    IMEM_SIZE, MAXDIM, assemble, schedule, tinytpu_isa,
)
from examples.tinytpu.act import spec as spec_mod  # noqa: E402

PRJ = os.path.abspath(os.environ.get(
    "TPU_PRJ", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "act_sweep.prj")))
COSIM_TIMEOUT = int(os.environ.get("ACT_COSIM_TIMEOUT", 600))


def vitis_bounded(prj, tcl, log, timeout=COSIM_TIMEOUT):
    """`cosim.vitis` with a deadline, because an RTL that never completes must
    be a verdict on the submission rather than a hung judge. Returns
    `(log text, whether it finished)`; on expiry the whole process group,
    ``vitis_hls`` and the ``xsim`` it spawned, is killed."""
    open(os.path.join(prj, "run.tcl"), "w").write(tcl)
    with open(os.path.join(prj, log), "w") as f:
        p = subprocess.Popen(
            ["bash", "-lc",
             f"source {cosim.VITIS} && cd {prj} && vitis_hls -f run.tcl"],
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
            finished = True
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
            finished = False
    return open(os.path.join(prj, log), errors="replace").read(), finished


def free_bytes(sp):
    """Bytes of `C` the spec leaves free: inside the write window, outside the
    output region. Everything else must equal what `isa_ref` leaves there."""
    free = np.zeros((MAXDIM, MAXDIM), np.int8)
    wrs, wcs = spec_mod.write_window(sp)
    free[wrs, wcs] = 1
    _, rs, cs = spec_mod.region(sp, sp["output"])
    free[rs, cs] = 0
    return free


def testbench(sp, prog, dist="small", seed=700):
    """One call, the whole of `C` compared against `isa_ref` -- the same
    anchor the functional judge uses, so a cosim mismatch is an RTL fault."""
    A, B, C0 = spec_mod.buffers(sp, dist, seed)
    want = isa_ref.run(prog, A, B, C0)
    words = assemble(prog)
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    free = free_bytes(sp)
    return "".join([
        "#include <cstdio>\n#include <cstdint>\n",
        'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
        cosim.carr("imem", imem, "uint64_t"),
        cosim.carr("A", A, "int8_t"),
        cosim.carr("B", B, "int8_t"),
        cosim.carr("C0", C0, "int8_t"),
        cosim.carr("want", want, "int8_t"),
        cosim.carr("free_byte", free.reshape(-1), "int8_t"),
        f"static alignas(64) int8_t C[{MAXDIM * MAXDIM}];\n",
        f"""
int main() {{
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++) C[i] = C0[i];
  tinytpu_isa(imem, A, B, C);
  int bad = 0;
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++)
    if (C[i] != want[i] && !free_byte[i]) bad++;
  printf("TB {sp['name']} mismatches = %d / {MAXDIM * MAXDIM}\\n", bad);
  return bad == 0 ? 0 : 1;
}}
"""])


def synthesize(prj=PRJ):
    from allo.dataflow import customize
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False,
            configs={"align_value": 64})
    cosim.patch_axi_depths(prj)
    open(os.path.join(prj, "tb.cpp"), "w").write("int main() { return 0; }\n")
    cosim.vitis(prj, cosim.TCL_SYN, "csynth.log")
    return prj


def last_rtl_progress(text):
    lines = [l for l in text.splitlines() if "RTL Simulation :" in l and "@" in l]
    return lines[-1].strip() if lines else ""


def measure(sp, prog, prj=PRJ, tag=None, timeout=COSIM_TIMEOUT):
    """`(cycles, one line about the run)` from one `cosim_design` on an
    existing build. `cycles` is None when the RTL did not pass, and the line
    says whether that was a mismatch, a timeout, or no testbench output."""
    tag = tag or sp["name"]
    open(os.path.join(prj, "tb.cpp"), "w").write(testbench(sp, prog))
    rpt = os.path.join(prj, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
    if os.path.exists(rpt):
        os.remove(rpt)
    text, finished = vitis_bounded(prj, cosim.TCL_COSIM, f"cosim_{tag}.log",
                                   timeout)
    lines = [l.strip() for l in text.splitlines() if "mismatches" in l]
    if not finished:
        return None, (f"RTL DID NOT COMPLETE within {timeout}s; csim said "
                      f"{lines[0] if lines else 'nothing'}; last RTL progress "
                      f"{last_rtl_progress(text) or 'none'}")
    return cosim.cycles(prj), (lines[-1] if lines else "no TB line")
