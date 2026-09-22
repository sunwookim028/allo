# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real cycle measurement: one csynth, then one cosim per submission.

Everything about the toolchain -- the Vitis path, the `-B/usr/bin` link flag
for Vitis 2023.2's binutils against this system's glibc, the `m_axi` depths
cosim needs and Allo does not emit, the `alignas(64)` the `align_value(64)`
promise requires -- is imported from `cosim.py` rather than restated. Prose:
docs/source/extensions/act_specs.rst."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis import cosim, isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    IMEM_SIZE, MAXDIM, assemble, schedule, tinytpu_isa,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402

PRJ = os.path.abspath(os.environ.get(
    "TPU_PRJ", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "act_sweep.prj")))


def testbench(sp, prog, dist="small", seed=700):
    """One call, the whole of `C` compared against `isa_ref` -- the same
    anchor the functional judge uses, so a cosim mismatch is an RTL fault."""
    A, B, C0 = spec_mod.buffers(sp, dist, seed)
    want = isa_ref.run(prog, A, B, C0)
    words = assemble(prog)
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    allowed, _ = _window_mask(sp)
    return "".join([
        "#include <cstdio>\n#include <cstdint>\n",
        'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
        cosim.carr("imem", imem, "uint64_t"),
        cosim.carr("A", A, "int8_t"),
        cosim.carr("B", B, "int8_t"),
        cosim.carr("C0", C0, "int8_t"),
        cosim.carr("want", want, "int8_t"),
        cosim.carr("free_byte", allowed.reshape(-1), "int8_t"),
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


def _window_mask(sp):
    """Bytes of `C` whose value the spec leaves free, and the output region."""
    allowed = np.zeros((MAXDIM, MAXDIM), np.int8)
    wrs, wcs = spec_mod.write_window(sp)
    allowed[wrs, wcs] = 1
    _, rs, cs = spec_mod.region(sp, sp["output"])
    allowed[rs, cs] = 0
    return allowed, (rs, cs)


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


def measure(sp, prog, prj=PRJ, tag=None):
    """`(cycles, testbench line)` from one `cosim_design` on an existing build."""
    tag = tag or sp["name"]
    open(os.path.join(prj, "tb.cpp"), "w").write(testbench(sp, prog))
    rpt = os.path.join(prj, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
    if os.path.exists(rpt):
        os.remove(rpt)
    text = cosim.vitis(prj, cosim.TCL_COSIM, f"cosim_{tag}.log")
    lines = [l.strip() for l in text.splitlines() if "mismatches" in l]
    return cosim.cycles(prj), (lines[-1] if lines else "no TB line")
