"""RTL probe of accu's dependence claim: `isa_dsl.ar_distance_program(d)` for
d = 1..5, one call each on one RTL instance, the whole of C compared.

    PRJ=<a cosim.py project, synthesized> python ar_distance_probe.py
    e.g. TPU_SHAPES=4x4x4 TPU_PRJ=$PWD/runs/probe python ../cosim.py
         PRJ=$PWD/runs/probe python ar_distance_probe.py

Programs below AR_RAW_DIST are assembled with check=False: they are exactly
what `check_program` refuses, run to show why. Each result line appears twice,
C simulation first, then RTL. Output behind logs/cosim_isa_ar_distance.log."""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4)))
from examples.accelerator.tinytpu_vitis import cosim, isa_ref
from examples.accelerator.tinytpu_vitis import microarch_isa as U
from examples.accelerator.tinytpu_vitis.isa_dsl import ar_distance_program
from examples.accelerator.tinytpu_vitis.stress_isa import operands
isa_ref.check_program = lambda p: None
prj = os.environ["PRJ"]
M = U.MAXDIM
src = ["#include <cstdio>\n#include <cstdint>\n",
       'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
       f"static alignas(64) int8_t C[{M*M}];\n"]
calls = []
crng = np.random.default_rng(5)
for i, d in enumerate((1, 2, 3, 4, 5)):
    prog = ar_distance_program(d)
    A, B = operands("full", 960 + d)
    C0 = crng.integers(-128, 128, M * M).astype(np.int8)
    gold = isa_ref.run(prog, A, B, C0)
    w = U.assemble(prog, check=False)
    imem = np.zeros(U.IMEM_SIZE, np.uint64); imem[:len(w)] = np.array(w, np.uint64)
    src += [cosim.carr(f"imem{i}", imem, "uint64_t"), cosim.carr(f"A{i}", A.reshape(-1), "int8_t"),
            cosim.carr(f"B{i}", B.reshape(-1), "int8_t"), cosim.carr(f"C0_{i}", C0, "int8_t"),
            cosim.carr(f"gold{i}", gold, "int8_t")]
    calls.append(f"""
  for (int i = 0; i < {M*M}; i++) C[i] = C0_{i}[i];
  tinytpu_isa(imem{i}, A{i}, B{i}, C);
  n = 0; for (int i = 0; i < {M*M}; i++) if (C[i] != gold{i}[i]) n++;
  printf("HAZ distance {d}: wrong = %d\\n", n);""")
src.append("int main() {\n  int n;" + "".join(calls) + '\n  printf("mismatches = 0\\n");\n  return 0;\n}\n')
open(os.path.join(prj, "tb.cpp"), "w").write("".join(src))
text = cosim.vitis(prj, cosim.TCL_COSIM, "cosim_hazard.log")
print("\n".join(l for l in text.splitlines() if l.startswith("HAZ")))
