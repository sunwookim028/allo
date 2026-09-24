# Per-pass IR dumper, v3 (correct). Key discovery baked in:
#
#   EVERY Allo schedule primitive is decorated with @wrapped_apply
#   (allo/customize.py:83), which calls _mlir_lower_pipeline(module) AFTER the
#   primitive body. _mlir_lower_pipeline runs loop_transformation + affine-loop-
#   normalize + CSE + affine-simplify-structures. So a primitive is NOT lazy: by
#   the time sch.split()/sch.pipeline() returns, the nest is already rewritten and
#   redundancy is already CSE'd away.
#
#   => To observe an individual pass in isolation you must apply NO primitive and
#      drive the PassManager by hand. That is Demo B.
#
# Run:  cd /home/zsm9/allo_sup && conda activate allo && python dump_ir_passes.py
import io
import allo
from allo.ir.types import int32
from allo._mlir.dialects import allo as allo_d
from allo._mlir.passmanager import PassManager

chunks = []


def dump(title, module):
    line = "=" * 78
    block = f"\n{line}\n{title}\n{line}\n{str(module).rstrip()}\n"
    chunks.append(block)
    print(block)


def section(name):
    bar = "#" * 78
    block = f"\n{bar}\n# {name}\n{bar}\n"
    chunks.append(block)
    print(block)


def run_pass(module, pipeline_str):
    with module.context:
        PassManager.parse(pipeline_str).run(module.operation)


# ============================================================================
# DEMO A -- a primitive is EAGER: sch.split applies the nest rewrite immediately
#           (because of the @wrapped_apply decorator running _mlir_lower_pipeline)
# ============================================================================
def splitme(A: int32[8], B: int32[8]) -> int32[8]:
    C: int32[8]
    for i in range(8):
        C[i] = A[i] + B[i]
    return C


section("DEMO A -- schedule primitives are EAGER (nest rewrite happens inside sch.split)")
schA = allo.customize(splitme)
dump("A0  Frontend: one flat loop `affine.for 0 to 8`.", schA.module)

schA.split("i", factor=2)
dump("A1  Immediately after sch.split('i',2): nest is ALREADY (0..4)x(0..2) with "
     "affine.apply #map. The split was applied inside the call (@wrapped_apply -> "
     "_mlir_lower_pipeline). Only leftover is a dead `allo.create_op_handle`.",
     schA.module)

allo_d.loop_transformation(schA.module)
dump("A2  A second manual loop_transformation just garbage-collects the dead "
     "`allo.create_op_handle`. Nothing else changes.", schA.module)


# ============================================================================
# DEMO B -- cse (and friends) in ISOLATION: apply NO primitive, drive passes by hand
# ============================================================================
def redundant(A: int32[8], B: int32[8]) -> int32[8]:
    C: int32[8]
    for i in range(8):
        C[i] = (A[i] + B[i]) * (A[i] + B[i])   # A[i]+B[i] is BUILT TWICE
    return C


section("DEMO B -- individual passes on a NON-scheduled module (so redundancy survives)")
schB = allo.customize(redundant)
dump("B0  Frontend (customize only, no primitive): DUPLICATED work -- %0..%4 and "
     "%5..%9 are two independent copies of A[i]+B[i].", schB.module)

allo_d.loop_transformation(schB.module)
dump("B1  After loop_transformation ONLY: duplicate is STILL present. This pass "
     "handles schedule directives/loops; it does NOT do CSE.", schB.module)

run_pass(schB.module, "builtin.module(func.func(affine-loop-normalize))")
dump("B2  After affine-loop-normalize ONLY: loop already canonical (0..8, step 1); "
     "duplicate still present.", schB.module)

run_pass(schB.module, "builtin.module(func.func(cse))")
dump("B3  After cse ONLY: the SECOND A[i]+B[i] (2 loads + 2 extsi + 1 addi) is "
     "ELIMINATED. The mul now squares a single value (`muli %x, %x`). <-- cse's "
     "isolated effect, finally visible.", schB.module)

run_pass(schB.module, "builtin.module(func.func(affine-simplify-structures))")
dump("B4  After affine-simplify-structures ONLY: nothing left to simplify here; "
     "no change.", schB.module)

buf = io.StringIO()
allo_d.emit_vhls(schB.module, buf, flatten=False)
buf.seek(0)
dump("B5  TRANSLATION emit_vhls: final HLS C++ -- one add feeding one mul.", buf.read())


OUT = "/home/zsm9/pe_core_implementation/Allo_extension/IR_PER_PASS_WALKTHROUGH.txt"
with open(OUT, "w") as f:
    f.write("Allo IR per-PASS walkthrough (each MLIR pass run individually).\n\n")
    f.write("KEY FINDING: every schedule primitive (sch.split/pipeline/...) is\n")
    f.write("@wrapped_apply-decorated (allo/customize.py:83) and runs the FULL lowering\n")
    f.write("pipeline (loop_transformation + affine-loop-normalize + CSE +\n")
    f.write("affine-simplify-structures) immediately. So primitives are EAGER, and to\n")
    f.write("see a single pass in isolation you must apply NO primitive and drive the\n")
    f.write("PassManager by hand (Demo B).\n\n")
    f.write("Demo A: primitive eagerness (sch.split rewrites the nest inside the call).\n")
    f.write("Demo B: cse isolated on a non-scheduled module.\n")
    f.write("Generated by allo_sup/dump_ir_passes.py.\n")
    f.write("".join(chunks))
print(f"\n[written] {OUT}")
