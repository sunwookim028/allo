import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
import rtl, stimulus, ref
U = rtl.RtlUnit(top="vpu_bf16_add", sources=["src/core/vpu/vpu_bf16_add.sv"],
                inputs=[("a_i", 16), ("b_i", 16)], outputs=[("result_o", 16)], shape="comb")
st = stimulus.binary_bf16()
out, _ = rtl.run(U, st)
R = out[:, 0].astype(np.uint16)
print("minitpu RTL == ref:", (R == ref.vpu_bf16_add(st[:,0], st[:,1])).all())
for name in ("bits", "native"):
    G = np.load(f"{name}_got.npy")
    print(name, "vs MiniTPU RTL: match", (G == R).sum(), "/", len(R), "mismatch", (G != R).sum())
np.save("minitpu_rtl.npy", R)
