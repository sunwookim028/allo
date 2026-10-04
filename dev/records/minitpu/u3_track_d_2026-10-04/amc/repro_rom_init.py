"""AMC finding: a constant table (memref.global -> rom-allocation -> amc.memory
ROM) reaches SystemVerilog with NO contents: AmcToHW.cpp:1546 creates the
seq.hlmem from shape and element type only and never reads
BankedAllocOp.init. ``out[i] = rom[a[i]]`` returns 0 on the ``amc`` target,
the right values on ``llvm``. Run in the AMC env under gcc-toolset-13."""
import os, sys, importlib.util, numpy as np
TMP = os.environ.get("TMPDIR", "/tmp")
src = '''
import numpy as np
from allo.ir.types import int32
N = 8
TABLE = np.array([11, 22, 33, 44, 55, 66, 77, 88], dtype=np.int32)
def lut(a: int32[N], out: int32[N]):
    rom: int32[8] = TABLE
    for i in range(N):
        k: int32 = a[i]
        out[i] = rom[k]
'''
kp = f"{TMP}/u3d_repro_rom.py"; open(kp, "w").write(src)
spec = importlib.util.spec_from_file_location("u3d_repro_rom", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo
a = np.array([3, 0, 7, 1, 5, 2, 6, 4], np.int32)
for tgt in ("llvm", "amc"):
    s = allo.customize(K.lut); f = s.build(target=tgt)
    out = np.zeros(8, np.int32); f(a, out)
    print(f"{tgt:5s} out={out.tolist()} want={K.TABLE[a].tolist()} {'OK' if (out == K.TABLE[a]).all() else 'WRONG'}")
    if tgt == "amc":
        files = f.dump_verilog(f"{TMP}/u3d_repro_rom_hdl")
        for p in files:
            if "rom" in p.name and "impl" in p.name:
                t = open(p).read()
                print(f"   {p.name}: {t.count(chr(10))} lines, 'initial' x{t.count('initial')}, literals 32'h x{t.count(chr(39)+'h')}, mem decl: {[l.strip() for l in t.splitlines() if 'mem0[' in l][:1]}")
