"""Derive a sequential variant of microarch_v2 by text transform.

The point is that the compute units are *byte-identical* to the pipelined
design's -- the array, the weight latch, the accumulator drain and the vector
unit are the same source -- so a cycle difference between the two is the
decoupled access-execute structure and nothing else. Only the top changes: the
four `async` processes and their queues collapse into one fetch-decode-dispatch
loop, which is `microarch.py`'s shape with `microarch_v2.py`'s datapath.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "microarch_v2.py"
DST = SRC.with_name("microarch_v2_serial.py")

s = SRC.read_text()
s = s.replace('tpu2 = ISA("CornellTPU-v2")', 'tpu2 = ISA("CornellTPU-v2-serial")')

# drop the four concurrent processes and the dataflow entry
i = s.index("# --- The four concurrent processes ---")
j = s.index("# ==========================================================================#\n# Scheduling.")
s = s[:i] + '''# --- Sequential top: one fetch-decode-dispatch loop ------------------------ #
# The measurement control for the pipelined design. Same units, same schedule,
# same instruction stream; the only difference is that this top calls one unit
# at a time and waits, instead of routing commands to four processes that run
# concurrently. It is `microarch.py`'s shape with `microarch_v2.py`'s datapath,
# so a cycle difference between the two isolates the decoupling from the array.
@tpu2.entry
def tinytpu2s(
    dmem: f32[DRAM_SIZE],
    imem: i32[IMEM_SIZE],
    n_instr: i32,
    n_ld: i32,
    n_ex: i32,
    n_st: i32,
):
    sp0: f32[SPAD_ROWS]
    sp1: f32[SPAD_ROWS]
    sp2: f32[SPAD_ROWS]
    sp3: f32[SPAD_ROWS]
    wst: f32[DIM, DIM]
    ac0: f32[AROWS]
    ac1: f32[AROWS]
    ac2: f32[AROWS]
    ac3: f32[AROWS]
    vreg: f32[VEC_REGS, VEC_LANES]
    for pc in arange(n_instr, name="pc"):
        base: i32 = pc * IWIDTH
        op: i32 = imem[base]
        a0: i32 = imem[base + 1]
        a1: i32 = imem[base + 2]
        a2: i32 = imem[base + 3]
        zf: i32 = 0
        if op == OP_MM0:
            zf = 1
        rl: i32 = 0
        if op == OP_ACCRELU:
            rl = 1
        if op == OP_DMA_LOAD:
            for r in arange(a2, name="r"):
                for e in arange(DIM, name="e"):
                    w: f32 = dmem[a0 + r * DIM + e]
                    if e == 0:
                        sp0[a1 + r] = w
                    elif e == 1:
                        sp1[a1 + r] = w
                    elif e == 2:
                        sp2[a1 + r] = w
                    else:
                        sp3[a1 + r] = w
        elif op == OP_DMA_STORE:
            for r in arange(a2, name="sr2"):
                for e in arange(DIM, name="se2"):
                    u: f32 = 0.0
                    if e == 0:
                        u = sp0[a0 + r]
                    elif e == 1:
                        u = sp1[a0 + r]
                    elif e == 2:
                        u = sp2[a0 + r]
                    else:
                        u = sp3[a0 + r]
                    dmem[a1 + r * DIM + e] = u
        elif op == OP_LOADW:
            loadw(sp0, sp1, sp2, sp3, wst, a0)
        elif op == OP_MM0 or op == OP_MM:
            mmu(sp0, sp1, sp2, sp3, wst, ac0, ac1, ac2, ac3, a0, a2, zf)
        elif op == OP_ACCST or op == OP_ACCRELU:
            accst(ac0, ac1, ac2, ac3, sp0, sp1, sp2, sp3, a1, a2, rl)
        elif op == OP_VLOAD:
            vload(sp0, sp1, sp2, sp3, vreg, a0, a1)
        elif op == OP_VSTORE:
            vstore(vreg, sp0, sp1, sp2, sp3, a0, a1)
        else:
            vpu(op, vreg, a0, a1, a2)


''' + s[j:]

# the process schedules have no counterpart here
for pat in (r"^seq_s = .*?\n(?:.*?\n)*?^ld_s = ", ):
    pass
s = re.sub(r"^seq_s = sequencer\.schedule\(\)\n(?:.+\n)*?\n", "", s, flags=re.M)
s = re.sub(r"^ld_s = loader\.schedule\(\)\n(?:.+\n)*?\n", "", s, flags=re.M)
s = re.sub(r"^st_s = storer\.schedule\(\)\n(?:.+\n)*?\n", "", s, flags=re.M)
s = re.sub(r"^ex_s = executor\.schedule\(\)\n(?:.+\n)*?\n", "", s, flags=re.M)
s = s.replace('''top_s = tinytpu2.schedule()
top_s.compose(seq_s, ld_s, ex_s, st_s)''',
'''top_s = tinytpu2s.schedule()
top_s.partition(top_s.buffer("vreg"), dim=1, kind=top_s.Complete)
top_s.partition(top_s.buffer("wst"), kind=top_s.Complete)
top_s.pipeline("e")
top_s.pipeline("se2")
top_s.compose(mmu_s, lw_s, as_s, vpu_s, vl_s, vs_s)''')
DST.write_text(s)
print("wrote", DST)
