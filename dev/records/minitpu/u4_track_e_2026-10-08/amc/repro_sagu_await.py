"""AMC finding E-A6 (crash, blocks scalar_agu on both schedules): the
``scalar_agu`` register-discipline kernel reduced by greedy line deletion
(``ddmin.py`` beside it, 175 -> 55 lines) while the ``amc`` build still dies
with
    error: 'loopschedule.await' op operation destroyed but still has uses
    LLVM ERROR: operation destroyed but still has uses
(the ``llvm`` target builds). What is left: fifteen 1-element-array
registers read at the top, four stores of them to a 2-D output, and a 2-deep
pipe's five copies ``p1_x = p0_x`` under ``if live == 1``. The same copies
alone (``repro_cond_copy.py cond``) build; the kernel needs the rest.
    python repro_sagu_await.py     (AMC env, under scl enable gcc-toolset-13)
"""
KERNEL = '''
def saguk(RST: uint32[N], SV: uint32[N], SOP: uint32[N], SRD: uint32[N], SRS: uint32[N], SIV: uint32[N], SLV: uint32[N],
        SIMM: uint32[N], KA: uint32[N, 4], IV: uint32[N, 8], SB: uint32[N], SS: uint32[N], LA: uint32[N], SLB: uint32[N],
        SREG: uint32[N, 4], OB: uint32[N], OS: uint32[N], OL: uint32[N], OW: uint32[N]):
    sreg0_r: UInt(32)[1] = 0
    sreg1_r: UInt(32)[1] = 0
    sreg2_r: UInt(32)[1] = 0
    sreg3_r: UInt(32)[1] = 0
    written_r: UInt(4)[1] = 0
    p0_v_r: UInt(1)[1] = 0
    p0_rd_r: UInt(2)[1] = 0
    p0_data_r: UInt(32)[1] = 0
    p0_prod_r: UInt(32)[1] = 0
    p0_mac_r: UInt(1)[1] = 0
    p1_v_r: UInt(1)[1] = 0
    p1_rd_r: UInt(2)[1] = 0
    p1_data_r: UInt(32)[1] = 0
    p1_prod_r: UInt(32)[1] = 0
    p1_mac_r: UInt(1)[1] = 0
    for t in range(N):
        sreg0: UInt(32) = sreg0_r[0]
        sreg1: UInt(32) = sreg1_r[0]
        sreg2: UInt(32) = sreg2_r[0]
        sreg3: UInt(32) = sreg3_r[0]
        written: UInt(4) = written_r[0]
        p0_v: UInt(1) = p0_v_r[0]
        p0_rd: UInt(2) = p0_rd_r[0]
        p0_data: UInt(32) = p0_data_r[0]
        p0_prod: UInt(32) = p0_prod_r[0]
        p0_mac: UInt(1) = p0_mac_r[0]
        p1_v: UInt(1) = p1_v_r[0]
        p1_rd: UInt(2) = p1_rd_r[0]
        p1_data: UInt(32) = p1_data_r[0]
        p1_prod: UInt(32) = p1_prod_r[0]
        p1_mac: UInt(1) = p1_mac_r[0]
        live: UInt(1) = RST[t]
        SREG[t, 0] = sreg0
        SREG[t, 1] = sreg1
        SREG[t, 2] = sreg2
        SREG[t, 3] = sreg3
        OW[t] = written
        if live == 1:
            p1_v = p0_v
            p1_rd = p0_rd
            p1_data = p0_data
            p1_prod = p0_prod
            p1_mac = p0_mac
        p1_v_r[0] = p1_v
        p1_rd_r[0] = p1_rd
        p1_data_r[0] = p1_data
        p1_prod_r[0] = p1_prod
        p1_mac_r[0] = p1_mac
'''
import os, importlib.util
TMP = os.environ.get("TMPDIR", "/tmp")
p = f"{TMP}/repro_sagu_await_k.py"
open(p, "w").write("from allo.ir.types import UInt, uint32\nN = 64\n" + KERNEL)
spec = importlib.util.spec_from_file_location("rsa", p); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo
allo.customize(K.saguk).build(target="llvm"); print("RESULT llvm built", flush=True)
allo.customize(K.saguk).build(target="amc"); print("RESULT amc built", flush=True)
