# FINAL EVA ALLO VERSION
#  
#
# SYS-CREDIT variant of the scoreboard chip: the
# systolic plane gets the SAME credit protocol the router plane has, making
# it a registered equivalent of golden EVA's rq/gt handshake (sync_register:
# 2-slot buffer, stall-not-drop). Senders hold words until credited ->
# LOSSLESS under any rate; the eva.py drop-on-full deviation is repaired.
# eva.py — EVA accelerator: M×N mesh of fused router+PE nodes, always-fire/bubble model.
# Each node carries two overlaid 4-dir networks: systolic (data word) + router (packets).
# To change data type: change Ty.

import allo
from allo.ir.types import float16, int16, int32, UInt, AlloType, Stream, float32
import allo.dataflow as df
import numpy as np

M, N = 2,2 # Mesh dimensions
NSTEP = 10 # how many 'cycles' the array runs -> set to high value

DATADRIVEN = 1 # TODO legacy. DELETE but also delete all references
DRF_DEPTH, IRF_DEPTH = 8, 8 # Data Register File, Instruction Register File
BUF_DEPTH = 2 # credit buffer depth

# SCOREBOARD: issue-to-retire
# depth = fp16 core latency (4) + 1; result ring > depth so a slot is never
# rewritten while in flight. NOTE: the enabling DEPENDENCE-inter-false pragma
# has NO Allo primitive.
SB_DEPTH, RESQ_DEPTH = 5, 8 # scoreboard metadata ring, result ring 
FWD = 1 # operand forwarding ON -> if FWD = 0 forwarding is inactiveand
FP_LAT = 1    # forward-ready threshold (matches bind_op latency)
MOV_LAT = 0

# OPTION-A knobs:
# defaults = the T=1/D=2; the II=1 RTL
# schedule needs ~6 tokens in flight (read-to-write span of the depth-6
# pipeline), so gen_kernel.py overrides these to 6/8 for the cosim kernel.
PRIME_TOKENS = 6   # initial tokens per stream (1 = plain END-PUT prime)
STREAM_DEPTH = 8   # link FIFO depth
LANELEN = NSTEP


# OPCODES
OP_ADD, OP_SUB, OP_MULT, OP_MOV = 0x0, 0x1, 0x2, 0x3
OP_RTR0 = 0x4                     # 0x4..0x7 = router send, dir = opcode[1:0]
OP_GEQ, OP_LT = 0x8, 0x9
OP_CRTR0 = 0xC                    # 0xC..0xF = CONDITIONAL router send (inject iff condition_reg), dir = opcode[1:0]
# EXTENDED ISA: EVA RTL's first-column of PEs has additional capabilities
# div/sqrt. 0xA/0xB are the only free slots between LT(0x9) and CRTR(0xC).
OP_DIV, OP_SQRT = 0xA, 0xB        # DIV a/b (2-cyc golden) ; SQRT sqrt(a) (2-cyc golden)



def get_eva_top(Ty: AlloType = float16):
    DATA_W = Ty.bits
    ID_W, MODE_W, ADDR_W, RQ_W = 4, 1, 4, 1 # destination ID, mode bit, target address inside PE, request/valid bit (bubble?)

    # router packet = {rq,id,mode,addr,data} packed LSB-first into one UIntbut
    D_OFF  = 0 # data  : bits [0 : 16)
    A_OFF  = D_OFF + DATA_W # addr  : bits [16 : 20)
    MD_OFF = A_OFF + ADDR_W # mode  : bit  [20]
    ID_OFF = MD_OFF + MODE_W # id    : bits [21 : 25)
    RQ_OFF = ID_OFF + ID_W # rq    : bit  [25]
    PKT_W  = RQ_OFF + RQ_W # total = 26 bits

    # mask for recording a packet into an int32 rout array: Allo's vhls
    # emission SIGN-extends UInt->int32 stores (sim zero-extends; same bug
    # class as Allo_bugs/uint_slice_signed_emission.py) -> force zero-ext
    PMASK  = (1 << PKT_W) - 1 if PKT_W < 32 else 0x7FFFFFFF
    Pkt    = UInt(PKT_W) # packets used for router stream

    SYS_W  = UInt(1 + DATA_W)                  # bit0 = vld, bits[1:1+DATA_W] = raw data

    @df.region()
    def top(
        # systolic in-/outputs on all edges
        in_w: Ty[M, LANELEN], in_e: Ty[M, LANELEN],
        in_n: Ty[N, LANELEN], in_s: Ty[N, LANELEN],
        out_w: Ty[M, LANELEN], out_e: Ty[M, LANELEN],
        out_n: Ty[N, LANELEN], out_s: Ty[N, LANELEN],
        # router in-/outputs on all edges
        rin_w: int32[M, LANELEN], rin_e: int32[M, LANELEN],
        rin_n: int32[N, LANELEN], rin_s: int32[N, LANELEN],
        rout_w: int32[M, LANELEN], rout_e: int32[M, LANELEN],
        rout_n: int32[N, LANELEN], rout_s: int32[N, LANELEN],
        # input-valid flags (systolic)
        iv_w: int32[M, LANELEN], iv_e: int32[M, LANELEN],
        iv_n: int32[N, LANELEN], iv_s: int32[N, LANELEN],
        # prime values (one per PE tile)
        prime_cfg: int32[M, N],
    ):
        # Systolic and router streams
        sys_e: Stream[SYS_W, STREAM_DEPTH][M, N + 1]
        sys_w: Stream[SYS_W, STREAM_DEPTH][M, N + 1]
        sys_s: Stream[SYS_W, STREAM_DEPTH][M + 1, N]
        sys_n: Stream[SYS_W, STREAM_DEPTH][M + 1, N]
        rtr_e: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_w: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_s: Stream[Pkt, STREAM_DEPTH][M + 1, N]
        rtr_n: Stream[Pkt, STREAM_DEPTH][M + 1, N]
        cr_e: Stream[int32, STREAM_DEPTH][M, N + 1]
        cr_w: Stream[int32, STREAM_DEPTH][M, N + 1]
        cr_s: Stream[int32, STREAM_DEPTH][M + 1, N]
        cr_n: Stream[int32, STREAM_DEPTH][M + 1, N]
        # SYS-CREDIT plane (mirrors cr_*): credits for the systolic links
        scr_e: Stream[int32, STREAM_DEPTH][M, N + 1]
        scr_w: Stream[int32, STREAM_DEPTH][M, N + 1]
        scr_s: Stream[int32, STREAM_DEPTH][M + 1, N]
        scr_n: Stream[int32, STREAM_DEPTH][M + 1, N]

        # ---- SPLIT: local PE<->router interface (NON-BLOCKING, no bubbles) ----
        # eject : router -> PE,  Stream[Pkt]   delivered packet (try_put/try_get)
        # inject: PE -> router,  Stream[int32] (egress_dir<<28) | (send_pkt&PMASK)
        eject: Stream[Pkt, STREAM_DEPTH][M, N]
        inject: Stream[int32, STREAM_DEPTH][M, N]

        @df.kernel(mapping=[M, N], args=[prime_cfg])
        def router(pcfg: int32[M, N]):
            i, j = df.get_pid()
            row_id: int32 = i
            col_id: int32 = j
            oe_r: Pkt = 0; ow_r: Pkt = 0; on_r: Pkt = 0; os_r: Pkt = 0
            rbuf:  Pkt[4, BUF_DEPTH] = 0
            rbcnt: UInt(8)[4] = 0
            rcred: UInt(8)[4] = 0
            cre_r: UInt(8) = BUF_DEPTH; crw_r: UInt(8) = BUF_DEPTH
            crs_r: UInt(8) = BUF_DEPTH; crn_r: UInt(8) = BUF_DEPTH
            # router-side pending core send (held until an egress port frees)
            csd_pkt: Pkt = 0
            csd_dir: int32 = 0
            zpkt: Pkt = 0
            zcr: int32 = 0
            # prime the mesh feedback edges (router + credit planes)
            for _pt in range(pcfg[i, j] - 1):
                rtr_e[i, j + 1].put(zpkt)
                rtr_w[i, j].put(zpkt)
                rtr_s[i + 1, j].put(zpkt)
                rtr_n[i, j].put(zpkt)
                cr_e[i, j].put(zcr)
                cr_w[i, j + 1].put(zcr)
                cr_s[i, j].put(zcr)
                cr_n[i + 1, j].put(zcr)
            # t=0 egress prime
            rtr_e[i, j + 1].put(oe_r)
            rtr_w[i, j].put(ow_r)
            rtr_s[i + 1, j].put(os_r)
            rtr_n[i, j].put(on_r)
            cr_e[i, j].put(cre_r)
            cr_w[i, j + 1].put(crw_r)
            cr_s[i, j].put(crs_r)
            cr_n[i + 1, j].put(crn_r)

            for t in range(NSTEP):
                p_w: Pkt = rtr_e[i, j].get()
                p_e: Pkt = rtr_w[i, j + 1].get()
                p_n: Pkt = rtr_s[i, j].get()
                p_s: Pkt = rtr_n[i + 1, j].get()
                rcred[0] += cr_e[i, j + 1].get()
                rcred[1] += cr_w[i, j].get()
                rcred[2] += cr_s[i + 1, j].get()
                rcred[3] += cr_n[i, j].get()

                # accept a new core send from the PE only when not already holding
                # (non-blocking: the PE keeps its send until we take it -> no bubbles)
                if csd_pkt[RQ_OFF] == 0:
                    inw, okg = inject[i, j].try_get()
                    if okg == 1 and ((inw >> RQ_OFF) & 1) == 1:
                        csd_pkt = inw & PMASK
                        csd_dir = (inw >> 28) & 3

                # ROUTER RECEIVE
                fin: Pkt[4] = 0
                fin[0] = p_w; fin[1] = p_e; fin[2] = p_n; fin[3] = p_s
                for d in range(4):
                    if fin[d][RQ_OFF] == 1 and rbcnt[d] < BUF_DEPTH:
                        rbuf[d, rbcnt[d]] = fin[d]; rbcnt[d] += 1

                hd: Pkt[4] = 0; hvld: int32[4] = 0; hit: int32[4] = 0; axis: int32[4] = 0
                axis[0] = col_id; axis[1] = col_id; axis[2] = row_id; axis[3] = row_id
                for d in range(4):
                    if rbcnt[d] > 0:
                        hd[d] = rbuf[d, 0]; hvld[d] = 1
                        if hd[d][Ty.bits + 5 : Ty.bits + 9] == axis[d]: hit[d] = 1
                o_crv: Pkt = 0; crv_in: int32 = -1
                if   hit[3] == 1: o_crv = hd[3]; crv_in = 3
                elif hit[2] == 1: o_crv = hd[2]; crv_in = 2
                elif hit[1] == 1: o_crv = hd[1]; crv_in = 1
                elif hit[0] == 1: o_crv = hd[0]; crv_in = 0

                o_out: Pkt[4] = 0; pop: int32[4] = 0; inj_done: int32 = 0
                idir: int32 = -1
                if csd_pkt[RQ_OFF] == 1: idir = 3 - csd_dir
                for o in range(4):
                    if rcred[o] > 0:
                        if idir == o:
                            o_out[o] = csd_pkt; rcred[o] -= 1; inj_done = 1
                        elif hvld[o] == 1 and hit[o] == 0:
                            o_out[o] = hd[o]; rcred[o] -= 1; pop[o] = 1

                # deliver to PE via non-blocking put; only pop from rbuf if accepted
                # (eject FIFO full -> leave the packet in rbuf, retry next cycle)
                if crv_in >= 0:
                    ejok = eject[i, j].try_put(o_crv)
                    if ejok == 1: pop[crv_in] = 1

                ret: int32[4] = 0
                for d in range(4):
                    if pop[d] == 1:
                        for sft in range(BUF_DEPTH - 1):
                            rbuf[d, sft] = rbuf[d, sft + 1]
                        rbcnt[d] -= 1; ret[d] = 1
                cre_r = ret[0]; crw_r = ret[1]; crs_r = ret[2]; crn_r = ret[3]
                oe_r = o_out[0]; ow_r = o_out[1]; os_r = o_out[2]; on_r = o_out[3]
                if inj_done == 1: csd_pkt = 0     # my packet went -> release hold

                # mesh egress
                rtr_e[i, j + 1].put(oe_r)
                rtr_w[i, j].put(ow_r)
                rtr_s[i + 1, j].put(os_r)
                rtr_n[i, j].put(on_r)
                cr_e[i, j].put(cre_r)
                cr_w[i, j + 1].put(crw_r)
                cr_s[i, j].put(crs_r)
                cr_n[i + 1, j].put(crn_r)

        @df.kernel(mapping=[M, N], args=[prime_cfg])
        def pe(pcfg: int32[M, N]):
            # SIMPLIFIED single-cycle compute core (scoreboard/forwarding removed):
            # fetch one instr/cycle, execute, write back / inject / systolic-TX
            # immediately. Keeps the ISA, DRF data-driven validity, and the
            # sys-credit systolic plane. Router talks over eject/inject (non-block).
            i, j = df.get_pid()
            irf: int32[IRF_DEPTH] = 0
            drf: Ty[DRF_DEPTH] = 0
            drf_full: int32[DRF_DEPTH] = 0
            dsmask: int32 = 0
            hold_v: Ty[4, 2] = 0; hold_cnt: UInt(8)[4] = 0
            scred: int32[4] = 0
            txp_v: int32[4] = 0; txp_d: Ty[4] = 0
            sc_r: int32[4] = 2
            txn_r: SYS_W = 0; txs_r: SYS_W = 0; txw_r: SYS_W = 0; txe_r: SYS_W = 0
            cfg_isz: int32 = 0; cfg_itsz: int32 = 0
            fetch_en: UInt(8) = 0; instr_cnt: UInt(8) = 0; iter_cnt: UInt(8) = 0
            condition_reg: UInt(8) = 0
            psd: int32 = 0             # pending inject word (held until router takes it)
            psd_v: int32 = 0
            row_id: int32 = i
            col_id: int32 = j
            zsys: SYS_W = 0
            zcr: int32 = 0
            # prime systolic + sys-credit planes
            for _pt in range(pcfg[i, j] - 1):
                sys_e[i, j + 1].put(zsys)
                sys_w[i, j].put(zsys)
                sys_s[i + 1, j].put(zsys)
                sys_n[i, j].put(zsys)
                scr_e[i, j].put(zcr)
                scr_w[i, j + 1].put(zcr)
                scr_s[i, j].put(zcr)
                scr_n[i + 1, j].put(zcr)
            sys_e[i, j + 1].put(txe_r)
            sys_w[i, j].put(txw_r)
            sys_s[i + 1, j].put(txs_r)
            sys_n[i, j].put(txn_r)
            scr_s[i, j].put(sc_r[0])
            scr_n[i + 1, j].put(sc_r[1])
            scr_e[i, j].put(sc_r[2])
            scr_w[i, j + 1].put(sc_r[3])

            for t in range(NSTEP):
                scred[0] += scr_n[i, j].get()
                scred[1] += scr_s[i + 1, j].get()
                scred[2] += scr_w[i, j].get()
                scred[3] += scr_e[i, j + 1].get()

                # delivered packet from router (non-blocking; bubble if none)
                o_crv, oke = eject[i, j].try_get()
                crv_vld: int32 = 0
                if oke == 1: crv_vld = o_crv[RQ_OFF]
                crv_data: Ty = o_crv[0 : Ty.bits].bitcast()
                crv_addr: int32 = o_crv[Ty.bits : Ty.bits + 4]
                crv_mode: int32 = o_crv[Ty.bits + 4]
                crv_raw:  int32 = o_crv[0 : Ty.bits]

                # systolic RX
                rx_w: SYS_W = sys_e[i, j].get()
                rx_e: SYS_W = sys_w[i, j + 1].get()
                rx_n: SYS_W = sys_s[i, j].get()
                rx_s: SYS_W = sys_n[i + 1, j].get()
                rxv: Ty[4] = 0; rxvld: int32[4] = 0
                rxv[0] = rx_n[1 : 1 + Ty.bits].bitcast(); rxvld[0] = rx_n[0]
                rxv[1] = rx_s[1 : 1 + Ty.bits].bitcast(); rxvld[1] = rx_s[0]
                rxv[2] = rx_w[1 : 1 + Ty.bits].bitcast(); rxvld[2] = rx_w[0]
                rxv[3] = rx_e[1 : 1 + Ty.bits].bitcast(); rxvld[3] = rx_e[0]
                for d in range(4):
                    if rxvld[d] == 1 and hold_cnt[d] < 2:
                        hold_v[d, hold_cnt[d]] = rxv[d]; hold_cnt[d] += 1

                # ---- fetch / decode ----
                pc: int32 = -1
                if fetch_en == 1: pc = instr_cnt
                instr: int32 = 0
                if pc >= 0: instr = irf[pc]
                op: int32 = instr & 0xF
                dst: int32 = (instr >> 4) & 0xF
                s1: int32 = (instr >> 8) & 0xF
                s2: int32 = (instr >> 12) & 0xF
                a: Ty = 0; b: Ty = 0
                if s1 >= 12: a = hold_v[s1 & 3, 0]
                else:        a = drf[s1]
                if s2 >= 12: b = hold_v[s2 & 3, 0]
                else:        b = drf[s2]
                a_vld: int32 = 1; b_vld: int32 = 1
                if s1 >= 12:
                    a_vld = 0
                    if hold_cnt[s1 & 3] > 0: a_vld = 1
                if s2 >= 12:
                    b_vld = 0
                    if hold_cnt[s2 & 3] > 0: b_vld = 1
                if s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1:
                    if drf_full[s1] == 0: a_vld = 0
                if s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1:
                    if drf_full[s2] == 0: b_vld = 0
                binop: int32 = 0
                if op == OP_ADD or op == OP_SUB or op == OP_MULT or op == OP_GEQ or op == OP_LT: binop = 1
                is_cond: int32 = 0
                if op >= OP_CRTR0 and op <= OP_CRTR0 + 3: is_cond = 1
                is_rtr: int32 = 0
                if op >= OP_RTR0 and op <= OP_RTR0 + 3: is_rtr = 1
                send_instr: int32 = 0
                if is_rtr == 1 or is_cond == 1: send_instr = 1

                # ---- grant (single-cycle: only data-driven + send-hold stalls) ----
                grant: int32 = 0
                if pc >= 0: grant = 1
                if pc >= 0 and (a_vld == 0 or (binop == 1 and b_vld == 0)): grant = 0
                if send_instr == 1 and psd_v == 1: grant = 0
                if grant == 1:
                    if instr_cnt == cfg_isz:
                        instr_cnt = 0
                        if iter_cnt == cfg_itsz - 1: fetch_en = 0
                        else: iter_cnt += 1
                    else: instr_cnt += 1

                # consume systolic operands + return sys-credit
                c1: int32 = -1; c2: int32 = -1
                if grant == 1 and s1 >= 12: c1 = s1 & 3
                if grant == 1 and s2 >= 12: c2 = s2 & 3
                if c1 >= 0:
                    hold_v[c1, 0] = hold_v[c1, 1]; hold_cnt[c1] -= 1
                if c2 >= 0 and c2 != c1:
                    hold_v[c2, 0] = hold_v[c2, 1]; hold_cnt[c2] -= 1
                for d in range(4): sc_r[d] = 0
                if c1 >= 0: sc_r[c1] = 1
                if c2 >= 0 and c2 != c1: sc_r[c2] = 1
                if grant == 1 and s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1: drf_full[s1] = 0
                if grant == 1 and s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1: drf_full[s2] = 0

                # ---- execute (single-cycle ALU) ----
                res: Ty = 0
                if op == OP_ADD:    res = a + b
                elif op == OP_SUB:  res = a - b
                elif op == OP_MULT: res = a * b
                elif op == OP_GEQ:
                    if a >= b: res = 1.0
                    else:      res = -1.0
                elif op == OP_LT:
                    if a < b:  res = 1.0
                    else:      res = -1.0
                else:               res = a
                if grant == 1 and op == OP_GEQ:
                    if a >= b: condition_reg = 1
                    else:      condition_reg = 0
                if grant == 1 and op == OP_LT:
                    if a < b:  condition_reg = 1
                    else:      condition_reg = 0

                # ---- dispatch immediately (was retire) ----
                if grant == 1:
                    inj_en: int32 = is_rtr
                    if is_cond == 1 and condition_reg == 1: inj_en = 1
                    if inj_en == 1:
                        pk: Pkt = 0
                        pk[0 : Ty.bits] = res.bitcast()
                        pk[Ty.bits : Ty.bits + 4] = dst
                        pk[Ty.bits + 5 : Ty.bits + 9] = s2
                        pk[RQ_OFF] = 1
                        dr: int32 = op & 3
                        psd = (pk & PMASK) | (dr << 28)
                        psd_v = 1
                    elif dst >= 12:
                        txp_v[dst & 3] = 1
                        txp_d[dst & 3] = res
                    else:
                        if dst < DRF_DEPTH and ((dsmask >> dst) & 1) == 1:
                            if drf_full[dst] == 0:
                                drf[dst] = res; drf_full[dst] = 1
                        else:
                            drf[dst & 7] = res

                # ---- apply delivered packet (config / DRF / systolic inject) ----
                if crv_vld == 1:
                    if crv_mode == 1:
                        if ((crv_addr >> 3) & 1) == 1: irf[crv_addr & 7] = crv_raw
                        elif crv_addr == 0:
                            dsmask = crv_raw & 0xFF
                            cfg_isz = (crv_raw >> 8) & 0x7
                            if ((crv_raw >> 15) & 1) == 1: fetch_en = 1; instr_cnt = 0; iter_cnt = 0
                        elif crv_addr == 1: cfg_itsz = crv_raw & 0xFF
                    elif ((crv_addr >> 2) & 3) == 3:
                        txp_v[crv_addr & 3] = 1
                        txp_d[crv_addr & 3] = crv_data
                    elif crv_addr < DRF_DEPTH and ((dsmask >> crv_addr) & 1) == 1:
                        if drf_full[crv_addr] == 0:
                            drf[crv_addr] = crv_data; drf_full[crv_addr] = 1
                    else:
                        drf[crv_addr] = crv_data

                # ---- systolic TX drain (credit-gated) ----
                txn_r = 0; txs_r = 0; txw_r = 0; txe_r = 0
                if txp_v[0] == 1 and scred[0] > 0:
                    twn: SYS_W = 0
                    twn[0] = 1; twn[1 : 1 + Ty.bits] = txp_d[0].bitcast()
                    txn_r = twn; txp_v[0] = 0; scred[0] -= 1
                if txp_v[1] == 1 and scred[1] > 0:
                    tws: SYS_W = 0
                    tws[0] = 1; tws[1 : 1 + Ty.bits] = txp_d[1].bitcast()
                    txs_r = tws; txp_v[1] = 0; scred[1] -= 1
                if txp_v[2] == 1 and scred[2] > 0:
                    tww: SYS_W = 0
                    tww[0] = 1; tww[1 : 1 + Ty.bits] = txp_d[2].bitcast()
                    txw_r = tww; txp_v[2] = 0; scred[2] -= 1
                if txp_v[3] == 1 and scred[3] > 0:
                    twe: SYS_W = 0
                    twe[0] = 1; twe[1 : 1 + Ty.bits] = txp_d[3].bitcast()
                    txe_r = twe; txp_v[3] = 0; scred[3] -= 1

                # ---- drain pending inject to router (non-blocking; holds if full) ----
                if psd_v == 1:
                    okp = inject[i, j].try_put(psd)
                    if okp == 1: psd_v = 0

                # ---- emit systolic + sys-credit ----
                sys_e[i, j + 1].put(txe_r)
                sys_w[i, j].put(txw_r)
                sys_s[i + 1, j].put(txs_r)
                sys_n[i, j].put(txn_r)
                scr_s[i, j].put(sc_r[0])
                scr_n[i + 1, j].put(sc_r[1])
                scr_e[i, j].put(sc_r[2])
                scr_w[i, j + 1].put(sc_r[3])

        @df.kernel(mapping=[1], args=[in_w, iv_w, prime_cfg])
        def drv_w(din_w: Ty[M, LANELEN], vd_w: int32[M, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[M] = 0
            sp: int32[M] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    sys_e[r, 0].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += scr_e[r, 0].get()
                    w: SYS_W = 0
                    if sp[r] < LANELEN and t >= sp[r]:
                        if vd_w[r, sp[r]] == 0:
                            sp[r] += 1
                        elif dcred[r] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_w[r, sp[r]].bitcast()
                            dcred[r] -= 1
                            sp[r] += 1
                    sys_e[r, 0].put(w)

        @df.kernel(mapping=[1], args=[in_e, iv_e, prime_cfg])
        def drv_e(din_e: Ty[M, LANELEN], vd_e: int32[M, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[M] = 0
            sp: int32[M] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    sys_w[r, N].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += scr_w[r, N].get()
                    w: SYS_W = 0
                    if sp[r] < LANELEN and t >= sp[r]:
                        if vd_e[r, sp[r]] == 0:
                            sp[r] += 1
                        elif dcred[r] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_e[r, sp[r]].bitcast()
                            dcred[r] -= 1
                            sp[r] += 1
                    sys_w[r, N].put(w)

        @df.kernel(mapping=[1], args=[in_n, iv_n, prime_cfg])
        def drv_n(din_n: Ty[N, LANELEN], vd_n: int32[N, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[N] = 0
            sp: int32[N] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    sys_s[0, c].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += scr_s[0, c].get()
                    w: SYS_W = 0
                    if sp[c] < LANELEN and t >= sp[c]:
                        if vd_n[c, sp[c]] == 0:
                            sp[c] += 1
                        elif dcred[c] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_n[c, sp[c]].bitcast()
                            dcred[c] -= 1
                            sp[c] += 1
                    sys_s[0, c].put(w)

        @df.kernel(mapping=[1], args=[in_s, iv_s, prime_cfg])
        def drv_s(din_s: Ty[N, LANELEN], vd_s: int32[N, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[N] = 0
            sp: int32[N] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    sys_n[M, c].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += scr_n[M, c].get()
                    w: SYS_W = 0
                    if sp[c] < LANELEN and t >= sp[c]:
                        if vd_s[c, sp[c]] == 0:
                            sp[c] += 1
                        elif dcred[c] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_s[c, sp[c]].bitcast()
                            dcred[c] -= 1
                            sp[c] += 1
                    sys_n[M, c].put(w)

        @df.kernel(mapping=[1], args=[out_w, prime_cfg])
        def col_w(dout_w: Ty[M, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    scr_w[r, 0].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = 2
                scr_w[r, 0].put(cret[r])
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    w: SYS_W = sys_w[r, 0].get()
                    cret[r] = 0
                    if w[0] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            dout_w[r, k[r]] = w[1 : 1 + Ty.bits].bitcast()
                            k[r] += 1
                    scr_w[r, 0].put(cret[r])

        @df.kernel(mapping=[1], args=[out_e, prime_cfg])
        def col_e(dout_e: Ty[M, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    scr_e[r, N].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = 2
                scr_e[r, N].put(cret[r])
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    w: SYS_W = sys_e[r, N].get()
                    cret[r] = 0
                    if w[0] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            dout_e[r, k[r]] = w[1 : 1 + Ty.bits].bitcast()
                            k[r] += 1
                    scr_e[r, N].put(cret[r])

        @df.kernel(mapping=[1], args=[out_n, prime_cfg])
        def col_n(dout_n: Ty[N, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[N] = 0
            cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    scr_n[0, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = 2
                scr_n[0, c].put(cret[c])
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    w: SYS_W = sys_n[0, c].get()
                    cret[c] = 0
                    if w[0] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            dout_n[c, k[c]] = w[1 : 1 + Ty.bits].bitcast()
                            k[c] += 1
                    scr_n[0, c].put(cret[c])

        @df.kernel(mapping=[1], args=[out_s, prime_cfg])
        def col_s(dout_s: Ty[N, LANELEN], pcfg: int32[M, N]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[N] = 0
            cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    scr_s[M, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = 2
                scr_s[M, c].put(cret[c])
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    w: SYS_W = sys_s[M, c].get()
                    cret[c] = 0
                    if w[0] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            dout_s[c, k[c]] = w[1 : 1 + Ty.bits].bitcast()
                            k[c] += 1
                    scr_s[M, c].put(cret[c])

        @df.kernel(mapping=[1], args=[rin_w, prime_cfg])
        def rdrv_w(rdin_w: int32[M, LANELEN], pcfg: int32[M, N]):
            dcred: int32[M] = 0; sp: int32[M] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    rtr_e[r, 0].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += cr_e[r, 0].get()
                    pw: Pkt = 0
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_w[r, sp[r]]
                        if cand[RQ_OFF] == 0: sp[r] += 1
                        elif dcred[r] > 0: pw = cand; dcred[r] -= 1; sp[r] += 1
                    rtr_e[r, 0].put(pw)

        @df.kernel(mapping=[1], args=[rin_e, prime_cfg])
        def rdrv_e(rdin_e: int32[M, LANELEN], pcfg: int32[M, N]):
            dcred: int32[M] = 0; sp: int32[M] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    rtr_w[r, N].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += cr_w[r, N].get()
                    pw: Pkt = 0
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_e[r, sp[r]]
                        if cand[RQ_OFF] == 0: sp[r] += 1
                        elif dcred[r] > 0: pw = cand; dcred[r] -= 1; sp[r] += 1
                    rtr_w[r, N].put(pw)

        @df.kernel(mapping=[1], args=[rin_n, prime_cfg])
        def rdrv_n(rdin_n: int32[N, LANELEN], pcfg: int32[M, N]):
            dcred: int32[N] = 0; sp: int32[N] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    rtr_s[0, c].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += cr_s[0, c].get()
                    pw: Pkt = 0
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_n[c, sp[c]]
                        if cand[RQ_OFF] == 0: sp[c] += 1
                        elif dcred[c] > 0: pw = cand; dcred[c] -= 1; sp[c] += 1
                    rtr_s[0, c].put(pw)

        @df.kernel(mapping=[1], args=[rin_s, prime_cfg])
        def rdrv_s(rdin_s: int32[N, LANELEN], pcfg: int32[M, N]):
            dcred: int32[N] = 0; sp: int32[N] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    rtr_n[M, c].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += cr_n[M, c].get()
                    pw: Pkt = 0
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_s[c, sp[c]]
                        if cand[RQ_OFF] == 0: sp[c] += 1
                        elif dcred[c] > 0: pw = cand; dcred[c] -= 1; sp[c] += 1
                    rtr_n[M, c].put(pw)

        @df.kernel(mapping=[1], args=[rout_w, prime_cfg])
        def rclc_w(rdout_w: int32[M, LANELEN], pcfg: int32[M, N]):
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    cr_w[r, 0].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = BUF_DEPTH
                cr_w[r, 0].put(cret[r])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    pw: Pkt = rtr_w[r, 0].get()
                    cret[r] = 0
                    if pw[RQ_OFF] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            rdout_w[r, k[r]] = pw & PMASK
                            k[r] += 1
                    cr_w[r, 0].put(cret[r])  # END-PUT: moved after the get
                              
        @df.kernel(mapping=[1], args=[rout_e, prime_cfg])
        def rclc_e(rdout_e: int32[M, LANELEN], pcfg: int32[M, N]):
            k: int32[M] = 0; cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, M) as r:
                    cr_e[r, N].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = BUF_DEPTH
                cr_e[r, N].put(cret[r])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    pw: Pkt = rtr_e[r, N].get()
                    cret[r] = 0
                    if pw[RQ_OFF] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            rdout_e[r, k[r]] = pw & PMASK
                            k[r] += 1
                    cr_e[r, N].put(cret[r])  # END-PUT: moved after the get

        @df.kernel(mapping=[1], args=[rout_n, prime_cfg])
        def rclc_n(rdout_n: int32[N, LANELEN], pcfg: int32[M, N]):
            k: int32[N] = 0; cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    cr_n[0, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = BUF_DEPTH
                cr_n[0, c].put(cret[c])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    pw: Pkt = rtr_n[0, c].get()
                    cret[c] = 0
                    if pw[RQ_OFF] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            rdout_n[c, k[c]] = pw & PMASK
                            k[c] += 1
                    cr_n[0, c].put(cret[c])  # END-PUT: moved after the get

        @df.kernel(mapping=[1], args=[rout_s, prime_cfg])
        def rclc_s(rdout_s: int32[N, LANELEN], pcfg: int32[M, N]):
            k: int32[N] = 0; cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            for _pt in range(pcfg[0, 0] - 1):
                with allo.meta_for(0, N) as c:
                    cr_s[M, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = BUF_DEPTH
                cr_s[M, c].put(cret[c])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    pw: Pkt = rtr_s[M, c].get()
                    cret[c] = 0
                    if pw[RQ_OFF] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            rdout_s[c, k[c]] = pw & PMASK
                            k[c] += 1
                    cr_s[M, c].put(cret[c])  # END-PUT: moved after the get

    return top

# =============================================================================
# EXTENDED region (golden pe_core_extended / fpu_extended.sv) — a VERBATIM copy
# of get_eva_top above, adding DIV (0xA) and SQRT (0xB) to the execute stage.
# Analog of golden's pe_group_extended: every node here is div/sqrt-capable
# (the base region stays untouched = golden pe_group). Column-0-only placement
# is a mesh-instantiation concern (M=N=1 here). SQRT reads operand a (matching
# allo/tests/dataflow/eva_pe_core.py); golden fp16_sqrt_ppl reads operand-2.
# =============================================================================
def get_eva_top_extended(Ty: AlloType = float16):
    # SCOREBOARD RESIZE for the extended FPU: the execute stage now contains
    # fsqrt_32ns_12 (12-cyc sqrt) and hdiv_16ns_8 (8-cyc div). Issue-to-retire
    # depth must cover the LONGEST op so the result ring parks it for its full
    # latency instead of forming a distance-1 recurrence (base was 5=add(4)+1).
    # These shadow the module globals for THIS region only (closure, like Ty);
    # the base get_eva_top keeps 5/8. RESQ_DEPTH must be a power of 2 (masked)
    # and > SB_DEPTH.
    SB_DEPTH, RESQ_DEPTH = 13, 16
    DATA_W = Ty.bits
    ID_W, MODE_W, ADDR_W, RQ_W = 4, 1, 4, 1

    # router packet = {rq,id,mode,addr,data} packed LSB-first into one UInt (data-OPAQUE, raw bits)
    D_OFF  = 0
    A_OFF  = D_OFF + DATA_W
    MD_OFF = A_OFF + ADDR_W
    ID_OFF = MD_OFF + MODE_W
    RQ_OFF = ID_OFF + ID_W
    PKT_W  = RQ_OFF + RQ_W
    # mask for recording a packet into an int32 rout array: Allo's vhls
    # emission SIGN-extends UInt->int32 stores (sim zero-extends; same bug
    # class as Allo_bugs/uint_slice_signed_emission.py) -> force zero-ext
    PMASK  = (1 << PKT_W) - 1 if PKT_W < 32 else 0x7FFFFFFF
    Pkt    = UInt(PKT_W)

    SYS_W  = UInt(1 + DATA_W)                  # bit0 = vld, bits[1:1+DATA_W] = raw data

    @df.region()
    def top_extended(
        in_w: Ty[M, LANELEN], in_e: Ty[M, LANELEN],
        in_n: Ty[N, LANELEN], in_s: Ty[N, LANELEN],
        out_w: Ty[M, LANELEN], out_e: Ty[M, LANELEN],
        out_n: Ty[N, LANELEN], out_s: Ty[N, LANELEN],
        rin_w: int32[M, LANELEN], rin_e: int32[M, LANELEN],
        rin_n: int32[N, LANELEN], rin_s: int32[N, LANELEN],
        rout_w: int32[M, LANELEN], rout_e: int32[M, LANELEN],
        rout_n: int32[N, LANELEN], rout_s: int32[N, LANELEN],
        iv_w: int32[M, LANELEN], iv_e: int32[M, LANELEN],
        iv_n: int32[N, LANELEN], iv_s: int32[N, LANELEN],
    ):
        sys_e: Stream[SYS_W, STREAM_DEPTH][M, N + 1]
        sys_w: Stream[SYS_W, STREAM_DEPTH][M, N + 1]
        sys_s: Stream[SYS_W, STREAM_DEPTH][M + 1, N]
        sys_n: Stream[SYS_W, STREAM_DEPTH][M + 1, N]
        rtr_e: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_w: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_s: Stream[Pkt, STREAM_DEPTH][M + 1, N]
        rtr_n: Stream[Pkt, STREAM_DEPTH][M + 1, N]
        cr_e: Stream[int32, STREAM_DEPTH][M, N + 1]
        cr_w: Stream[int32, STREAM_DEPTH][M, N + 1]
        cr_s: Stream[int32, STREAM_DEPTH][M + 1, N]
        cr_n: Stream[int32, STREAM_DEPTH][M + 1, N]
        # SYS-CREDIT plane (mirrors cr_*): credits for the systolic links
        scr_e: Stream[int32, STREAM_DEPTH][M, N + 1]
        scr_w: Stream[int32, STREAM_DEPTH][M, N + 1]
        scr_s: Stream[int32, STREAM_DEPTH][M + 1, N]
        scr_n: Stream[int32, STREAM_DEPTH][M + 1, N]

        @df.kernel(mapping=[M, N])
        def node():
            i, j = df.get_pid()
            irf: int32[IRF_DEPTH] = 0
            drf: Ty[DRF_DEPTH] = 0
            drf_full: int32[DRF_DEPTH] = 0
            dsmask: int32 = 0
            
            crv_vld: int32 = 0
            crv_data: Ty = 0
            crv_addr: int32 = 0
            crv_mode: int32 = 0
            crv_raw:  int32 = 0
            
            csd_vld: int32 = 0
            csd_pkt: Pkt = 0
            csd_dir: int32 = 0
            row_id: int32 = i
            col_id: int32 = j

            oe_r: Pkt = 0; ow_r: Pkt = 0; on_r: Pkt = 0; os_r: Pkt = 0
            txn_r: SYS_W = 0; txs_r: SYS_W = 0; txw_r: SYS_W = 0; txe_r: SYS_W = 0
            
            # NARROW COUNTERS (2026-07-07): routed CP = hold_cnt(int32, reg[31]!)
            # -> operand-vld -> grant -> consume -> hold_cnt', 22 logic levels
            # @186MHz. These regs are protocol-bounded but Vitis can't prove it
            # -> the narrow declaration asserts it; values never leave the
            # range, so sim + RTL behavior are unchanged.
            hold_v: Ty[4, 2] = 0; hold_cnt: UInt(8)[4] = 0
            
            rbuf:  Pkt[4, BUF_DEPTH] = 0
            rbcnt: UInt(8)[4] = 0
            rcred: UInt(8)[4] = 0

            cre_r: UInt(8) = BUF_DEPTH; crw_r: UInt(8) = BUF_DEPTH
            crs_r: UInt(8) = BUF_DEPTH; crn_r: UInt(8) = BUF_DEPTH

            # SYS-CREDIT state: sender credits per TX dir (dst&3: 0=N,1=S,
            # 2=W,3=E), pending TX word per dir (holds until credited =
            # golden sender-side stall), registered credit returns per RX
            # side (hold d: 0=top,1=btm,2=lft,3=rgt; init 2 = hold capacity)
            scred: int32[4] = 0
            txp_v: int32[4] = 0
            txp_d: Ty[4] = 0
            txp_r: int32[4] = 0
            sc_r: int32[4] = 2

            cfg_isz: int32 = 0             # stays int32: RHS-only, and
            cfg_itsz: int32 = 0            # `cfg_itsz - 1` must be -1 at itsz=0
            fetch_en: UInt(8) = 0
            instr_cnt: UInt(8) = 0         # 0..cfg_isz <= 7
            iter_cnt: UInt(8) = 0          # 0..cfg_itsz-1 <= 254
            condition_reg: UInt(8) = 0

            # scoreboard: sb_* = parallel metadata arrays (no structs in
            # Allo), slot 0 retires this cycle; resq/cmpq = result rings
            sb_v: UInt(8)[SB_DEPTH] = 0;  sb_dst: UInt(8)[SB_DEPTH] = 0
            sb_cmp: UInt(8)[SB_DEPTH] = 0; sb_rtr: UInt(8)[SB_DEPTH] = 0
            sb_inj: UInt(8)[SB_DEPTH] = 0; sb_dir: UInt(8)[SB_DEPTH] = 0
            sb_id: UInt(8)[SB_DEPTH] = 0; sb_rvld: UInt(8)[SB_DEPTH] = 0
            sb_ix: UInt(8)[SB_DEPTH] = 0   # resq index, 0..RESQ_DEPTH-1
            resq: Ty[RESQ_DEPTH] = 0
            cmpq: UInt(8)[RESQ_DEPTH] = 0
            resq_wr: UInt(8) = 0           # ring ptr, masked &(RESQ_DEPTH-1)

            # OPTION-A extra prime: (PRIME_TOKENS-1) NEUTRAL leading tokens
            # per output (0-pkt / 0-word / 0-CREDIT — never re-put the reg
            # values: BUF_DEPTH credits twice = silent packet drops). The
            # whole fabric just sees T-1 no-op cycles first. T=1 -> no code.
            zpkt: Pkt = 0
            zsys: SYS_W = 0
            zcr: int32 = 0
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                rtr_e[i, j + 1].put(zpkt)
                rtr_w[i, j].put(zpkt)
                rtr_s[i + 1, j].put(zpkt)
                rtr_n[i, j].put(zpkt)
                sys_e[i, j + 1].put(zsys)
                sys_w[i, j].put(zsys)
                sys_s[i + 1, j].put(zsys)
                sys_n[i, j].put(zsys)
                cr_e[i, j].put(zcr)
                cr_w[i, j + 1].put(zcr)
                cr_s[i, j].put(zcr)
                cr_n[i + 1, j].put(zcr)
                scr_e[i, j].put(zcr)
                scr_w[i, j + 1].put(zcr)
                scr_s[i, j].put(zcr)
                scr_n[i + 1, j].put(zcr)

            # END-PUT prime (RTL deadlock fix, see bubble_model plan 07-05):
            # emit the t=0 outputs BEFORE the loop (regs hold init values
            # here: bubble pkts/words + BUF_DEPTH credits); the in-loop puts
            # move to the iteration END. Stream word sequences are BIT-
            # IDENTICAL to eva.py; every feedback edge starts with one token
            # so the RTL's read-before-write schedule cannot deadlock.
            rtr_e[i, j + 1].put(oe_r)
            rtr_w[i, j].put(ow_r)
            rtr_s[i + 1, j].put(os_r)
            rtr_n[i, j].put(on_r)
            sys_e[i, j + 1].put(txe_r)
            sys_w[i, j].put(txw_r)
            sys_s[i + 1, j].put(txs_r)
            sys_n[i, j].put(txn_r)
            cr_e[i, j].put(cre_r)
            cr_w[i, j + 1].put(crw_r)
            cr_s[i, j].put(crs_r)
            cr_n[i + 1, j].put(crn_r)
            scr_s[i, j].put(sc_r[0])
            scr_n[i + 1, j].put(sc_r[1])
            scr_e[i, j].put(sc_r[2])
            scr_w[i, j + 1].put(sc_r[3])

            for t in range(NSTEP):
                p_w: Pkt = rtr_e[i, j].get()
                p_e: Pkt = rtr_w[i, j + 1].get()
                p_n: Pkt = rtr_s[i, j].get()
                p_s: Pkt = rtr_n[i + 1, j].get()
                rcred[0] += cr_e[i, j + 1].get()
                rcred[1] += cr_w[i, j].get()
                rcred[2] += cr_s[i + 1, j].get()
                rcred[3] += cr_n[i, j].get()
                scred[0] += scr_n[i, j].get()
                scred[1] += scr_s[i + 1, j].get()
                scred[2] += scr_w[i, j].get()
                scred[3] += scr_e[i, j + 1].get()

                fin: Pkt[4] = 0
                fin[0] = p_w; fin[1] = p_e; fin[2] = p_n; fin[3] = p_s
                for d in range(4):
                    if fin[d][RQ_OFF] == 1 and rbcnt[d] < BUF_DEPTH:
                        rbuf[d, rbcnt[d]] = fin[d]; rbcnt[d] += 1
                        
                hd: Pkt[4] = 0; hvld: int32[4] = 0; hit: int32[4] = 0; axis: int32[4] = 0
                axis[0] = col_id; axis[1] = col_id; axis[2] = row_id; axis[3] = row_id
                for d in range(4):
                    if rbcnt[d] > 0:
                        hd[d] = rbuf[d, 0]; hvld[d] = 1
                        if hd[d][Ty.bits + 5 : Ty.bits + 9] == axis[d]: hit[d] = 1
                o_crv: Pkt = 0; crv_in: int32 = -1
                if   hit[3] == 1: o_crv = hd[3]; crv_in = 3
                elif hit[2] == 1: o_crv = hd[2]; crv_in = 2
                elif hit[1] == 1: o_crv = hd[1]; crv_in = 1
                elif hit[0] == 1: o_crv = hd[0]; crv_in = 0
                
                o_out: Pkt[4] = 0; pop: int32[4] = 0; inj_done: int32 = 0
                idir: int32 = -1
                if csd_pkt[RQ_OFF] == 1: idir = 3 - csd_dir
                for o in range(4):
                    if rcred[o] > 0:
                        if idir == o:
                            o_out[o] = csd_pkt; rcred[o] -= 1; inj_done = 1
                        elif hvld[o] == 1 and hit[o] == 0:
                            o_out[o] = hd[o]; rcred[o] -= 1; pop[o] = 1
                if crv_in >= 0: pop[crv_in] = 1
                
                ret: int32[4] = 0
                for d in range(4):
                    if pop[d] == 1:
                        for sft in range(BUF_DEPTH - 1):
                            rbuf[d, sft] = rbuf[d, sft + 1]
                        rbcnt[d] -= 1; ret[d] = 1
                cre_r = ret[0]; crw_r = ret[1]; crs_r = ret[2]; crn_r = ret[3]
                
                oe_r = o_out[0]; ow_r = o_out[1]; os_r = o_out[2]; on_r = o_out[3]
                if inj_done == 1: csd_pkt = 0
                
                crv_vld = o_crv[RQ_OFF]
                crv_data = o_crv[0 : Ty.bits].bitcast()
                crv_addr = o_crv[Ty.bits : Ty.bits + 4]
                crv_mode = o_crv[Ty.bits + 4]
                crv_raw  = o_crv[0 : Ty.bits]

                rx_w: SYS_W = sys_e[i, j].get()
                rx_e: SYS_W = sys_w[i, j + 1].get()
                rx_n: SYS_W = sys_s[i, j].get()
                rx_s: SYS_W = sys_n[i + 1, j].get()

                rxv: Ty[4] = 0; rxvld: int32[4] = 0
                rxv[0] = rx_n[1 : 1 + Ty.bits].bitcast(); rxvld[0] = rx_n[0]
                rxv[1] = rx_s[1 : 1 + Ty.bits].bitcast(); rxvld[1] = rx_s[0]
                rxv[2] = rx_w[1 : 1 + Ty.bits].bitcast(); rxvld[2] = rx_w[0]
                rxv[3] = rx_e[1 : 1 + Ty.bits].bitcast(); rxvld[3] = rx_e[0]
                for d in range(4):
                    if rxvld[d] == 1 and hold_cnt[d] < 2:
                        hold_v[d, hold_cnt[d]] = rxv[d]; hold_cnt[d] += 1
                   
                # (4b) RETIRE sb[0] BEFORE fetch. SYS-CREDIT: a TX retire
                # whose pending slot is still occupied STALLS (sb frozen, no
                # issue) — the golden pipeline-stall on ungranted sys TX
                retire_ok: int32 = 1
                if sb_v[0] == 1 and sb_rtr[0] == 0 and sb_dst[0] >= 12:
                    if sb_rvld[0] == 1 and txp_v[sb_dst[0] & 3] == 1: retire_ok = 0
                if sb_v[0] == 1 and retire_ok == 1:
                    wb: Ty = resq[sb_ix[0]]
                    if sb_cmp[0] == 1: condition_reg = cmpq[sb_ix[0]]
                    if sb_rtr[0] == 1:
                        if sb_inj[0] == 1 and csd_pkt[RQ_OFF] == 0:
                            csd_pkt[0 : Ty.bits] = wb.bitcast()
                            csd_pkt[Ty.bits : Ty.bits + 4] = sb_dst[0]
                            csd_pkt[Ty.bits + 5 : Ty.bits + 9] = sb_id[0]
                            csd_pkt[RQ_OFF] = sb_rvld[0]
                            csd_dir = sb_dir[0]
                    elif sb_dst[0] >= 12:
                        if sb_rvld[0] == 1:       # bubble result = no-op
                            txp_v[sb_dst[0] & 3] = 1
                            txp_d[sb_dst[0] & 3] = wb
                            txp_r[sb_dst[0] & 3] = 1
                    else:
                        if sb_rvld[0] == 1:
                            if sb_dst[0] < DRF_DEPTH and ((dsmask >> sb_dst[0]) & 1) == 1:
                                if drf_full[sb_dst[0]] == 0:
                                    drf[sb_dst[0]] = wb
                                    drf_full[sb_dst[0]] = 1
                            else:
                                drf[sb_dst[0] & 7] = wb

                pc: int32 = -1
                if fetch_en == 1: pc = instr_cnt
                instr: int32 = 0
                if pc >= 0: instr = irf[pc]
                op: int32 = instr & 0xF
                dst: int32 = (instr >> 4) & 0xF
                s1: int32 = (instr >> 8) & 0xF
                s2: int32 = (instr >> 12) & 0xF
                a: Ty = 0; b: Ty = 0
                if s1 >= 12: a = hold_v[s1 & 3, 0]
                else:        a = drf[s1]
                if s2 >= 12: b = hold_v[s2 & 3, 0]
                else:        b = drf[s2]
                
                a_vld: int32 = 1; b_vld: int32 = 1
                if s1 >= 12:
                    a_vld = 0
                    if hold_cnt[s1 & 3] > 0: a_vld = 1
                if s2 >= 12:
                    b_vld = 0
                    if hold_cnt[s2 & 3] > 0: b_vld = 1
                if s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1:
                    if drf_full[s1] == 0: a_vld = 0
                if s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1:
                    if drf_full[s2] == 0: b_vld = 0
                binop: int32 = 0
                if op == OP_ADD or op == OP_SUB or op == OP_MULT or op == OP_DIV or op == OP_GEQ or op == OP_LT: binop = 1
                # RAW scan over in-flight sb[1..] (sb[0] retired above ->
                # forwarded); cmp_busy gates CRTR until condition_reg final
                raw: int32 = 0; cmp_busy: int32 = 0
                for k in range(SB_DEPTH - 1):
                    if sb_v[k + 1] == 1 and sb_rtr[k + 1] == 0 and sb_dst[k + 1] < 12:
                        if s1 < 12 and (sb_dst[k + 1] & 7) == (s1 & 7): raw = 1
                        if binop == 1 and s2 < 12 and (sb_dst[k + 1] & 7) == (s2 & 7): raw = 1
                    if sb_v[k + 1] == 1 and sb_cmp[k + 1] == 1: cmp_busy = 1
                is_cond: int32 = 0
                if op >= OP_CRTR0 and op <= OP_CRTR0 + 3: is_cond = 1

                grant: int32 = 0
                if pc >= 0: grant = 1
                if pc >= 0 and DATADRIVEN == 1 and (a_vld == 0 or (binop == 1 and b_vld == 0)): grant = 0
                if pc >= 0 and (raw == 1 or (is_cond == 1 and cmp_busy == 1)): grant = 0
                if retire_ok == 0: grant = 0     # sys-credit structural stall
                if grant == 1:
                    if instr_cnt == cfg_isz:
                        instr_cnt = 0
                        if iter_cnt == cfg_itsz - 1: fetch_en = 0
                        else: iter_cnt += 1
                    else: instr_cnt += 1
                    
                c1: int32 = -1; c2: int32 = -1
                if grant == 1 and s1 >= 12: c1 = s1 & 3
                if grant == 1 and s2 >= 12: c2 = s2 & 3
                if c1 >= 0:
                    hold_v[c1, 0] = hold_v[c1, 1]; hold_cnt[c1] -= 1
                if c2 >= 0 and c2 != c1:
                    hold_v[c2, 0] = hold_v[c2, 1]; hold_cnt[c2] -= 1
                # SYS-CREDIT: return a credit per consumed hold entry
                for d in range(4):
                    sc_r[d] = 0
                if c1 >= 0: sc_r[c1] = 1
                if c2 >= 0 and c2 != c1: sc_r[c2] = 1
                    
                if grant == 1 and s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1: drf_full[s1] = 0
                if grant == 1 and s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1: drf_full[s2] = 0

                res: Ty = 0
                if op == OP_ADD:    res = a + b
                elif op == OP_SUB:  res = a - b
                elif op == OP_MULT: res = a * b
                elif op == OP_GEQ:
                    if a >= b: res = 1.0
                    else:      res = -1.0
                elif op == OP_LT:
                    if a < b:  res = 1.0
                    else:      res = -1.0
                else:               res = a         # covers MOV + (non-col-0) DIV/SQRT
                # EXTENDED FPU (golden pe_core_extended) — PHYSICAL COLUMN 0 ONLY.
                # meta_if is a build-time gate on the pid: columns 1..N-1 trace
                # WITHOUT these arms, so no fsqrt/hdiv hardware is instantiated
                # there (true analog of golden's heterogeneous mesh). A DIV/SQRT
                # opcode on a non-col-0 PE falls through to the `else: res = a`
                # (MOV) above. At M=N=1 the only node is col 0, so identical.
                with allo.meta_if(j == 0):
                    sq_in: float32 = 0.0
                    sq_out: float32 = 0.0
                    if op == OP_DIV:  res = a / b    # golden fpu_extended div
                    elif op == OP_SQRT:              # golden fpu_extended sqrt(operand a)
                        sq_in = a                    # fp16 -> fp32 (SqrtOp has no f16 path)
                        sq_out = allo.sqrt(sq_in)
                        res = sq_out                 # fp32 -> fp16
                
                res_vld: int32 = a_vld
                if op == OP_ADD or op == OP_SUB or op == OP_MULT or op == OP_DIV or op == OP_GEQ or op == OP_LT:
                    res_vld = a_vld * b_vld
                if grant == 0: res_vld = 0
                # ISSUE: result -> ring, metadata -> sb tail; dispatch
                # happens at retire (4b). CRTR reads condition_reg HERE -
                # final, because cmp_busy held the PC until compares retired.
                is_rtr: int32 = 0
                if op >= OP_RTR0 and op <= OP_RTR0 + 3: is_rtr = 1
                # shift the scoreboard down (FROZEN during a retire stall);
                # tail defaults to a bubble
                if retire_ok == 1:
                    for k in range(SB_DEPTH - 1):
                        sb_v[k] = sb_v[k + 1];     sb_dst[k] = sb_dst[k + 1]
                        sb_cmp[k] = sb_cmp[k + 1]; sb_rtr[k] = sb_rtr[k + 1]
                        sb_inj[k] = sb_inj[k + 1]; sb_dir[k] = sb_dir[k + 1]
                        sb_id[k] = sb_id[k + 1];   sb_rvld[k] = sb_rvld[k + 1]
                        sb_ix[k] = sb_ix[k + 1]
                    sb_v[SB_DEPTH - 1] = 0
                if grant == 1:
                    resq[resq_wr] = res
                    cq: int32 = 0
                    if op == OP_GEQ:
                        if a >= b: cq = 1
                    if op == OP_LT:
                        if a < b: cq = 1
                    cmpq[resq_wr] = cq
                    sb_v[SB_DEPTH - 1] = 1
                    sb_dst[SB_DEPTH - 1] = dst
                    sb_ix[SB_DEPTH - 1] = resq_wr
                    sb_cmp[SB_DEPTH - 1] = 0
                    if op == OP_GEQ or op == OP_LT: sb_cmp[SB_DEPTH - 1] = 1
                    rtrf: int32 = is_rtr
                    if is_cond == 1: rtrf = 1
                    sb_rtr[SB_DEPTH - 1] = rtrf
                    inj: int32 = is_rtr
                    if is_cond == 1 and condition_reg == 1: inj = 1
                    sb_inj[SB_DEPTH - 1] = inj
                    sb_dir[SB_DEPTH - 1] = op & 3
                    sb_id[SB_DEPTH - 1] = s2
                    sb_rvld[SB_DEPTH - 1] = res_vld
                    resq_wr = (resq_wr + 1) & (RESQ_DEPTH - 1)

                # SYS-CREDIT drain: emit a pending word only when credited
                # (registered golden gt); otherwise a bubble goes out
                txn_r = 0; txs_r = 0; txw_r = 0; txe_r = 0
                if txp_v[0] == 1 and scred[0] > 0:
                    twn: SYS_W = 0
                    twn[0] = 1
                    twn[1 : 1 + Ty.bits] = txp_d[0].bitcast()
                    txn_r = twn; txp_v[0] = 0; scred[0] -= 1
                if txp_v[1] == 1 and scred[1] > 0:
                    tws: SYS_W = 0
                    tws[0] = 1
                    tws[1 : 1 + Ty.bits] = txp_d[1].bitcast()
                    txs_r = tws; txp_v[1] = 0; scred[1] -= 1
                if txp_v[2] == 1 and scred[2] > 0:
                    tww: SYS_W = 0
                    tww[0] = 1
                    tww[1 : 1 + Ty.bits] = txp_d[2].bitcast()
                    txw_r = tww; txp_v[2] = 0; scred[2] -= 1
                if txp_v[3] == 1 and scred[3] > 0:
                    twe: SYS_W = 0
                    twe[0] = 1
                    twe[1 : 1 + Ty.bits] = txp_d[3].bitcast()
                    txe_r = twe; txp_v[3] = 0; scred[3] -= 1
                if crv_vld == 1:
                    if crv_mode == 1:
                        # & 1: signedness-immune (Allo emits UInt slices as SIGNED
                        # intN; bare ==1 compares -1==1 when the top bit is set)
                        if ((crv_addr >> 3) & 1) == 1: irf[crv_addr & 7] = crv_raw
                        elif crv_addr == 0:
                            dsmask = crv_raw & 0xFF
                            cfg_isz = (crv_raw >> 8) & 0x7
                            if ((crv_raw >> 15) & 1) == 1: fetch_en = 1; instr_cnt = 0; iter_cnt = 0
                        elif crv_addr == 1: cfg_itsz = crv_raw & 0xFF
                    elif ((crv_addr >> 2) & 3) == 3:
                        # ROUTER DATA pkt to sys addr 0xC..0xF writes the SYSTOLIC TX
                        # (golden pe_core.sv:353; LDL `LMOV systop(0)` injects the systolic
                        # stream this way). dir = crv_addr&3: 0=N,1=S,2=W,3=E. Credit-gated
                        # via txp (drains next cycle) = registered equivalent of the golden reg.
                        txp_v[crv_addr & 3] = 1
                        txp_d[crv_addr & 3] = crv_data
                        txp_r[crv_addr & 3] = 1
                    elif crv_addr < DRF_DEPTH and ((dsmask >> crv_addr) & 1) == 1:
                        if drf_full[crv_addr] == 0:
                            drf[crv_addr] = crv_data; drf_full[crv_addr] = 1
                    else:
                        drf[crv_addr] = crv_data

                # END-PUT prime: the 12 puts, moved verbatim from the loop
                # top — regs now hold this iteration's results, i.e. exactly
                # what the old scheme emitted at the NEXT iteration's top
                rtr_e[i, j + 1].put(oe_r)
                rtr_w[i, j].put(ow_r)
                rtr_s[i + 1, j].put(os_r)
                rtr_n[i, j].put(on_r)
                sys_e[i, j + 1].put(txe_r)
                sys_w[i, j].put(txw_r)
                sys_s[i + 1, j].put(txs_r)
                sys_n[i, j].put(txn_r)
                cr_e[i, j].put(cre_r)
                cr_w[i, j + 1].put(crw_r)
                cr_s[i, j].put(crs_r)
                cr_n[i + 1, j].put(crn_r)
                scr_s[i, j].put(sc_r[0])
                scr_n[i + 1, j].put(sc_r[1])
                scr_e[i, j].put(sc_r[2])
                scr_w[i, j + 1].put(sc_r[3])

        @df.kernel(mapping=[1], args=[in_w, iv_w])
        def drv_w(din_w: Ty[M, LANELEN], vd_w: int32[M, LANELEN]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[M] = 0
            sp: int32[M] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    sys_e[r, 0].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += scr_e[r, 0].get()
                    w: SYS_W = 0
                    if sp[r] < LANELEN and t >= sp[r]:
                        if vd_w[r, sp[r]] == 0:
                            sp[r] += 1
                        elif dcred[r] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_w[r, sp[r]].bitcast()
                            dcred[r] -= 1
                            sp[r] += 1
                    sys_e[r, 0].put(w)

        @df.kernel(mapping=[1], args=[in_e, iv_e])
        def drv_e(din_e: Ty[M, LANELEN], vd_e: int32[M, LANELEN]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[M] = 0
            sp: int32[M] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    sys_w[r, N].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += scr_w[r, N].get()
                    w: SYS_W = 0
                    if sp[r] < LANELEN and t >= sp[r]:
                        if vd_e[r, sp[r]] == 0:
                            sp[r] += 1
                        elif dcred[r] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_e[r, sp[r]].bitcast()
                            dcred[r] -= 1
                            sp[r] += 1
                    sys_w[r, N].put(w)

        @df.kernel(mapping=[1], args=[in_n, iv_n])
        def drv_n(din_n: Ty[N, LANELEN], vd_n: int32[N, LANELEN]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[N] = 0
            sp: int32[N] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    sys_s[0, c].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += scr_s[0, c].get()
                    w: SYS_W = 0
                    if sp[c] < LANELEN and t >= sp[c]:
                        if vd_n[c, sp[c]] == 0:
                            sp[c] += 1
                        elif dcred[c] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_n[c, sp[c]].bitcast()
                            dcred[c] -= 1
                            sp[c] += 1
                    sys_s[0, c].put(w)

        @df.kernel(mapping=[1], args=[in_s, iv_s])
        def drv_s(din_s: Ty[N, LANELEN], vd_s: int32[N, LANELEN]):
            # SYS-CREDIT driver = golden tb: hold each value until credited,
            # never send before its original slot cycle (eligibility)
            dcred: int32[N] = 0
            sp: int32[N] = 0
            zw: SYS_W = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    sys_n[M, c].put(zw)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += scr_n[M, c].get()
                    w: SYS_W = 0
                    if sp[c] < LANELEN and t >= sp[c]:
                        if vd_s[c, sp[c]] == 0:
                            sp[c] += 1
                        elif dcred[c] > 0:
                            w[0] = 1
                            w[1 : 1 + Ty.bits] = din_s[c, sp[c]].bitcast()
                            dcred[c] -= 1
                            sp[c] += 1
                    sys_n[M, c].put(w)

        @df.kernel(mapping=[1], args=[out_w])
        def col_w(dout_w: Ty[M, LANELEN]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    scr_w[r, 0].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = 2
                scr_w[r, 0].put(cret[r])
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    w: SYS_W = sys_w[r, 0].get()
                    cret[r] = 0
                    if w[0] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            dout_w[r, k[r]] = w[1 : 1 + Ty.bits].bitcast()
                            k[r] += 1
                    scr_w[r, 0].put(cret[r])

        @df.kernel(mapping=[1], args=[out_e])
        def col_e(dout_e: Ty[M, LANELEN]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    scr_e[r, N].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = 2
                scr_e[r, N].put(cret[r])
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    w: SYS_W = sys_e[r, N].get()
                    cret[r] = 0
                    if w[0] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            dout_e[r, k[r]] = w[1 : 1 + Ty.bits].bitcast()
                            k[r] += 1
                    scr_e[r, N].put(cret[r])

        @df.kernel(mapping=[1], args=[out_n])
        def col_n(dout_n: Ty[N, LANELEN]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[N] = 0
            cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    scr_n[0, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = 2
                scr_n[0, c].put(cret[c])
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    w: SYS_W = sys_n[0, c].get()
                    cret[c] = 0
                    if w[0] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            dout_n[c, k[c]] = w[1 : 1 + Ty.bits].bitcast()
                            k[c] += 1
                    scr_n[0, c].put(cret[c])

        @df.kernel(mapping=[1], args=[out_s])
        def col_s(dout_s: Ty[N, LANELEN]):
            # SYS-CREDIT collector: consumes every word, returns a credit per
            # real word (rclc pattern; pre-put 2 = advertised hold capacity)
            k: int32[N] = 0
            cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    scr_s[M, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = 2
                scr_s[M, c].put(cret[c])
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    w: SYS_W = sys_s[M, c].get()
                    cret[c] = 0
                    if w[0] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            dout_s[c, k[c]] = w[1 : 1 + Ty.bits].bitcast()
                            k[c] += 1
                    scr_s[M, c].put(cret[c])

        @df.kernel(mapping=[1], args=[rin_w])
        def rdrv_w(rdin_w: int32[M, LANELEN]):
            dcred: int32[M] = 0; sp: int32[M] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    rtr_e[r, 0].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += cr_e[r, 0].get()
                    pw: Pkt = 0
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_w[r, sp[r]]
                        if cand[RQ_OFF] == 0: sp[r] += 1
                        elif dcred[r] > 0: pw = cand; dcred[r] -= 1; sp[r] += 1
                    rtr_e[r, 0].put(pw)

        @df.kernel(mapping=[1], args=[rin_e])
        def rdrv_e(rdin_e: int32[M, LANELEN]):
            dcred: int32[M] = 0; sp: int32[M] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    rtr_w[r, N].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    dcred[r] += cr_w[r, N].get()
                    pw: Pkt = 0
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_e[r, sp[r]]
                        if cand[RQ_OFF] == 0: sp[r] += 1
                        elif dcred[r] > 0: pw = cand; dcred[r] -= 1; sp[r] += 1
                    rtr_w[r, N].put(pw)

        @df.kernel(mapping=[1], args=[rin_n])
        def rdrv_n(rdin_n: int32[N, LANELEN]):
            dcred: int32[N] = 0; sp: int32[N] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    rtr_s[0, c].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += cr_s[0, c].get()
                    pw: Pkt = 0
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_n[c, sp[c]]
                        if cand[RQ_OFF] == 0: sp[c] += 1
                        elif dcred[c] > 0: pw = cand; dcred[c] -= 1; sp[c] += 1
                    rtr_s[0, c].put(pw)

        @df.kernel(mapping=[1], args=[rin_s])
        def rdrv_s(rdin_s: int32[N, LANELEN]):
            dcred: int32[N] = 0; sp: int32[N] = 0
            zp: Pkt = 0                         # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    rtr_n[M, c].put(zp)
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    dcred[c] += cr_n[M, c].get()
                    pw: Pkt = 0
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_s[c, sp[c]]
                        if cand[RQ_OFF] == 0: sp[c] += 1
                        elif dcred[c] > 0: pw = cand; dcred[c] -= 1; sp[c] += 1
                    rtr_n[M, c].put(pw)

        @df.kernel(mapping=[1], args=[rout_w])
        def rclc_w(rdout_w: int32[M, LANELEN]):
            k: int32[M] = 0
            cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    cr_w[r, 0].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = BUF_DEPTH
                cr_w[r, 0].put(cret[r])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    pw: Pkt = rtr_w[r, 0].get()
                    cret[r] = 0
                    if pw[RQ_OFF] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            rdout_w[r, k[r]] = pw & PMASK
                            k[r] += 1
                    cr_w[r, 0].put(cret[r])  # END-PUT: moved after the get
                              
        @df.kernel(mapping=[1], args=[rout_e])
        def rclc_e(rdout_e: int32[M, LANELEN]):
            k: int32[M] = 0; cret: int32[M] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, M) as r:
                    cr_e[r, N].put(zc)
            with allo.meta_for(0, M) as r:
                cret[r] = BUF_DEPTH
                cr_e[r, N].put(cret[r])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, M) as r:
                    pw: Pkt = rtr_e[r, N].get()
                    cret[r] = 0
                    if pw[RQ_OFF] == 1:
                        cret[r] = 1
                        if k[r] < LANELEN:
                            rdout_e[r, k[r]] = pw & PMASK
                            k[r] += 1
                    cr_e[r, N].put(cret[r])  # END-PUT: moved after the get

        @df.kernel(mapping=[1], args=[rout_n])
        def rclc_n(rdout_n: int32[N, LANELEN]):
            k: int32[N] = 0; cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    cr_n[0, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = BUF_DEPTH
                cr_n[0, c].put(cret[c])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    pw: Pkt = rtr_n[0, c].get()
                    cret[c] = 0
                    if pw[RQ_OFF] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            rdout_n[c, k[c]] = pw & PMASK
                            k[c] += 1
                    cr_n[0, c].put(cret[c])  # END-PUT: moved after the get

        @df.kernel(mapping=[1], args=[rout_s])
        def rclc_s(rdout_s: int32[N, LANELEN]):
            k: int32[N] = 0; cret: int32[N] = 0
            zc: int32 = 0                       # Option-A neutral prime
            with allo.meta_for(0, PRIME_TOKENS - 1) as _pt:
                with allo.meta_for(0, N) as c:
                    cr_s[M, c].put(zc)
            with allo.meta_for(0, N) as c:
                cret[c] = BUF_DEPTH
                cr_s[M, c].put(cret[c])      # END-PUT prime: t=0 credit
            for t in range(NSTEP):
                with allo.meta_for(0, N) as c:
                    pw: Pkt = rtr_s[M, c].get()
                    cret[c] = 0
                    if pw[RQ_OFF] == 1:
                        cret[c] = 1
                        if k[c] < LANELEN:
                            rdout_s[c, k[c]] = pw & PMASK
                            k[c] += 1
                    cr_s[M, c].put(cret[c])  # END-PUT: moved after the get

    return top_extended

def run_eva(mod, ins, ivs, outs, rins, routs):
    """Invoke a built EVA module with args in the module's TRUE (discovery) order.

    prog/win/win1/dsync are GONE -> program + data are loaded via router packets in rins (rin_s).
    Each systolic driver declares args=[in_X, iv_X], so Allo discovers input+mask as a PAIR:
        (in_w,iv_w), (in_e,iv_e), (in_n,iv_n), (in_s,iv_s),
        out_w, out_e, out_n, out_s, rin_w..rin_s, rout_w..rout_s
      ins  = [in_w, in_e, in_n, in_s]      ivs  = [iv_w, iv_e, iv_n, iv_s]   (int32 masks)
      outs = [out_w, out_e, out_n, out_s]  rins/routs = [*_w, *_e, *_n, *_s] (int32 packets)
    """
    iw, ie, in_, is_ = ins
    vw, ve, vn, vs = ivs
    mod(iw, vw, ie, ve, in_, vn, is_, vs, *outs, *rins, *routs)

def get_scheduled_eva(Ty: AlloType = float16, pipeline_node=False, partition_rf=False, extended=False):
    # extended=True selects the div/sqrt region (get_eva_top_extended); the
    # schedule (partition + pipeline) is identical — same kernel/buffer names.
    s = df.customize(get_eva_top_extended(Ty) if extended else get_eva_top(Ty))
    if partition_rf:
        for i in range(M):
            for j in range(N):
                for buf in ("irf", "drf", "drf_full", "rbuf", "rbcnt", "rcred", "hold_v", "hold_cnt",#,
                            # scoreboard state must be registers, not RAMs
                            "resq", "cmpq", "sb_v", "sb_dst", "sb_cmp", "sb_rtr",
                            "sb_inj", "sb_dir", "sb_id", "sb_rvld", "sb_ix", "sb_long",
                            "scred", "txp_v", "txp_d", "txp_r", "sc_r"):
                    s.partition(f"node_{i}_{j}:{buf}")
    if pipeline_node:
        for i in range(M):
            for j in range(N):
                s.pipeline(f"node_{i}_{j}:t", initiation_interval=1)
    return s

if __name__ == "__main__":
    # SCOREBOARD + SYS-CREDIT lossless (II=1 w/ pragmas).  Params (module-level): M=N=1, NSTEP=10, Ty=float16.
    # Run:  python eva_sb_syscredit.py   then   cd prj_eva_sb_syscredit && vitis_hls -f run.tcl
    import os, re, sys
    # `python eva_sb_syscredit.py extended` -> div/sqrt region into a SEPARATE
    # project; plain run is byte-identical to before (base region).
    EXT = len(sys.argv) > 1 and sys.argv[1] == "extended"
    EXT = True
    pname = "4x4test_all_on_1407fin_prj_eva_sb_syscredit_extended" if EXT else "1307f_prj_eva_sb_syscredit"
    P = os.path.join(os.path.dirname(os.path.abspath(__file__)), pname)
    s = get_scheduled_eva(float16, pipeline_node=True, partition_rf=True, extended=False)
    s.build(target="vhls", mode="csyn", project=P)
    kp = os.path.join(P, "kernel.cpp"); src = open(kp).read()
    src = re.sub(r"(union \{ (?:uint16_t from; half to|half from; uint16_t to);\} _converter\w*);", r"\1 = {};", src)
    #II=1 dependence pragmas — Allo has NO primitive for these; comment
    # this block OUT for the II=3 "pure Allo" build (see PARAMS.md).
    for ring in ("resq", "cmpq"):
        src = src.replace("#pragma HLS array_partition variable=%s complete dim=1" % ring,
            "#pragma HLS array_partition variable=%s complete dim=1\n#pragma HLS dependence variable=%s type=inter dependent=false" % (ring, ring))
    for scal in ("res", "wb"):
        src = src.replace("half %s;" % scal,
            "half %s;\n#pragma HLS dependence variable=%s type=inter dependent=false" % (scal, scal))
    open(kp, "w").write(src)
    print("generated HLS project at", P)
