"""EVA blocks as plain functions. Connections are agent-generated.

  router(...) : routing compute (buffer, XY-hit, arbitrate, credit) on abstract ports
  pe(...)     : datapath compute (fetch/decode/ALU/regfiles/sequencer) on abstract ports
"""
from allo.ir.types import float16, int32, UInt

Ty = float16
DATA_W = Ty.bits
RQ_OFF = DATA_W + 4 + 1 + 4          # data(16) addr(4) mode(1) id(4) -> rq bit
PKT_W = RQ_OFF + 1
PMASK = (1 << PKT_W) - 1
Pkt = UInt(PKT_W)
SYS_W = UInt(1 + DATA_W)

BUF_DEPTH = 2
IRF_DEPTH, DRF_DEPTH = 8, 8
OP_ADD, OP_SUB, OP_MULT, OP_MOV = 0x0, 0x1, 0x2, 0x3
OP_RTR0 = 0x4
OP_GEQ, OP_LT = 0x8, 0x9
OP_CRTR0 = 0xC
SEND_ROUTER, SEND_SYS = 0, 1


def eva_router(
    row_id: int32, col_id: int32,
    pkt_in: Pkt[4], cred_in: int32[4], inject_word: int32, inject_vld: int32,   # in
    inject_ready: int32[1], pkt_out: Pkt[4], cred_out: int32[4],                # out
    eject_word: Pkt[1], eject_vld: int32[1], eject_ready: int32,                # out data/valid, in ready
    rbuf: Pkt[4, BUF_DEPTH], rbcnt: int32[4], rcred: int32[4], csd_pkt: Pkt[1], csd_dir: int32[1],  # state
):
    for d in range(4):
        rcred[d] += cred_in[d]
    inject_ready[0] = 0
    if csd_pkt[0][RQ_OFF] == 0: inject_ready[0] = 1
    if inject_ready[0] == 1 and inject_vld == 1 and ((inject_word >> RQ_OFF) & 1) == 1:
        csd_pkt[0] = inject_word & PMASK
        csd_dir[0] = (inject_word >> 28) & 3
    for d in range(4):
        if pkt_in[d][RQ_OFF] == 1 and rbcnt[d] < BUF_DEPTH:
            rbuf[d, rbcnt[d]] = pkt_in[d]; rbcnt[d] += 1
    hd: Pkt[4] = 0; hvld: int32[4] = 0; hit: int32[4] = 0; axis: int32[4] = 0
    axis[0] = col_id; axis[1] = col_id; axis[2] = row_id; axis[3] = row_id
    for d in range(4):
        if rbcnt[d] > 0:
            hd[d] = rbuf[d, 0]; hvld[d] = 1
            if hd[d][DATA_W + 5 : DATA_W + 9] == axis[d]: hit[d] = 1
    o_crv: Pkt = 0; crv_in: int32 = -1
    if   hit[3] == 1: o_crv = hd[3]; crv_in = 3
    elif hit[2] == 1: o_crv = hd[2]; crv_in = 2
    elif hit[1] == 1: o_crv = hd[1]; crv_in = 1
    elif hit[0] == 1: o_crv = hd[0]; crv_in = 0
    o_out: Pkt[4] = 0; pop: int32[4] = 0; inj_done: int32 = 0
    idir: int32 = -1
    if csd_pkt[0][RQ_OFF] == 1: idir = 3 - csd_dir[0]
    for o in range(4):
        if rcred[o] > 0:
            if idir == o:
                o_out[o] = csd_pkt[0]; rcred[o] -= 1; inj_done = 1
            elif hvld[o] == 1 and hit[o] == 0:
                o_out[o] = hd[o]; rcred[o] -= 1; pop[o] = 1
    eject_vld[0] = 0
    if crv_in >= 0:
        eject_word[0] = o_crv; eject_vld[0] = 1
        if eject_ready == 1: pop[crv_in] = 1
    ret: int32[4] = 0
    for d in range(4):
        if pop[d] == 1:
            for sft in range(BUF_DEPTH - 1):
                rbuf[d, sft] = rbuf[d, sft + 1]
            rbcnt[d] -= 1; ret[d] = 1
    for d in range(4):
        cred_out[d] = ret[d]; pkt_out[d] = o_out[d]
    if inj_done == 1: csd_pkt[0] = 0


def eva_pe(
    row_id: int32, col_id: int32,
    ext_val: Ty[4], ext_vld: int32[4],               # in: operand ports
    wr_valid: int32, wr_mode: int32, wr_addr: int32, wr_data: Ty, wr_raw: int32,  # in: decoded write
    send_ready: int32,                               # in: backpressure
    ext_consume: int32[4],                           # out: consumed operands
    send_valid: int32[1], send_kind: int32[1], send_dir: int32[1], send_data: Ty[1], send_dst: int32[1], send_id: int32[1],  # out: send req
    irf: int32[IRF_DEPTH], drf: Ty[DRF_DEPTH], drf_full: int32[DRF_DEPTH], ctl: int32[7],  # state
):
    dsmask: int32 = ctl[0]
    cfg_isz: int32 = ctl[1]; cfg_itsz: int32 = ctl[2]
    fetch_en: int32 = ctl[3]; instr_cnt: int32 = ctl[4]; iter_cnt: int32 = ctl[5]
    condition_reg: int32 = ctl[6]

    for d in range(4):
        ext_consume[d] = 0
    send_valid[0] = 0; send_kind[0] = 0; send_dir[0] = 0
    send_data[0] = 0; send_dst[0] = 0; send_id[0] = 0

    pc: int32 = -1
    if fetch_en == 1: pc = instr_cnt
    instr: int32 = 0
    if pc >= 0: instr = irf[pc]
    op: int32 = instr & 0xF
    dst: int32 = (instr >> 4) & 0xF
    s1: int32 = (instr >> 8) & 0xF
    s2: int32 = (instr >> 12) & 0xF

    a: Ty = 0; b: Ty = 0
    if s1 >= 12: a = ext_val[s1 & 3]
    else:        a = drf[s1]
    if s2 >= 12: b = ext_val[s2 & 3]
    else:        b = drf[s2]
    a_vld: int32 = 1; b_vld: int32 = 1
    if s1 >= 12: a_vld = ext_vld[s1 & 3]
    if s2 >= 12: b_vld = ext_vld[s2 & 3]
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

    fire_router: int32 = is_rtr
    if is_cond == 1 and condition_reg == 1: fire_router = 1
    fire_sys: int32 = 0
    if is_rtr == 0 and is_cond == 0 and dst >= 12: fire_sys = 1
    any_send: int32 = 0
    if fire_router == 1 or fire_sys == 1: any_send = 1

    grant: int32 = 0
    if pc >= 0: grant = 1
    if pc >= 0 and (a_vld == 0 or (binop == 1 and b_vld == 0)): grant = 0
    if any_send == 1 and send_ready == 0: grant = 0
    if grant == 1:
        if instr_cnt == cfg_isz:
            instr_cnt = 0
            if iter_cnt == cfg_itsz - 1: fetch_en = 0
            else: iter_cnt += 1
        else: instr_cnt += 1

    c1: int32 = -1; c2: int32 = -1
    if grant == 1 and s1 >= 12: c1 = s1 & 3
    if grant == 1 and s2 >= 12: c2 = s2 & 3
    if c1 >= 0: ext_consume[c1] = 1
    if c2 >= 0 and c2 != c1: ext_consume[c2] = 1
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
    else:               res = a
    if grant == 1 and op == OP_GEQ:
        if a >= b: condition_reg = 1
        else:      condition_reg = 0
    if grant == 1 and op == OP_LT:
        if a < b:  condition_reg = 1
        else:      condition_reg = 0

    if grant == 1:
        if fire_router == 1:
            send_valid[0] = 1; send_kind[0] = SEND_ROUTER
            send_dir[0] = op & 3; send_data[0] = res
            send_dst[0] = dst; send_id[0] = s2
        elif fire_sys == 1:
            send_valid[0] = 1; send_kind[0] = SEND_SYS
            send_dir[0] = dst & 3; send_data[0] = res
        elif is_rtr == 0 and is_cond == 0:
            if dst < DRF_DEPTH and ((dsmask >> dst) & 1) == 1:
                if drf_full[dst] == 0:
                    drf[dst] = res; drf_full[dst] = 1
            else:
                drf[dst & 7] = res

    if wr_valid == 1:
        if wr_mode == 1:
            if ((wr_addr >> 3) & 1) == 1: irf[wr_addr & 7] = wr_raw
            elif wr_addr == 0:
                dsmask = wr_raw & 0xFF
                cfg_isz = (wr_raw >> 8) & 0x7
                if ((wr_raw >> 15) & 1) == 1: fetch_en = 1; instr_cnt = 0; iter_cnt = 0
            elif wr_addr == 1: cfg_itsz = wr_raw & 0xFF
        else:
            if wr_addr < DRF_DEPTH and ((dsmask >> wr_addr) & 1) == 1:
                if drf_full[wr_addr] == 0:
                    drf[wr_addr] = wr_data; drf_full[wr_addr] = 1
            else:
                drf[wr_addr] = wr_data

    ctl[0] = dsmask; ctl[1] = cfg_isz; ctl[2] = cfg_itsz
    ctl[3] = fetch_en; ctl[4] = instr_cnt; ctl[5] = iter_cnt
    ctl[6] = condition_reg
