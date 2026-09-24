import warnings, sys, os
warnings.simplefilter('ignore')
import numpy as np, allo.dataflow as df
from allo.ir.types import float16
# workloads live in Allo/EVA/tests/, the design (eva.py) one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eva

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# eva_workloads.py — load real workloads onto the EVA chip (eva.py) and verify them.
#
# A "program" for EVA is just the array tuple the chip runs: (prog instructions, win/win1 twiddle
# preloads, systolic input buffers + valid masks, output buffers, router perimeter buffers). Each
# loader below sizes the chip (eva.M/N/IRF_DEPTH/NSTEP/LANELEN) for its workload and returns
#   (args, check)
# where `args` is fed verbatim to eva.run_eva and `check()` reads the (now-populated) output buffers
# and compares against a golden reference. The chip itself (eva.get_eva_top) never changes per
# workload — the load_* functions are pure program generators.
#
#   python eva_workloads.py fft   # 8-point radix-2 DIT FFT on the 8x8 grid   vs numpy.fft
#   python eva_workloads.py mmmr  # router-loaded weight-stationary Y = X @ W  vs numpy
# ═══════════════════════════════════════════════════════════════════════════════════════════════

# ── shared ISA encode helpers (LSB-first: op | dst<<4 | s1<<8 | s2<<12) ──
ADD, SUB, MULT, MOV = eva.OP_ADD, eva.OP_SUB, eva.OP_MULT, eva.OP_MOV
RTR = eva.OP_RTR0                                   # 0x4..0x7 = N(up)/S(down)/W/E router send
SYSTOP, SYSBTM, SYSLFT, SYSRGT = 0xC, 0xD, 0xE, 0xF
enc = lambda op, dst, s1, s2=0: op | (dst << 4) | (s1 << 8) | (s2 << 12)

zi = lambda *s: np.zeros(s, dtype=np.int32)
zf = lambda *s: np.zeros(s, dtype=np.float16)

# ── router packet pack (mirrors eva.py Pkt layout for fp16: data|addr|mode|id|rq, LSB-first) ──
DW = float16.bits                                   # data-field width = 16
def pack_pkt(data, addr, mode, idf, rq=1):
    return (data & ((1 << DW) - 1)) | (addr << DW) | (mode << (DW + 4)) | (idf << (DW + 5)) | (rq << (DW + 9))

def load_prog_packets(prog, M, N, data=(), cfg=None):
    """Pack each node's IRF (and optional DRF data) as router packets from the SOUTH edge (golden
    t0_rtr_btm): packet id=row rides NORTH up its column, delivers to node(row,col), writes the RF.
    data = list of (slot, vals[M,N]) fp16 arrays -> appended as mode=0 DRF writes. (IRF/DRF <= 8.)
    cfg = (kernel_len, iter_size, sync_mask) or None: per node append config_reg_1 (iter_size) then
    config_reg_0 (start-bit | instr_size | sync_mask) LAST, so the PE only fires after it's loaded."""
    # prog is indexed [row, col, slot] -> instruction word for each node's IRF slot
    # MxN: mesh
    # data: optional list of (slot, vals[M,N]) fp16 arrays to preload into DRF
    # cfg: per-node config-register values 
    rin_s = zi(N, eva.LANELEN)                       # south router-driver buffers, one lane per column, LANELEN packets deep. rin_s[j]: ordered packet stream for column j
    for j in range(N): # one column at a time
        k = 0                                        # contiguous source-pointer fill for column j. all of node 0's packets, then node 1's. packets for different rows share the lane back-to-back
        for i in range(M):                           # target rows up this column
            for s in range(eva.IRF_DEPTH):           # one packet per IRF slot (addr 8+s -> irf[s])
                rin_s[j, k] = pack_pkt(int(prog[i, j, s]), 8 + s, 1, i); k += 1
                # one packet per instruction slot. pack_pkt(payload, addr, mode, id)
                # payload = prog[i,j,s] - instruction word for node (i,j), slot s
                # addr -> slot s lands in irf[s]
                # mode = 1: register-file write mode (for IRF, config regs)
                # id = i -> deliver to row i
            for (slot, vals) in data:                # mode=0 DRF writes: addr=slot, data=fp16 bits
                rin_s[j, k] = pack_pkt(int(np.float16(vals[i, j]).view(np.uint16)), slot, 0, i); k += 1
                # for each (slot, vals): inject one DRF write for this node
                # np.float16(...) takes fp16 value and reinterprets its bit pattern as raw 16-bit integer (bitcast)
            if cfg is not None:                      # config regs (mode=1): iter_size, then START packet LAST
                klen, itsz, smask = cfg(i, j) if callable(cfg) else cfg   # per-node cfg (FFT: klen/smask vary) or uniform
                # config_reg_1 = iter_size, at addr=1, mode=1. Masked to 8 bits
                rin_s[j, k] = pack_pkt(itsz & 0xFF, 1, 1, i); k += 1                          # config_reg_1 = iter_size
                # Assemble the config_reg_0 word as bitfield
                # bit 15 = start/enable bit -> turn PE on
                # bits 10:8 = instr_size = klen -1 (a length-klen kernel is encoded as klen-1, so 0 means one instruction; 3 bits via & 0x7)
                start = (1 << 15) | (((klen - 1) & 0x7) << 8) | (smask & 0xFF)                # bit15 start | instr_size | sync
                rin_s[j, k] = pack_pkt(start, 0, 1, i); k += 1                                # config_reg_0 (fires fetch_en)
                # writing it fligs fetch_en -> PE begins fetching/executing immediately
                # Doing it last guarantees the IRF, DRF, iter_size are all  in place before node can fire
    return rin_s # return the (N, LANELEN) stream, ready to drive into south router ports


def _bundle(prog, win, win1, in_w, ivw, M, N, L, out_e=None, out_s=None):
    """Pack the canonical run_eva arg set; west systolic carries data, all router perimeter idle."""
    # test helper
    # assemble the big, positional argument set that eva.run_eva demands for the case where a workload
    # feeds data in from the west systolic edge and leaves every other port idle
    # ins: systolic data inputs
    # ivs: systolic valid masks
    # outs: systolic data outputs
    # rins: router packet inputs (drivers)
    # routs: router packet outputs (collectors)
    # prog/win/win1: for program + twiddle preloads
    # defaults factory that builds full 6-list x 4-direction run_eva set with west-fed systolic data, idle router
    # each workload only overrides the ports it actually drives/reads
    out_e = zf(M, L) if out_e is None else out_e
    out_s = zf(N, L) if out_s is None else out_s
    return {
        "prog": prog, "win": win, "win1": win1,
        "ins":  [in_w, zf(M, L), zf(N, L), zf(N, L)],
        "ivs":  [ivw, zi(M, L), zi(N, L), zi(N, L)],
        "outs": [zf(M, L), out_e, zf(N, L), out_s],
        "rins": [zi(M, L), zi(M, L), zi(N, L), zi(N, L)],
        "routs":[zi(M, L), zi(M, L), zi(N, L), zi(N, L)],
    }


# ─────────────────────────────────────── FFT (8-point) ───────────────────────────────────────
# row2lane=real PE, row2lane+1=imag PE
# pair is vertically adjacent -> talk over the N/S systolic links
# each PE has its twiddle W=Wr#jWi preloaded into r0/r1
def _butterfly(imag):
    # FFT program generators -> returns list of encoded instructions that gets stamped into one
    # PE's IRF
    # 0-7: PE's general/DRF registers. r7 = router receive register
    # one radix-2 butterfly with complex twiddle
    # DIT butterfly computes X[k]=A+WxB and X[k+N/2]=A-WB
    # real and imag PE each own one component of the data and exchange the cross-term of complex multiply
    # over vertical link
    # mid: two PEs cooperate to form true complex product
    mid = [enc(MOV, SYSTOP, 5), enc(ADD, 6, 4, SYSTOP)] if imag else \
          [enc(MOV, SYSBTM, 5), enc(SUB, 6, 4, SYSBTM)]
    return [enc(MOV, 2, SYSLFT), enc(MOV, 3, SYSLFT), enc(MULT, 4, 0, 3), enc(MULT, 5, 1, 3)] + mid + \
           [enc(ADD, SYSRGT, 2, 6), enc(SUB, SYSRGT, 2, 6)]

def _shuffle(row, stride, D, nop):
    # FFT program generators -> returns list of encoded instructions that gets stamped into one
    # PE's IRF
    # between butterfly stages, DIT needs to reorder data across rows stride apart
    # uses router network to swap one element between the two rows of stride-pair
    upper = (row // stride) % 2 == 0 # top row of its stride-pair
    send = enc(RTR + 1, 7, 1, row + stride) if upper else enc(RTR + 0, 7, 0, row - stride)  # DMOV / UMOV
    # upper node: send  r1 down to node row+stride, then recv (keeps its own r0)
    # lower node: send r0 up to node row-stride, then recv (keeps its own r1)
    recv = enc(MOV, 1, 7) if upper else enc(MOV, 0, 7)
    return [enc(MOV, 0, SYSLFT), enc(MOV, 1, SYSLFT), send] + [nop] * D + \
           [recv, enc(MOV, SYSRGT, 0), enc(MOV, SYSRGT, 1)]
    # nop*D is a hand-tuned delay that must cover the router's delivery latency, so recv doesn't read r7 before the partner's packet has arrived


#In short: _butterfly emits the per-PE microcode for a radix-2 complex
#butterfly, using the vertical systolic link to split the complex multiply
#across the real/imag PE pair; _shuffle emits the microcode that swaps one
#element between stride-separated rows over the router, with a NOP delay slot
#to cover NoC latency. Together they're the two building blocks load_fft tiles
#across the 8×8 grid (butterflies on columns 0/2/4, shuffles on columns 1/3).


# OUTDATED
def load_fft(x, D1=6, D3=12):
    """8-point radix-2 DIT FFT of complex x[8] on the full 8x8 grid (3 butterfly stages + 2 router
    shuffles + relay cols). Bit-reversed left feed; outputs read off the east edge."""
    M, N = 8, 8
    eva.M, eva.N = M, N
    OFF = [0, 7, 12 + D1, 19 + D1, 24 + D1 + D3]               # col0..4 (compute + 2 shuffles)
    OFF += [OFF[4] + 7, OFF[4] + 8, OFF[4] + 9]                # col5,6,7 relays
    eva.IRF_DEPTH = OFF[7] + 4
    eva.NSTEP = eva.IRF_DEPTH + 2
    eva.LANELEN = eva.NSTEP
    L = eva.LANELEN
    NOP = enc(MOV, 6, 6)                                       # touches only r6

    w8 = [complex(np.cos(2 * np.pi * k / 8), -np.sin(2 * np.pi * k / 8)) for k in range(4)]
    def twid(col, lane):
        if col == 0: return 1 + 0j
        if col == 2: return (1 + 0j) if lane % 2 == 0 else -1j
        return w8[lane]
    relay = [enc(MOV, SYSRGT, SYSLFT), enc(MOV, SYSRGT, SYSLFT)]

    prog = zi(M, N, eva.IRF_DEPTH); prog[:] = NOP
    win, win1 = zf(M, N), zf(M, N)
    for col in range(N):
        for row in range(M):
            lane, imag = row // 2, (row % 2 == 1)
            if col in (0, 2, 4):
                seq = _butterfly(imag)
                W = twid(col, lane)
                win[row, col], win1[row, col] = np.float16(W.real), np.float16(W.imag)
            elif col in (1, 3):
                seq = _shuffle(row, 2 if col == 1 else 4, D1 if col == 1 else D3, NOP)
            else:
                seq = relay
            for k, ins in enumerate(seq):
                prog[row, col, OFF[col] + k] = ins

    br = [0, 4, 2, 6, 1, 5, 3, 7]                              # bit-reversed input order
    in_w, ivw = zf(M, L), zi(M, L)
    for lane in range(4):
        p0, p1 = x[br[2 * lane]], x[br[2 * lane + 1]]
        in_w[2 * lane, 0:2] = [np.float16(p0.real), np.float16(p1.real)]
        in_w[2 * lane + 1, 0:2] = [np.float16(p0.imag), np.float16(p1.imag)]
        ivw[2 * lane, 0:2] = 1; ivw[2 * lane + 1, 0:2] = 1

    args = _bundle(prog, win, win1, in_w, ivw, M, N, L)
    out_e = args["outs"][1]

    def check():
        oe = out_e.astype(np.float32)
        X = np.zeros(8, np.complex64)
        for lane in range(4):                                 # row 2L = [X[L], X[L+4]].re ; 2L+1 = .im
            X[lane]     = oe[2 * lane, 0] + 1j * oe[2 * lane + 1, 0]
            X[lane + 4] = oe[2 * lane, 1] + 1j * oe[2 * lane + 1, 1]
        gold = np.fft.fft(x)
        err = np.abs(X - gold)
        ok = bool(np.all(err < 0.05))
        lines = [f"=== 8-point FFT on 8x8 EVA chip ===  (max |err| = {err.max():.4f})"]
        for k in range(8):
            lines.append(f"  X[{k}]  got {X[k].real:+.4f}{X[k].imag:+.4f}j   "
                         f"gold {gold[k].real:+.4f}{gold[k].imag:+.4f}j   |err|={err[k]:.4f}")
        lines.append(f"FFT-8 {'PASS' if ok else 'FAIL'}")
        return ok, "\n".join(lines)

    return args, check


# ───────────────── FFT (2-point butterfly), program + weights LOADED OVER THE ROUTER ─────────────────
def load_fft2_router(x, PROG_CYCLES=None):
    """2-point radix-2 butterfly with program + weight loaded over the router (mirrors load_mmm_router;
    the router-loaded analog of the direct-arg fft_eva.py). M=2 N=1: row0 = real path, row1 = imag path.
    Complex inputs (in0,in1) stream from the WEST (row0 = real parts, row1 = imag); the two cores
    exchange cross-products VERTICALLY; the two outputs (in0 +/- W*in1) exit EAST. W = 1 - 0j so r0 = 1.0
    is loaded to drf[0] and r1 = w.i stays 0. DATADRIVEN=1: the vertical exchange + per-row start-skew
    (the two START packets deliver on different cycles) resolve via operand stalls, no static schedule."""
    I0, I1, I2, I3 = 0xE23, 0xE33, 0x3042, 0x3152             # golden butterfly (LSB-first), see fft_eva.py
    I4_R, I5_R = 0x5D3, 0xD461                                # real row: send prod.ir DOWN, recv from below
    I4_I, I5_I = 0x5C3, 0xC460                                # imag row: send prod.ii UP,   recv from above
    I6, I7 = 0x62F0, 0x62F1                                   # out0 = in0+prod ; out1 = in0-prod -> EAST
    M, N = 2, 1
    KLEN = 8                                                  # 8-instr butterfly = full IRF
    eva.M, eva.N = M, N
    eva.IRF_DEPTH = 8
    eva.DATADRIVEN = 1                                        # PC stalls until operands valid (golden instr_gt)
    PER_NODE = eva.IRF_DEPTH + 1 + 2                          # packets/node: 8 IRF + 1 weight(drf[0]) + 2 cfg regs
    if PROG_CYCLES is None:                                   # activations must land AFTER every START delivers
        PROG_CYCLES = M * PER_NODE + M + 4
    eva.PROG_CYCLES = PROG_CYCLES
    eva.NSTEP = PROG_CYCLES + KLEN + 2 * M + 2 * N + 16       # prefix + butterfly + stall/drain margin
    eva.LANELEN = eva.NSTEP
    L = eva.LANELEN

    prog = zi(M, N, eva.IRF_DEPTH)
    prog[0, 0, :8] = [I0, I1, I2, I3, I4_R, I5_R, I6, I7]     # row0 = real path
    prog[1, 0, :8] = [I0, I1, I2, I3, I4_I, I5_I, I6, I7]     # row1 = imag path
    win = zf(M, N); win[0, 0] = 1.0; win[1, 0] = 1.0         # r0 = w.r = 1.0 -> drf[0] for both cores

    rin_s = load_prog_packets(prog, M, N, data=[(0, win)],    # prog (mode=1) + weight->drf[0] (mode=0)
                              cfg=(KLEN, 1, 0))               # + config regs: iter_size=1 butterfly, no sync mask

    in_w, ivw = zf(M, L), zi(M, L)                            # 2 complex tokens per row (in0 at t, in1 at t+1)
    in_w[0, PROG_CYCLES + 0], in_w[0, PROG_CYCLES + 1] = np.float16(x[0].real), np.float16(x[1].real)
    in_w[1, PROG_CYCLES + 0], in_w[1, PROG_CYCLES + 1] = np.float16(x[0].imag), np.float16(x[1].imag)
    ivw[0, PROG_CYCLES:PROG_CYCLES + 2] = 1
    ivw[1, PROG_CYCLES:PROG_CYCLES + 2] = 1

    args = _bundle(zi(M, N, eva.IRF_DEPTH), zf(M, N), zf(M, N), in_w, ivw, M, N, L)  # ZERO prog/win
    args["rins"][3] = rin_s                                   # south router driver carries the program+weight
    out_e = args["outs"][1]

    def check():
        oe = out_e.astype(np.float32)                        # east collector compacts: [:,0]=out0, [:,1]=out1
        got = np.array([oe[0, 0] + 1j * oe[1, 0], oe[0, 1] + 1j * oe[1, 1]], np.complex64)
        gold = np.fft.fft(np.asarray(x[:2], np.complex64))   # 2-point DFT (W=1): [x0+x1, x0-x1]
        err = np.abs(got - gold)
        ok = bool(np.all(err < 0.02))
        lines = [f"=== ROUTER-LOADED 2-point FFT butterfly (PROG_CYCLES={PROG_CYCLES}) ===",
                 f"in0 = {complex(x[0]):+.5f}   in1 = {complex(x[1]):+.5f}"]
        for k in range(2):
            lines.append(f"  X[{k}]  got {got[k].real:+.5f}{got[k].imag:+.5f}j   "
                         f"gold {gold[k].real:+.5f}{gold[k].imag:+.5f}j   |err|={err[k]:.4f}")
        lines.append(f"FFT-2 {'PASS' if ok else 'FAIL'}")
        return ok, "\n".join(lines)

    return args, check


# ───────────────── FFT (8-point), program + twiddles LOADED OVER THE ROUTER ─────────────────
def load_fft8_router(x, PROG_CYCLES=None, MARGIN=140):
    """8-point radix-2 DIT FFT, program + twiddles loaded over the router (router-loaded, data-driven
    analog of fft8_eva.py). Grid M=8 rows x N=5 cols; rows paired into 4 complex lanes (2L=real,2L+1=imag).
    Pipeline L->R: col0 stage-1 butterfly (W=1) | col1 shuffle stride-2 | col2 stage-2 (W=1 / -j) |
    col3 shuffle stride-4 | col4 stage-3 (W8^L) -> outputs off the EAST edge (col4->sys_e[:,5], N=5 so no
    relay cols). Each PE runs a SHORT kernel (<=8 IRF): butterfly=8, shuffle=6. The shuffle's r7 landing
    slot is a SYNC register (dsmask bit 7): recv MOV rX,r7 STALLS until the partner's router packet
    delivers -- replaces fft8_eva.py's D-NOP pad, so the shuffle fits 8-slot IRF with no static schedule."""
    M, N = 8, 5
    eva.M, eva.N = M, N
    eva.IRF_DEPTH = 8
    eva.DATADRIVEN = 1
    PER_NODE = eva.IRF_DEPTH + 2 + 2                          # packets/node: 8 IRF + 2 twiddles + 2 cfg regs
    if PROG_CYCLES is None:
        PROG_CYCLES = M * PER_NODE + M + 4
    eva.PROG_CYCLES = PROG_CYCLES
    eva.NSTEP = PROG_CYCLES + MARGIN                          # prefix + 5-stage pipeline + drain margin
    eva.LANELEN = eva.NSTEP
    L = eva.LANELEN

    NOP = enc(MOV, 6, 6)                                      # touches only r6 (safe filler)
    def butterfly(imag):                                      # 8 instrs; out0/out1 = in0 +/- W*in1, W=(r0,r1)
        mid = [enc(MOV, SYSTOP, 5), enc(ADD, 6, 4, SYSTOP)] if imag else \
              [enc(MOV, SYSBTM, 5), enc(SUB, 6, 4, SYSBTM)]
        return [enc(MOV, 2, SYSLFT), enc(MOV, 3, SYSLFT), enc(MULT, 4, 0, 3), enc(MULT, 5, 1, 3)] + mid + \
               [enc(ADD, SYSRGT, 2, 6), enc(SUB, SYSRGT, 2, 6)]
    def shuffle(row, stride):                                 # 6 instrs; swap upper.r1 <-> lower.r0 via r7 sync
        upper = (row // stride) % 2 == 0
        send = enc(RTR + 1, 7, 1, row + stride) if upper else enc(RTR + 0, 7, 0, row - stride)  # DMOV / UMOV
        recv = enc(MOV, 1, 7) if upper else enc(MOV, 0, 7)   # MOV rX, r7 (stalls until partner delivers)
        return [enc(MOV, 0, SYSLFT), enc(MOV, 1, SYSLFT), send, recv,
                enc(MOV, SYSRGT, 0), enc(MOV, SYSRGT, 1)]

    w8 = [complex(np.cos(2 * np.pi * k / 8), -np.sin(2 * np.pi * k / 8)) for k in range(4)]
    def twid(col, lane):                                      # col0=1 ; col2=1|-j alt ; col4=W8^lane
        if col == 0: return 1 + 0j
        if col == 2: return (1 + 0j) if lane % 2 == 0 else -1j
        return w8[lane]

    BCOL = {0, 2, 4}
    prog = zi(M, N, eva.IRF_DEPTH); prog[:] = NOP
    win, win1 = zf(M, N), zf(M, N)
    KLEN, SMASK = zi(M, N), zi(M, N)
    for col in range(N):
        for row in range(M):
            lane, imag = row // 2, (row % 2 == 1)
            if col in BCOL:
                seq = butterfly(imag)
                W = twid(col, lane)
                win[row, col], win1[row, col] = np.float16(W.real), np.float16(W.imag)   # r0=W.re, r1=W.im
                KLEN[row, col], SMASK[row, col] = 8, 0
            else:                                             # shuffle columns 1 (stride 2) / 3 (stride 4)
                seq = shuffle(row, 2 if col == 1 else 4)
                KLEN[row, col], SMASK[row, col] = 6, 0x80     # r7 = sync register
            for k, ins in enumerate(seq):
                prog[row, col, k] = ins

    def cfg(i, j): return (int(KLEN[i, j]), 1, int(SMASK[i, j]))        # per-node kernel_len + sync mask
    rin_s = load_prog_packets(prog, M, N, data=[(0, win), (1, win1)], cfg=cfg)   # IRF + twiddles + cfg via router

    br = [0, 4, 2, 6, 1, 5, 3, 7]                             # bit-reversed input feed at the west edge
    in_w, ivw = zf(M, L), zi(M, L)
    for lane in range(4):
        p0, p1 = x[br[2 * lane]], x[br[2 * lane + 1]]
        in_w[2 * lane,     PROG_CYCLES + 0], in_w[2 * lane,     PROG_CYCLES + 1] = np.float16(p0.real), np.float16(p1.real)
        in_w[2 * lane + 1, PROG_CYCLES + 0], in_w[2 * lane + 1, PROG_CYCLES + 1] = np.float16(p0.imag), np.float16(p1.imag)
        ivw[2 * lane,     PROG_CYCLES:PROG_CYCLES + 2] = 1
        ivw[2 * lane + 1, PROG_CYCLES:PROG_CYCLES + 2] = 1

    args = _bundle(zi(M, N, eva.IRF_DEPTH), zf(M, N), zf(M, N), in_w, ivw, M, N, L)   # ZERO prog/win
    args["rins"][3] = rin_s                                   # south router driver carries program + twiddles
    out_e = args["outs"][1]

    def check():
        oe = out_e.astype(np.float32)                        # east collector compacts: [:,0]=X[L], [:,1]=X[L+4]
        X = np.zeros(8, np.complex64)
        for lane in range(4):
            X[lane]     = oe[2 * lane, 0] + 1j * oe[2 * lane + 1, 0]
            X[lane + 4] = oe[2 * lane, 1] + 1j * oe[2 * lane + 1, 1]
        gold = np.fft.fft(np.asarray(x, np.complex64))
        err = np.abs(X - gold)
        ok = bool(np.all(err < 0.05))
        lines = [f"=== ROUTER-LOADED 8-point FFT (M={M} N={N} PROG_CYCLES={PROG_CYCLES}) ==="]
        for k in range(8):
            lines.append(f"  X[{k}]  got {X[k].real:+.4f}{X[k].imag:+.4f}j   "
                         f"gold {gold[k].real:+.4f}{gold[k].imag:+.4f}j   |err|={err[k]:.4f}"
                         f"{'' if err[k] < 0.05 else '  <-- FAIL'}")
        lines.append(f"FFT-8 {'PASS' if ok else 'FAIL'}  (max |err| = {err.max():.4f})")
        return ok, "\n".join(lines)

    return args, check


# load_mmm (direct-arg program load) REMOVED 2026-07-06: it predated the
# router-load interface -- its prog/win arrays never reached the chip, so it
# FAILED in the simulator. Use load_mmm_router (mmmr) instead.


# ───────────────────── MMM, but program + weights LOADED OVER THE ROUTER (no direct args) ─────────────────────
def load_mmm_router(W, X, PROG_CYCLES=None):
    """MMM with program+weights delivered over the router AND a LOOPING program counter: the 4-instr
    kernel lives ONCE in irf[0..3] and loops B times (INSTR_SIZE=4, ITER_SIZE=B), so any batch count B
    fits the 8-slot IRF (no unrolling). The chip gets a ZERO prog/win -> end-to-end router load + looping.
    NOTE: multi-node arrays also need a per-node START offset (the i+3j skew) = next chip step; this path
    is exact for a single PE (1x1, off=0)."""
    I0, I1, I2, I3, NOP = 0xE13, 0x1022, 0x1F3, 0xC2D0, 0x773 # 4-instr MMM kernel
    W = np.asarray(W, np.float16); X = np.asarray(X, np.float16)
    B, M = X.shape; N = W.shape[1]
    KLEN = 4                                                  # MMM kernel length (I0..I3)
    eva.M, eva.N = M, N
    eva.IRF_DEPTH = 8
    eva.DATADRIVEN = 1                                        # PC stalls until operands valid (golden instr_gt; no skew)
    PER_NODE = eva.IRF_DEPTH + 1 + 2                          # packets/node: 8 IRF + 1 weight(drf) + 2 config regs
    if PROG_CYCLES is None:                                   # activations must land AFTER every START packet delivers
        PROG_CYCLES = M * PER_NODE + M + 4                    # M nodes/col streamed in turn + north transit + margin
    eva.PROG_CYCLES = PROG_CYCLES
    eva.NSTEP = PROG_CYCLES + KLEN * B + M + 3 * N + 2        # prefix + B kernel loops + drain margin
    eva.LANELEN = eva.NSTEP
    L = eva.LANELEN

    prog = zi(M, N, eva.IRF_DEPTH); prog[:] = NOP             # kernel ONCE in irf[0..3] (no unroll, no skew)
    win = zf(M, N) # preload this PE's stationary weight
    for i in range(M):
        for j in range(N):
            win[i, j] = W[i, j]
            for k, ins in enumerate((I0, I1, I2, I3)):
                prog[i, j, k] = ins

    rin_s = load_prog_packets(prog, M, N, data=[(0, win)],    # prog (mode=1) + weight->drf[0] (mode=0)
                              cfg=(KLEN, B, 0))               # + config_reg_1 (iter=B) + config_reg_0 (START), no sync

    in_w, ivw = zf(M, L), zi(M, L)                            # batch b's kernel iter starts at ec=4b -> cycle PC+4b
    for b in range(B):
        for i in range(M):
            in_w[i, PROG_CYCLES + KLEN * b] = X[b, i]
            ivw[i, PROG_CYCLES + KLEN * b] = 1
    ivn = np.ones((N, L), np.int32)                           # north accumulator seed = valid 0

    args = _bundle(zi(M, N, eva.IRF_DEPTH), zf(M, N), zf(M, N), in_w, ivw, M, N, L)  # ZERO prog/win
    args["ivs"][2] = ivn
    args["rins"][3] = rin_s                                   # south router driver carries the program+data
    out_s = args["outs"][3]

    def check():
        got = np.array([[float(out_s[j, b]) for j in range(N)] for b in range(B)], np.float32)
        gold = (X.astype(np.float32) @ W.astype(np.float32))
        ok = bool(np.allclose(got, gold, atol=1e-2))
        return ok, (f"=== ROUTER-LOADED + LOOPING MMM {M}x{N} B={B} (INSTR_SIZE={KLEN} ITER_SIZE={B} "
                    f"PROG_CYCLES={PROG_CYCLES}) ===\nexpected Y = X@W =\n{gold}\ngot out_s =\n{got}\n"
                    f"MMM {'PASS' if ok else 'FAIL'}")
    return args, check


# ─────────────────────────────────────────── runner ───────────────────────────────────────────
def run(loader_result):
# takes a loader's tuple and runs that program on the chip
    args, check = loader_result # unpack program bundle (args) and verifier closure (check)
    mod = df.build(eva.get_eva_top(float16), target='simulator') # build the fp16 EVA chip
    eva.run_eva(mod, args["ins"], args["ivs"], args["outs"], args["rins"], args["routs"])
    # invoke the chip with the W/E/N/S I/O lists, mutating the output buffers in place
    return check() # run verifier against now-populated outputs and return its (ok, report)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "fft" # pick 
    # OUTDATED
    if which == "fft":
        rng = np.random.default_rng(8)
        x = (rng.uniform(-1, 1, 8) + 1j * rng.uniform(-1, 1, 8)).astype(np.complex64)
        ok, report = run(load_fft(x))
    elif which == "fft2r":                                      # router-loaded 2-point butterfly (golden bit patterns)
        b16 = lambda h: np.uint16(h).view(np.float16) # helper reinterpreting a raw 16-bit hex value as fp16
        x = np.array([b16(0x2617) + 1j * b16(0x2AF7),           # golden EVA inputs (tb_pe_group_fft.cpp)
                      b16(0x26A3) + 1j * b16(0x25FF)], np.complex64)
        ok, report = run(load_fft2_router(x)) # run router-loaded 2-point FFT and capture result
    elif which == "fft8r":                                      # router-loaded 8-point FFT
        rng = np.random.default_rng(8) # same seeded RNG for reproducibility
        x = (rng.uniform(-1, 1, 8) + 1j * rng.uniform(-1, 1, 8)).astype(np.complex64)
        # same 8 random complex inputs
        ok, report = run(load_fft8_router(x)) # run router-loaded 8-point fFT and capture result
    elif which == "mmmr":
        W = np.array([[1, 3], [5, 7]], np.float16)             # golden EVA: y=[22,34]/[14,26]
        # 2x2 data-driven: skew emerges from stalls (B=1)
        ok, report = run(load_mmm_router(W, np.array([[2, 4]], np.float16)))      # golden -> [22,34]
    else:
        print(f"unknown workload '{which}' (use: fft | fft2r | fft8r | mmmr)"); sys.exit(2)
    print(report)
    sys.exit(0 if ok else 1)
