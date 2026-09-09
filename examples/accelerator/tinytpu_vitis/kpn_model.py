"""A KPN model of tinytpu_isa's channel structure, with bounded FIFOs and
DEADLOCK REPORTING -- which the Allo simulator does not provide.

Each unit is a generator yielding ('get', ch) or ('put', ch, n). A cooperative
scheduler runs them; when every live process is blocked it prints each one's
blocked operation and the occupancy of the channel it is waiting on. That is
the report needed to say what the FIFO problem *is* rather than guess.
"""
import sys, collections
sys.path.insert(0, "/home/sk3463/allo")
from examples.accelerator.tinytpu_vitis import microarch_isa as U

T, NPROG = U.T, U.NPROG

def fields(w):
    g = lambda lo, hi: (int(w) >> lo) & ((1 << (hi - lo)) - 1)
    return dict(op=g(0,6), f0=g(6,18), f1=g(18,30), f2=g(30,42), f3=g(42,54), nr=g(54,61))

def build(prog, QD):
    P = [fields(w) for w in prog] + [dict(op=0,f0=0,f1=0,f2=0,f3=0,nr=0)] * (NPROG - len(prog))
    cap = collections.defaultdict(lambda: QD)
    q = collections.defaultdict(collections.deque)

    def seq():
        for c in range(NPROG):
            yield ('put', 'c_dld', 1)
    def dma_ld():
        for c in range(NPROG):
            yield ('get', 'c_dld'); yield ('put', 'c_spm', 1)
            if P[c]['op'] == U.OP_DMA_LD:
                for _ in range(P[c]['nr']): yield ('put', 'dma2sp', 1)
    def spm():
        for c in range(NPROG):
            yield ('get', 'c_spm'); yield ('put', 'c_vru', 1)
            if P[c]['op'] == U.OP_DMA_LD:
                for _ in range(P[c]['nr']): yield ('get', 'dma2sp')
            if P[c]['op'] == U.OP_VLD:
                for _ in range(P[c]['nr']): yield ('put', 'sp2vr', 1)
    def vru():
        for c in range(NPROG):
            yield ('get', 'c_vru'); yield ('put', 'c_acc', 1)
            yield ('put', 'wcol0', 1)                      # header, EVERY instr
            if P[c]['op'] == U.OP_VLD:
                for _ in range(P[c]['nr']): yield ('get', 'sp2vr')
            if P[c]['op'] == U.OP_MM:
                for _ in range(T): yield ('put', 'wcol0', 1)
                for _ in range(P[c]['nr']): yield ('put', 'acol0', 1)
    def pe(i, j):
        def g():
            for c in range(NPROG):
                if j == 0:
                    yield ('get', f'wcol{i}')
                    if i != T-1: yield ('put', f'wcol{i+1}', 1)
                else:
                    yield ('get', f'wrow{i}_{j-1}')
                if j != T-1: yield ('put', f'wrow{i}_{j}', 1)
                if P[c]['op'] == U.OP_MM:
                    if j == 0:
                        yield ('get', f'wcol{i}')
                        for _ in range(T-1-i):
                            yield ('get', f'wcol{i}'); yield ('put', f'wcol{i+1}', 1)
                    else:
                        yield ('get', f'wrow{i}_{j-1}')
                    if j != T-1: yield ('put', f'wrow{i}_{j}', 1)
                    for m in range(P[c]['nr']):
                        if j == 0:
                            yield ('get', f'acol{i}')
                            if i != T-1: yield ('put', f'acol{i+1}', 1)
                        else:
                            yield ('get', f'a_fwd{i}_{j-1}')
                        if i > 0: yield ('get', f'p_fwd{i-1}_{j}')
                        if i != T-1: yield ('put', f'p_fwd{i}_{j}', 1)
                        else:
                            if j > 0: yield ('get', f'cw{j-1}')
                            yield ('put', f'cw{j}', 1)
                        if j != T-1: yield ('put', f'a_fwd{i}_{j}', 1)
        return g()
    def accu():
        for c in range(NPROG):
            yield ('get', 'c_acc'); yield ('put', 'c_dst', 1)
            if P[c]['op'] == U.OP_MM:
                for _ in range(P[c]['nr']): yield ('get', f'cw{T-1}')
            if P[c]['op'] == U.OP_MVOUT:
                for _ in range(P[c]['nr']): yield ('put', 'ac2sp', 1)
    def dma_st():
        for c in range(NPROG):
            yield ('get', 'c_dst')
            if P[c]['op'] == U.OP_MVOUT:
                for _ in range(P[c]['nr']): yield ('get', 'ac2sp')

    procs = {'sequencer': seq(), 'dma_ld': dma_ld(), 'spm': spm(), 'vru': vru(),
             'accu': accu(), 'dma_st': dma_st()}
    for i in range(T):
        for j in range(T):
            procs[f'pe{i}_{j}'] = pe(i, j)
    return procs, q, cap

def run(prog, QD, verbose=True):
    procs, q, cap = build(prog, QD)
    pending, done = {}, set()
    for n in procs:
        pending[n] = None
    progress = True
    while progress:
        progress = False
        for n, gen in procs.items():
            if n in done: continue
            while True:
                if pending[n] is None:
                    try: pending[n] = next(gen)
                    except StopIteration:
                        done.add(n); progress = True; break
                act = pending[n]
                if act[0] == 'get':
                    if q[act[1]]:
                        q[act[1]].popleft(); pending[n] = None; progress = True
                    else: break
                else:
                    if len(q[act[1]]) < cap[act[1]]:
                        q[act[1]].append(1); pending[n] = None; progress = True
                    else: break
    if len(done) == len(procs):
        return True, None
    if verbose:
        print(f"    DEADLOCK at QD={QD}: {len(procs)-len(done)} processes blocked")
        for n in procs:
            if n in done: continue
            a = pending[n]
            occ = len(q[a[1]])
            print(f"      {n:10s} blocked on {a[0]:3s} {a[1]:12s} (occupancy {occ}/{cap[a[1]]})")
    return False, pending

if __name__ == "__main__":
    prog = U.gemm_program(False)
    print(f"{U.M}x{U.K}x{U.N}: {len(prog)} instrs, NPROG={NPROG}")
    lo = None
    for QD in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        ok, _ = run(prog, QD, verbose=False)
        print(f"  QD={QD:4d} : {'ok' if ok else 'DEADLOCK'}")
        if ok and lo is None: lo = QD
    print(f"\n  minimum working depth in the model: {lo}")
    if lo:
        print(f"\n  --- report at QD={lo//2 if lo>4 else 2} (the largest failing depth) ---")
        run(prog, lo // 2 if lo > 4 else 2)
