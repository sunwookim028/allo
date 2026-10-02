import sys, os, time, signal
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "dev/records/minitpu/u3_fifo_composed_2026-10-02/scripts")
import allo.dataflow as df
import cmp_composed
from examples.minitpu.units import vpu_fifo as u
def handler(signum, frame): raise TimeoutError()
signal.signal(signal.SIGALRM, handler)
make, run = u.VARIANTS["composed"]
cmd_all, spans = cmp_composed.trace_composed_all("w32d4", 200)
def hangs(n):
    cmd = {p: v[:n] for p, v in cmd_all.items()}
    mod = df.build(make(n, 32), target="simulator")
    signal.alarm(20)
    try:
        run(mod, cmd, n, 32); signal.alarm(0); return False
    except TimeoutError:
        return True
lo, hi = 40, 200  # 40 ok (fill+wrap), 200 hangs
assert not hangs(lo) and hangs(hi)
while hi - lo > 1:
    mid = (lo + hi) // 2
    if hangs(mid): hi = mid
    else: lo = mid
    print("bisect", lo, hi, flush=True)
n = hi
print(f"minimal hanging prefix n={n}; last ok n={lo}")
c = {p: v[:n] for p, v in cmd_all.items()}
print("rst :", "".join(map(str, c["rst_ni"][36:n])))
print("push:", "".join(map(str, c["push_i"][36:n])))
print("pop :", "".join(map(str, c["pop_i"][36:n])))
cnt = 0; cnt_s = 0
for t in range(n):
    if c["rst_ni"][t] == 0: cnt = 0; continue
    cnt += c["push_i"][t] - c["pop_i"][t]
    cnt_s += c["push_i"][t] - c["pop_i"][t]
print(f"RTL count at end {cnt}; stream count (no reset) at end {cnt_s}")
