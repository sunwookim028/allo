"""Part A: does any consumer of a ``vpu_fifo`` hold the head word before popping?

    python peek_check.py [--scratch DIR]

From the worktree root after ``source examples/minitpu/harness/env-zhang21.sh``.
Two of MiniTPU's own testbenches, unchanged, under Verilator ``--trace``:

* ``tb_mxu_single_port`` (``mxu`` alone, DIM = 2): the input FIFO and both
  lanes' output FIFOs, through ``harness/vcd_seed.extract`` (the U2 seed path).
* ``tb_matrix_command_ii`` (the whole ``vpu``: stream engine, pop engine,
  MXU): the input FIFO, lane 0's output FIFO and ``mxu_pop_engine``'s ports,
  dumped by scope so the VCD stays small.

For every pop of each FIFO instance: how many cycles the head word was
visible (``empty_o == 0``) before the cycle of the pop. 0 = popped in the
first cycle it was visible. For the output FIFO the gap is the program's
(a vmatpop issued later), so the question there is answered by the pop
engine's own ports: ``mxu_output_pop_o`` against ``mxu_output_valid_i`` and
``popping_q``, which the script tabulates too.
"""
import argparse, collections, os, subprocess, sys

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import rtl, vcd_seed  # noqa: E402
from examples.minitpu.units import vpu_fifo as u  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scratch", default="/work/shared/users/phd/sk3463/scratch/u3fc/peek")
a = ap.parse_args()
home = rtl.minitpu_home()


def held_before_pop(cmd, seen, label):
    """Per pop: cycles since the head became visible; plus push->pop distance."""
    push, pop, empty = cmd["push_i"], cmd["pop_i"], seen["empty_o"]
    n = len(pop)
    hist = collections.Counter()
    first_visible = None
    pushes = []  # cycles of pushes, FIFO order
    pairs = []
    for t in range(n):
        if empty[t] == 0 and first_visible is None:
            first_visible = t
        if pop[t] == 1 and empty[t] == 0:
            held = t - first_visible
            hist[held] += 1
            pt = pushes.pop(0) if pushes else None
            pairs.append((t, pt, held))
            # after this pop the next head (if any) is visible from t + 1
            first_visible = t + 1 if (len(pushes) > 0 or push[t] == 1) else None
        if push[t] == 1:
            pushes.append(t)
        if empty[t] == 1:
            first_visible = None
    tot = sum(hist.values())
    print(f"PEEK {label}: {tot} pops; head visible before the pop (cycles): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
    for t, pt, held in pairs[:6]:
        print(f"    pop at cycle {t}: pushed at {pt}, push->pop {None if pt is None else t - pt} cycles, head visible {held} cycle(s) before the pop")
    return hist


# ---- tb_mxu_single_port: the harness's seed path (cached VCD) ----
for lab, inst, cmd, seen, _legal in u.seeds():
    held_before_pop(cmd, seen, f"tb_mxu_single_port {lab} [{inst}]")

# ---- tb_matrix_command_ii: the whole VPU, dumped by scope ----
tb = "tb_matrix_command_ii"
d = os.path.join(a.scratch, tb)
vcd = os.path.join(d, "dump.vcd")
if not os.path.exists(vcd):
    os.makedirs(d, exist_ok=True)
    top = os.path.join(d, "u3_peek_top.sv")
    scopes = [f"{tb}.dut.i_mxu.i_input_fifo",
              f"{tb}.dut.i_mxu.gen_lane_output[0].i_output_fifo",
              f"{tb}.dut.i_matrix_ctrl.i_pop_engine"]
    with open(top, "w", encoding="utf-8") as f:
        f.write("`timescale 1ns/1ps\nmodule u3_peek_top;\n  %s tb();\n" % tb
                + '  initial begin $dumpfile("%s"); %s end\nendmodule\n'
                % (vcd, " ".join(f"$dumpvars(0, tb.{s[len(tb) + 1:]});" for s in scopes)))
    cmd = ["verilator", "--build-jobs", "8", "--binary", "--timing", "--trace", "-Wno-fatal",
           "--top-module", "u3_peek_top", "-f", "src/core/sequencer/sequencer.f", "-f", "src/core/vpu.f",
           "tb/isa_latency_pkg.sv", f"tb/{tb}.sv", top, "--Mdir", d]
    env = dict(os.environ, CXX=rtl._cxx())
    r = subprocess.run(cmd, cwd=home, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)
    r = subprocess.run([os.path.join(d, "Vu3_peek_top")], cwd=d, capture_output=True, text=True, timeout=900)
    print(r.stdout[-600:])
    if "PASS" not in r.stdout:
        print("tb did not pass", r.stderr[-1000:])
        sys.exit(1)

def rows(scope, names, clk="clk_i"):
    ch = vcd_seed.read_vcd(vcd, scope, [clk] + names)
    edges = [t for (t, v), (_, prev) in zip(ch[clk][1:], ch[clk][:-1]) if v == 1 and prev == 0]
    out = {}
    for nme in names:
        cur, i, c, col = 0, 0, ch[nme], []
        for t in edges:
            while i < len(c) and c[i][0] < t:
                cur = c[i][1]
                i += 1
            col.append(cur)
        out[nme] = col
    return out

fi = rows("dut.i_mxu.i_input_fifo", ["push_i", "pop_i", "empty_o", "full_o"])
held_before_pop({"push_i": fi["push_i"], "pop_i": fi["pop_i"]}, {"empty_o": fi["empty_o"]},
                f"{tb} input FIFO (257 b, full VPU)")
print(f"    input FIFO full_o ever 1: {sum(fi['full_o'])} cycles; cycles non-empty: {sum(1 - x for x in fi['empty_o'])}; pops: {sum(fi['pop_i'])}; pushes: {sum(fi['push_i'])}")
fo = rows("dut.i_mxu.gen_lane_output[0].i_output_fifo", ["push_i", "pop_i", "empty_o", "full_o"])
held_before_pop({"push_i": fo["push_i"], "pop_i": fo["pop_i"]}, {"empty_o": fo["empty_o"]},
                f"{tb} output FIFO lane 0 (64 b x 16, full VPU)")
pe = rows("dut.i_matrix_ctrl.i_pop_engine", ["vmatpop_valid_i", "popping_q", "mxu_output_valid_i", "mxu_output_pop_o"])
n = len(pe["popping_q"])
same = sum(1 for t in range(n) if pe["mxu_output_pop_o"][t] == 1)
lag = collections.Counter()
t0 = None
for t in range(n):
    if pe["vmatpop_valid_i"][t] == 1:
        t0 = t
    if pe["mxu_output_pop_o"][t] == 1:
        # cycles from the vmatpop issue to the pop, and whether valid was already up at issue
        lag[(t - t0) if t0 is not None else None] += 1
        t0 = None
waits = collections.Counter()
run = 0
for t in range(n):
    if pe["popping_q"][t] == 1 and pe["mxu_output_valid_i"][t] == 0:
        run += 1
    elif pe["mxu_output_pop_o"][t] == 1:
        waits[run] += 1
        run = 0
valid_no_pop = sum(1 for t in range(n) if pe["mxu_output_valid_i"][t] == 1 and pe["popping_q"][t] == 0)
print(f"POP-ENGINE {tb}: {same} pops; pop_o == popping_q && valid_i on every one: "
      f"{all(pe['mxu_output_pop_o'][t] == (pe['popping_q'][t] & pe['mxu_output_valid_i'][t]) for t in range(n))}; "
      f"issue->pop cycles {dict(sorted(lag.items(), key=lambda kv: (kv[0] is None, kv[0])))}; "
      f"cycles popping_q waited on valid before each pop {dict(sorted(waits.items()))}; "
      f"cycles with valid_i=1 and no vmatpop pending (the head sits in the FIFO, unread): {valid_no_pop}")
