# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the tree's two output latencies from its RTL, from both sides.

A unit with two output depths has two numbers in its interface, and MiniTPU's
sharpest lesson is what happens when one of them is written down but never
measured: their mid-tree tap was declared at 11 against a root at 15, and for
a whole commit **nothing used the tap's number** -- wiring it up afterwards
took a span-64 softmax from 150 bundles to 142. A declared latency that
nothing measures is not an interface, it is a comment.

So `red_full` and `red_group` are declared in `ReduceParams` (derived from the
geometry, never chosen beside it) and this measures both against the emitted
Verilog.

**Two-sided, which is the part that is easy to get wrong.** The comparison is
for EQUALITY in both directions, not `>=`, and the scan runs DOWNWARD:

    the smallest offset k from which every later offset also agrees

A first-success scan from below, or a `>=` comparison, accepts a declared
latency that is merely a safe lower bound -- which is exactly how a number
nothing uses survives unnoticed. Here that means: if the RTL is SLOWER than
declared the write lands later than the declared offset and the check fails;
if the RTL is FASTER -- the honest failure mode for a conservative
declaration -- the write has already landed at `declared - 1` and the check
fails too. Both directions fault.

**Coincidence is rejected by value, not only by cycle.** Every word in the
stream carries different operands, and the write at the measured offset must
carry that word's own reduction. A cycle that matches while the bus holds a
previous word's value is not a match.

**Mixed traffic, back to back.** MiniTPU's op tag was one pipeline stage
early, and *a single-op stream is immune to that bug*; a uniform test proves
nothing. This unit has one operation but TWO outputs at two depths, so the
analogue is the same shape: a stream of identical words cannot distinguish the
tap from the root or a tap taken one stage early, and a single word cannot
distinguish a pipelined unit from a sequential one at all. The probe therefore
runs a back-to-back stream of all-different words AND a one-word run, and
requires the same latencies from both.

    python reduce_latency_probe.py            # the default 8:2 export
    python reduce_latency_probe.py 16:4
"""

import os
import random
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

ASIC = os.path.join(HERE, "asic_reduce")
WORK = os.path.join(HERE, ".probe_work")

TB = r"""
`timescale 1ns/1ps
module tb;
  reg ap_clk = 0, ap_rst = 1, ap_start = 0;
  wire ap_done, ap_idle, ap_ready;
  reg  [%(IN_W)d-1:0] words [0:%(NPAD)d-1];
  integer expect_full [0:%(NW)d-1];
  reg  [%(GRP_W)d-1:0] expect_grp [0:%(NW)d-1];
  integer cycle = 0, fed = 0, got_full = 0, got_grp = 0;

  wire [%(IN_W)d-1:0] v282_dout = words[fed];
  // NEVER runs dry. `ap_block_state2_pp0_stage0_iter1 = (v282_empty_n ==
  // 1'b0)` in the emitted RTL: iter1 blocks whenever the input reads empty,
  // whether or not it still needs a word, so a source that stops at the last
  // word deadlocks the DRAIN and measures nothing. The loop is bounded by the
  // trip count on `empty` instead, which is also the steady-state case worth
  // measuring: back to back at II=1.
  wire v282_empty_n = 1'b1;
  wire v282_read;
  wire [%(GRP_W)d-1:0] v284_din;  wire v284_write;
  wire [%(ACC_W)d-1:0] v283_din;  wire v283_write;

  %(TOP)s dut (
    .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start),
    .ap_done(ap_done), .ap_idle(ap_idle), .ap_ready(ap_ready),
    .v282_dout(v282_dout), .v282_num_data_valid(4'd0), .v282_fifo_cap(4'd0),
    .v282_empty_n(v282_empty_n), .v282_read(v282_read),
    .v284_din(v284_din), .v284_num_data_valid(4'd0), .v284_fifo_cap(4'd0),
    .v284_full_n(1'b1), .v284_write(v284_write),
    .v283_din(v283_din), .v283_num_data_valid(4'd0), .v283_fifo_cap(4'd0),
    .v283_full_n(1'b1), .v283_write(v283_write),
    .empty(16'd%(NW)d));

  always #1 ap_clk = ~ap_clk;

  // Cycle-stamped events. The INPUT stamp is the handshake cycle, which is
  // the only defensible zero: `ap_start` is a region-level fact and would
  // fold the feeder's own latency into the unit's.
  always @(posedge ap_clk) begin
    if (!ap_rst) begin
      cycle <= cycle + 1;
      if (v282_read && v282_empty_n) begin
        $display("EV %%0d IN %%0d", cycle, fed);
        fed <= fed + 1;
      end
      if (v283_write) begin
        $display("EV %%0d FULL %%0d %%0d", cycle, got_full, $signed(v283_din));
        got_full <= got_full + 1;
      end
      if (v284_write) begin
        $display("EV %%0d GRP %%0d %%h", cycle, got_grp, v284_din);
        got_grp <= got_grp + 1;
      end
    end
  end

  initial begin
    $readmemh("%(WORDS)s", words);
    repeat (4) @(posedge ap_clk);
    ap_rst = 0;
    @(posedge ap_clk);
    ap_start = 1;
    wait (ap_done);
    @(posedge ap_clk);
    $display("DONE %%0d in=%%0d full=%%0d grp=%%0d", cycle, fed, got_full, got_grp);
    $finish;
  end

  initial begin
    #200000;
    $display("TIMEOUT");
    $finish;
  end
endmodule
"""


def expected(word, lanes, groups, in_bits):
    """The tree's two answers for one packed word, computed the way the unit
    declares them: lane `g + s*RED_GROUPS` lands at leaf `g*GROUP_SIZE + s`,
    so group `g` is the sum of the lanes congruent to `g` mod RED_GROUPS."""
    mask = (1 << in_bits) - 1
    sign = 1 << (in_bits - 1)
    lane = []
    for k in range(lanes):
        v = (word >> (in_bits * k)) & mask
        lane.append(v - (1 << in_bits) if v & sign else v)
    per_group = [sum(lane[g::groups]) for g in range(groups)]
    return sum(lane), per_group


def run_sim(top, rtl_files, words, lanes, groups, in_bits, acc_bits, tag):
    work = os.path.join(WORK, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    hexpath = os.path.join(work, "words.hex")
    with open(hexpath, "w") as fh:
        for w in list(words) + [0] * 8:
            fh.write("%x\n" % w)
    src = TB % {"IN_W": lanes * in_bits, "GRP_W": groups * acc_bits,
                "ACC_W": acc_bits, "NW": len(words),
                "NPAD": len(words) + 8, "TOP": top, "WORDS": hexpath}
    tbpath = os.path.join(work, "tb.v")
    open(tbpath, "w").write(src)
    exe = os.path.join(work, "sim")
    comp = subprocess.run(["iverilog", "-g2005", "-o", exe, tbpath] + rtl_files,
                          capture_output=True, text=True)
    if comp.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(f"iverilog failed:\n{comp.stdout}\n{comp.stderr}")
    sim = subprocess.run([exe], capture_output=True, text=True, timeout=300)
    return sim.stdout


def parse(out):
    ev = {"IN": {}, "FULL": {}, "GRP": {}}
    val = {"FULL": {}, "GRP": {}}
    for line in out.splitlines():
        m = re.match(r"EV (\d+) (IN|FULL|GRP) (\d+)(?: (-?[0-9a-fA-F]+))?", line)
        if not m:
            continue
        cyc, kind, idx, v = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        ev[kind][idx] = cyc
        if v is not None and kind != "IN":
            val[kind][idx] = v
    return ev, val, ("TIMEOUT" in out), re.search(r"DONE .*", out)


def measure(ev, val, words, lanes, groups, in_bits, acc_bits):
    """Per-word offsets, and the value check that rejects a coincidence."""
    mask_acc = (1 << acc_bits) - 1
    offs = {"FULL": [], "GRP": []}
    bad = []
    for w in range(len(words)):
        full, per_group = expected(words[w], lanes, groups, in_bits)
        if w not in ev["IN"]:
            bad.append(f"word {w} was never consumed")
            continue
        for kind in ("FULL", "GRP"):
            if w not in ev[kind]:
                bad.append(f"word {w}: no {kind} write")
                continue
            offs[kind].append(ev[kind][w] - ev["IN"][w])
        if w in val["FULL"] and int(val["FULL"][w]) != full:
            bad.append(f"word {w}: red_full {val['FULL'][w]} != {full}")
        if w in val["GRP"]:
            raw = int(val["GRP"][w], 16)
            for g in range(groups):
                got = (raw >> (acc_bits * g)) & mask_acc
                want = per_group[g] & mask_acc
                if got != want:
                    bad.append(f"word {w} group {g}: {got} != {want}")
    return offs, bad


def two_sided(offs, declared, name):
    """The downward, monotone-suffix rule, then EQUALITY against `declared`.

    `offs[w]` is word `w`'s measured offset. The measured latency is the
    smallest offset from which every later word also agrees -- so one early
    word that happens to line up cannot set the number. Then the comparison is
    `==`: a declared value the RTL beats is as much a failure as one it
    misses, because a scheduler booking the declared value would leave the
    unit's real slack unused and a consumer written against it would be wrong
    in the other direction the day the geometry changes."""
    if not offs:
        return None, [f"{name}: nothing measured"]
    measured = None
    for k in range(len(offs) - 1, -1, -1):
        if len(set(offs[k:])) != 1:
            break
        measured = offs[k]
    if measured is None:
        return None, [f"{name}: offsets never settle: {offs}"]
    problems = []
    if declared is None:
        problems.append(
            f"{name}: measured {measured} but the unit declares NOTHING. An "
            f"output whose latency is not in the interface is the MiniTPU "
            f"failure exactly -- a number nothing books.")
    if len(set(offs)) != 1:
        problems.append(f"{name}: offsets are not uniform across the stream: "
                        f"{offs} -- a unit whose latency depends on position "
                        f"in the stream has no single declared latency")
    if declared is not None and measured != declared:
        direction = "FASTER" if measured < declared else "slower"
        problems.append(
            f"{name}: RTL is {direction} than declared -- measured "
            f"{measured}, declared {declared}. Equality, not a bound: a "
            f"declaration the RTL beats is the failure mode nothing catches.")
    return measured, problems


def main(argv):
    from examples.tinytpu.ip.reduce import ReduceParams
    import json

    lanes, groups = 8, 2
    for a in argv:
        if ":" in a:
            lanes, groups = (int(x) for x in a.split(":"))
    dest = os.path.join(ASIC, f"reduce_{lanes}_{groups}")
    meta = json.load(open(os.path.join(dest, "MANIFEST.json")))
    top = meta["tops"]["tree"]
    manifest = os.path.join(dest, "sv2v_manifest_tree.f")
    rtl_files = [os.path.join(dest, "rtl", ln.strip())
                 for ln in open(manifest) if ln.strip()
                 and not ln.startswith("#")]

    p = ReduceParams(RED_LANES=lanes, RED_GROUPS=groups,
                     DOT_MAX=max(8, lanes))
    in_bits, acc_bits = p.RED_IN_BITS, p.RED_ACC_BITS
    rng = random.Random(20260922)
    lim = 1 << (in_bits - 1)

    def word():
        v = 0
        for k in range(lanes):
            x = rng.randrange(-lim, lim) & ((1 << in_bits) - 1)
            v |= x << (in_bits * k)
        return v

    # Every word different: a uniform stream cannot tell the tap from the
    # root, nor a tap taken one pipeline stage early.
    stream = [word() for _ in range(12)]
    while len(set(stream)) != len(stream):
        stream = [word() for _ in range(12)]
    cases = [("stream", stream), ("single", [stream[0]])]

    dec_full = getattr(p, "RED_FULL_LAT", None)
    dec_grp = getattr(p, "RED_GROUP_LAT", None)
    dec_slack = getattr(p, "RED_TAP_SLACK", None)
    print(f"top {top}")
    print(f"declared: red_full {dec_full}, red_group {dec_grp}, "
          f"tap slack {dec_slack}")
    ok = True
    results = {}
    for tag, words in cases:
        out = run_sim(top, rtl_files, words, lanes, groups, in_bits,
                      acc_bits, tag)
        ev, val, timeout, done = parse(out)
        if timeout:
            print(f"  {tag}: SIMULATION TIMED OUT")
            ok = False
            continue
        offs, bad = measure(ev, val, words, lanes, groups, in_bits, acc_bits)
        m_full, p1 = two_sided(offs["FULL"], dec_full, "red_full")
        m_grp, p2 = two_sided(offs["GRP"], dec_grp, "red_group")
        problems = bad + p1 + p2
        slack = (m_full - m_grp) if (m_full is not None
                                     and m_grp is not None) else None
        if (slack is not None and dec_slack is not None
                and slack != dec_slack):
            problems.append(
                f"tap slack: measured {slack}, declared {dec_slack}")
        results[tag] = {"red_full": m_full, "red_group": m_grp,
                        "tap_slack": slack, "words": len(words)}
        print(f"  {tag} ({len(words)} words): red_full {m_full}, red_group "
              f"{m_grp}, tap slack {slack}"
              + ("" if not problems else "  <-- PROBLEMS"))
        for msg in problems:
            print(f"      {msg}")
        ok &= not problems
    if (results.get("stream", {}).get("red_full")
            != results.get("single", {}).get("red_full")):
        print("      a one-word run and a back-to-back stream disagree on "
              "red_full -- the pipelined and drained latencies are different "
              "numbers and only one of them is declared")
        ok = False
    print("\nLATENCY PROBE " + ("OK" if ok else "FAILED"))
    shutil.rmtree(WORK, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
