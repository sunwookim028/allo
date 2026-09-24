# csim‑vs‑cosim regression harness — design sketch

**Problem it solves.** The SystemC emitter compiles *different code* for csim vs synthesis
(the `#ifdef __SYNTHESIS__` splits: loop shape `while(1)` vs finite `for`, `ap_int` shim vs
plain `ac_int` alias, `wait()` placement, memcpy guards). So **csim passing does not prove the
RTL is correct**, and any emitter change (e.g. the loop‑shape transform) can make csim pass
while cosim silently diverges. Today the only defense is a slow manual csim/csyn/cosim sweep.

**Key leverage.** `df.build(..., mode="cosim")` **already** builds a csim "software golden" and
diffs the Catapult RTL against it (SCVerify + Xcelium), printing `cosim MATCH` / mismatch. So we
do not need to re‑implement the comparison — we need to (a) run it across all designs and record
the verdict against an expected baseline, and (b) add a *fast* tier so most regressions are
caught in seconds, not minutes.

---

## Two tiers (because cosim is minutes/design, snapshots are milliseconds)

| tier | what | speed | catches | runs |
|------|------|-------|---------|------|
| **A. Emission snapshot** | emit `kernel.cpp`, normalize, diff vs golden | ~1 s/design | *any* change to emitted C++ (incl. every `#ifdef` split) | every commit / pre‑push |
| **B. Cosim equivalence** | `mode="cosim"`, record MATCH vs baseline | 2–8 min/design | the RTL actually diverging from csim | nightly / pre‑merge |

Tier A is the workhorse: it fails loudly the instant an edit changes *what is emitted* for any
construct — including the synthesis‑only branches a human never reads. Tier B confirms the
emitted RTL is still *functionally* equal to csim on real stimulus. A green A + green B is the
"safe to land an emitter change" gate.

---

## Tier A — emission snapshot tests (fast)

**Idea.** For a set of tiny canonical designs (one per emitter feature), emit the SystemC
**without synthesizing** and diff against a checked‑in golden `.cpp`. Emitting is cheap:
`csyn_subdir.py` already shows `df.build(region, target="systemc", mode="csyn", project=prj)`
**writes `kernel.cpp` without launching Catapult** (you just don't call the returned module).

**Canonical corpus** — one minimal kernel per construct, so a diff pinpoints the feature:

```
snapshots/
  wire.py              # Wire link            -> sc_signal read/write
  channel_vr.py        # Channel valid_ready  -> Combinational Pop/Push
  channel_vo.py        # Channel valid_only   -> _dat/_vld + modulario accessors
  stream.py            # Stream depth>=1      -> AlloFifoC, empty()/full() sideband
  stream_nb.py         # try_get/try_put      -> PopNB/PushNB + _nb temp
  memport_ro.py        # random-access input  -> AlloMem req/rsp
  memport_rmw.py       # C[i]+=... (both)     -> AlloMemW + single-shot (NOT free-run)
  loopshape.py         # dead-IV for t        -> while(1) under __SYNTHESIS__
  bitslice.py          # x(hi,lo) packed      -> ap_sel vs ac_int alias split
  wideword.py          # >64-bit link         -> ap_sel<W,true> ctor
  goldens/<name>.cpp   # checked-in expected emission
```

**IMPORTANT — snapshot BOTH branches.** Emit each design once as csim and once as synthesis so
the golden captures the `#ifdef __SYNTHESIS__` divergence itself. Cheapest way: emit the single
`kernel.cpp` (it contains both `#ifdef`/`#else` arms as text) and snapshot that whole file — the
divergence is visible in the golden without compiling either arm.

**Normalization** (before diff — these vary run‑to‑run and must be stripped):
- the `// Generated date: ...` line
- absolute paths / `$PROJECT_HOME` in comments
- (SSA names like `v810`, `l_S_t_0_t` are deterministic for fixed input — keep them; a change
  there is a real change)

```python
# tests/dataflow/regression/snapshot.py  (sketch)
import re, sys, difflib, importlib, os
import allo.dataflow as df

def emit(mod_name, region_name, prj):
    region = getattr(importlib.import_module(mod_name), region_name)
    df.build(region, target="systemc", mode="csyn", project=prj)  # writes kernel.cpp, no Catapult
    return open(os.path.join(prj, "kernel.cpp")).read()

_NORM = [(re.compile(r'^//\s*Generated date:.*$', re.M), '// Generated date: <stripped>'),
         (re.compile(r'/[^ \n]*/kernel\.cpp'), '<path>/kernel.cpp')]
def normalize(s):
    for pat, rep in _NORM: s = pat.sub(rep, s)
    return s

def check(case):                       # case = (name, module, region)
    name, mod, region = case
    got = normalize(emit(mod, region, f"/tmp/snap/{name}"))
    gold_path = f"snapshots/goldens/{name}.cpp"
    if os.environ.get("UPDATE_GOLDENS"):
        open(gold_path, "w").write(got); return "UPDATED"
    want = normalize(open(gold_path).read())
    if got == want: return "PASS"
    sys.stdout.writelines(difflib.unified_diff(
        want.splitlines(True), got.splitlines(True), gold_path, "emitted", n=3))
    return "FAIL"
```

Update goldens deliberately: `UPDATE_GOLDENS=1 python snapshot.py`, then **read the diff in
code review** — a golden change is the emitter‑behavior change, made reviewable.

Runs in seconds, no Catapult/Xcelium license, no `__SYNTHESIS__` compile. This is what you run
before every push.

---

## Tier B — cosim equivalence sweep (slow, authoritative)

**Idea.** Run `mode="cosim"` on every dataflow design and compare the verdict to a checked‑in
**baseline manifest**. A regression = a design whose status *worsened* (MATCH→mismatch, or
build/synth error that used to pass). Known‑broken designs are `xfail` in the manifest, so the
sweep is green when reality matches the baseline — not only when everything passes.

**Manifest** (`manifest.yaml`) — the expected state of the world:

```yaml
# status: pass | xfail:<reason> | skip:<reason>
- {design: test_producer_consumer, region: top,        status: pass}
- {design: test_pingpong_gemm,     region: top,        status: pass}
- {design: rvn_router,             region: rvn_router,  status: pass, env: {NVC: "2", ALLO_DESIGN_TOP: router_0}}
- {design: test_mlp,               region: top,        status: xfail: "float int->float conv (systemc-csynth-status)"}
- {design: test_smith_waterman,    region: top,        status: skip: "cosim >20min"}
defaults: {timeout_s: 900, seeds: [0]}
```

**Runner** (sketch — one subprocess per design, bounded parallelism, parse the verdict):

```python
# tests/dataflow/regression/cosim_sweep.py  (sketch)
import subprocess, yaml, concurrent.futures as cf, os

VERDICT = {"cosim MATCH": "pass"}         # anything else with the design present = mismatch
def run_one(row):
    env = {**os.environ, **row.get("env", {})}
    # each design's test drives df.build(mode="cosim"); reuse csyn_subdir-style build-subdir
    p = subprocess.run(["conda","run","-n","allo","python", row["driver"]],
                       env=env, capture_output=True, text=True,
                       timeout=row.get("timeout_s", 900))
    out = p.stdout + p.stderr
    if "cosim MATCH" in out:                     got = "pass"
    elif "Unknown path" in out or "SCHD-30" in out: got = "synth_error"
    elif "MISMATCH" in out or "diff" in out:     got = "mismatch"
    else:                                        got = "build_error"
    return row["design"], got

def main():
    rows = yaml.safe_load(open("manifest.yaml"))
    exp  = {r["design"]: r["status"].split(":")[0] for r in rows if "design" in r}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:      # Catapult license-bound; keep low
        results = dict(ex.map(run_one, [r for r in rows if not r["status"].startswith("skip")]))
    regressions = [d for d, got in results.items()
                   if exp[d] == "pass" and got != "pass"]        # was passing, now not
    fixed       = [d for d, got in results.items()
                   if exp[d].startswith("xfail") and got == "pass"]  # xfail now passes -> update
    print("REGRESSIONS:", regressions or "none")
    print("NEWLY-PASSING (update manifest):", fixed or "none")
    raise SystemExit(1 if regressions else 0)
```

**Non‑determinism.** Some designs are legitimately non‑deterministic (arrival order under
arbitration; see the `nb_nondeterminism` designs). For those the golden compare must be
**order‑tolerant** (multiset / per‑flow order), exactly like `replay_ref.py`'s payload‑set
check — not a positional diff. Mark them `compare: multiset` in the manifest and have the
design's cosim checker use the tolerant comparison. cosim's built‑in SCVerify diff is
positional, so these designs use the design‑level replay checker instead of the raw SCVerify
verdict.

**Concurrency / license.** Catapult + Xcelium are license‑ and CPU‑heavy; cap `max_workers`
low (≈4) and kill the stale‑Catapult‑holds‑license failure mode (seen this session — a leftover
`synth/base` catapult blocked new runs). Always build in a subdir (the SCHD‑30/degraded‑port
trap).

---

## What gates what

- **pre‑push (fast):** Tier A snapshots + the two golden csim unit tests already in CLAUDE.md
  (`test_df_unit.py`, `test_region_stateful.py`). Seconds. Blocks obviously‑wrong emitter edits.
- **pre‑merge / nightly (slow):** Tier B cosim sweep. A single regression (a design that *was*
  MATCH and now isn't) fails the gate and names the design. `xfail→pass` prints a "update the
  manifest" note rather than failing.
- **on an emitter change specifically:** run BOTH, and require the Tier‑A golden diff to be
  reviewed (it *is* the behavior change) and Tier‑B to show no regressions.

## Why this catches the specific risk

The loop‑shape transform is the poster child: it only changes the `#ifdef __SYNTHESIS__` arm, so
csim is byte‑identical and a csim‑only test sees nothing. **Tier A** catches it immediately (the
golden `kernel.cpp` gains a `while(1)` arm — a reviewable diff), and **Tier B** proves the new
RTL still equals csim on real stimulus (the router `COSIM_PASS` we ran by hand becomes an
automated row). Every future `#ifdef` split inherits the same coverage for free.

## Incremental build‑out (do it in this order)

1. Write the Tier‑A snapshot runner + the 10 canonical designs + capture initial goldens.
   (Half a day; immediately useful, no license needed.)
2. Turn this session's manual runs into manifest rows + the Tier‑B runner
   (`rvn_router` NVC=2, `test_producer_consumer`, `test_pingpong_gemm`, …). Seed from the known
   upstream 15/21‑pass baseline (see the `dataflow systemc cosim sweep` note).
3. Wire both into `make regress-fast` / `make regress-cosim`; document `UPDATE_GOLDENS=1`.
4. Backfill `xfail` reasons from the memory notes (float csynth, char‑width, etc.).
