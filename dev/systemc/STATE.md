# Project state — `/home/zsm9/allo_sup`

Last updated 2026-08-12.

## Remotes

| Remote | Points at | Role |
|---|---|---|
| `mine` | `choonsik1/allo` | Your fork. Where this work is pushed. |
| `origin` | `cornell-zhang/allo` | Upstream baseline. |
| `sup` | `sunwookim028/allo` | The fork this checkout was originally taken from (hence `allo_sup`). |
| `fangtang`, `vincent` | collaborators' forks | Read-only reference. |

## Local branches

| Branch | Role |
|---|---|
| `SystemC-emitter` | **Current working branch** — the SystemC/Catapult emitter work. |
| `wire` | Predecessor line: Wire/Channel link types, the emitter fix series. |
| `main` | Integration branch. |
| `ext/sup-main`, `ext/x1-connections-backend` | Imported external lines. |
| `fix/nb-stream-scalar` | Non-blocking stream scalar fix. |
| `ip-stream-integration` | IP/stream integration work. |

## Current work

The SystemC/Catapult backend and its evaluation. Recent commits on `SystemC-emitter`:
`#pragma hls_resource` for fully-partitioned local arrays, the `agents/noc` branch
cleanup, the emitter documentation pass, and `synth_top` + ns clock-period rendering.

**The evaluation lives outside this repo**, in `/home/zsm9/final_noc` (its own git repo):
NoC designs rebuilt in Allo and measured against MatchLib WHVCRouter, MatchLib
ArbitratedCrossbar and RaveNoC. Authoritative numbers are in
`final_noc/designs/whvcrouter/RESULTS.md` and
`final_noc/designs/router_rvn_equiv/reports/README.md`.

Headline (Genus 20.1 high effort, Nangate 45nm, 2.0 ns, `concat_rtl.v`, 0 black boxes):

| design | Allo | reference | verdict |
|---|---|---|---|
| WHVCRouter | 2 cyc/step, 510 MHz, 32,415 µm² | MatchLib 1 cyc, 509 MHz, 31,784 µm² | MatchLib 2.04× per area |
| Crossbar | 1 cyc/step, 746 MHz, 6,466 µm² | MatchLib 3 cyc, 775 MHz, 6,056 µm² | **Allo 2.71× per area** |
| RaveNoC-equivalent | 2 cyc/step, 606 MHz, 10,001 µm² | RaveNoC 1 cyc, 567 MHz, 7,857 µm² | RaveNoC 2.38× per area |

## Known gaps in the backend

- ~~**Stateful category**~~ — **DONE 2026-08-16.** `x: T @ Stateful` now emits in the
  SC_THREAD reset action (per-instance, reset-initialised) rather than as the Vitis
  function-scope `static`. Scalar + array, csim and cosim bit-exact:
  `tests/dataflow/test_stateful_systemc.py`.
- **Bit-slicing under csynth** — the `(hi,lo)` bit-range is csim-only, so packed-stream
  designs fail at their slice site. Root cause and the fix that would delete it are in
  `BACKEND.md`.
- **Bounded-loop pipelining** — a finite loop emits as a Catapult reset action and is
  therefore not pipelined; only run-forever loops get the `while(1)` shape.
- **Inner-loop directives** — the emitter builds the reset + `while(1)` + bounded-for
  shape but does not auto-label or emit UNROLL/PIPELINE on inner loops. Manual
  `s.pipeline` does work.
- **Induction-variable width inflation** — the frontend widens intermediates to 33/65
  bits (`x*2+1` on an `int32` yields 65 bits). Verified identical on `vhls` and
  `catapult`, so it is a frontend issue affecting **all** backends, not an emitter bug.

## Next candidates

1. Period bisection for real Fmax — every Fmax except the WHVCRouter pair is a lower
   bound (slack +192 to +1170 ps), because Genus stops optimising once the constraint
   is met.
2. Full dataflow sweep after the `hls_resource` emitter change (only
   `test_systemc_backend.py` was re-run: 26 passed, 2 pre-existing failures).
3. `rvn_router` II=1 — blocked by the `vmask → pdst → odst` arbiter recurrence.
4. WHVCRouter II=1 — blocked by `popm → bhd`.
5. Induction-variable width narrowing (see above) — a 34% area lever, and it would
   benefit every backend.

## Provenance note

`STATE.md` and `BRANCHES.md` as inherited from the `sup` fork described
`sunwookim028/allo`'s branch topology and PR queue, none of which exists in this
checkout. They are in `archive/` unchanged.
