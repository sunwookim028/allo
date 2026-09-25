# The paid path, exercised at $0 — 2026-09-25

Everything between a clean checkout and a first paid run, run for real on this
host, **spending nothing**. The resumability work made no cloud call at all, so
`gcp_setup.sh`, `preflight.py`'s gcloud reads, `smoke.py` and the caps firing
were all unexercised. This is what could be cleared without money, and what
could not.

Worktree at `origin/main` `9cedd143`, its own `mlir/build`, its own
`chia.env` copied from the committed `chia.env.example` and deleted afterwards.

## What was exercised, and what it found

| thing | result |
| --- | --- |
| `gcp_setup.sh` (no flags) | **OK**, exit 0, no global gcloud config or ADC setting changed. ADC present; quota project left at `test-adrs` and overridden per process. `chia2026-tinytpu` (#762944961825) ACTIVE, linked to CHIA2026 (`01BF39-94AA3F-36BACB`, open), `billingEnabled=True`; `aiplatform` and `billingbudgets` both enabled. |
| `preflight.py --budget-usd 30` | **OK** — the first time its three gcloud reads have run. $124.26 spent over 37 sessions, cap $500, remaining $375.74. |
| `spend.py report .` | $124.26 / 37 sessions since the cutover; run 1 $28.54, run 3 $20.72, ~$75.00 attributed to no run. $221.31 of historical spend correctly held on `test-adrs`, outside the cap. |
| `test_harness.py` (all nine phases) | **101/101 cases pass, 74.5 min**, exit 0. `harness-test-20260925-033220/results.json`. `a.noop` scored exactly 175 / 674 --- the row derived from `reproduce.sh`, not a literal --- and `accept.py`'s own control, measured on a worktree it built itself, reproduced all five: 175 / 265 / 421 / 482 / 674 |

**The GCP budget on the account is an alert, not a cap.** `gcp_setup.sh`
reports it plainly: *"CHIA2026 credit consumption": 900 USD per custom period,
alerts at 50 %, 90 %, 100 %*. Nothing at Google stops a run. The only ceiling
that stops anything is `CHIA_TOTAL_CAP_USD` in this repository, which makes the
refusals below the whole of the protection.

## The caps, shown refusing

A cap that has never refused is not a cap. Each line below is a real refusal
from the real code, at $0.

**The cumulative ceiling.** Lowering `CHIA_TOTAL_CAP_USD` is one test; the
stronger one is the **real $500 ceiling** against a fabricated opencode
database (`OPENCODE_DB` pointed at a scratch SQLite file holding seven
`google-vertex` sessions after the cutover, $490 in total — no money, and the
live database untouched):

| | |
| --- | --- |
| real ceiling $500, $490 "spent", `--budget-usd 30` | **REFUSED**, remaining $10.00 |
| real ceiling $500, $490 "spent", `--budget-usd 9` | allowed — the boundary is arithmetic, not luck |
| `CHIA_TOTAL_CAP_USD=130`, live $124.26, `--budget-usd 30` | **REFUSED**, remaining $5.74 |
| `CHIA_TOTAL_CAP_USD=154.26`, live $124.2602, `--budget-usd 30` | **REFUSED** — the fractional cents count |

**The per-run cap.** `--budget-usd` absent, `0`, `-5`, `lots` and `inf` are all
**REFUSED** before any network call: the cap is checked first because it needs
none. `loop.Budget.check` refuses on the same fabricated $490 run at caps of
$490 and $100 and allows $495 and $600 — the boundary is
`spent + largest_call_seen > cap` with `DEFAULT_CALL_USD = $3.50`.
`swarm.py`'s hard cap (`spent >= budget_usd`, which kills every worker) fires at
$490 and $489.99 and not at $600.

*One sharp edge found:* `Budget.check` applies `largest_call` even when
`billable` is false, so the **scripted test model refuses to start under a cap
below $3.50** although it cannot be billed at all. `test_harness.py` passes
`--budget-usd 5` and is unaffected; a future $0 caller passing less would be
refused for no reason.

## The project pin holds, and an environment variable cannot move it

Paid runs are CHIA2026 only. Every attempt below was **REFUSED**:

| attempt | refused by |
| --- | --- |
| `CHIA_BILLING_ACCOUNT` set to the general `test-adrs` account | the billing link: `chia2026-tinytpu` bills CHIA2026, not the configured account |
| `GOOGLE_CLOUD_PROJECT=test-adrs` | the same check, from the other side |
| **both** switched together (a consistent lie) | `billing.json`, which is **tracked**: "record a new cutover before switching accounts" |
| `GOOGLE_VERTEX_PROJECT` disagreeing with `GOOGLE_CLOUD_PROJECT` | the explicit disagreement check |
| `TINYTPU_OPENCODE_BASE_URL=https://evil.example/v1` (to skip the billing checks) | loopback-only |
| `TINYTPU_OPENCODE_MODEL=anthropic/claude` | "this gate only knows how Vertex AI bills" |
| `GOOGLE_CLOUD_PROJECT` unset | "source chia.env" |

The load-bearing one is the third: the account cannot be redirected by
environment alone, because `billing.json` is a committed file and the gate
compares the two. `TINYTPU_OPENCODE_BASE_URL` on loopback is the one documented
bypass and it is sound — it cannot reach Vertex.

**What is *not* pinned:** `CHIA_TOTAL_CAP_USD` can be raised arbitrarily by an
environment variable, with no ceiling on the ceiling. That is deliberate and
documented ("raise it here; no code change needed"), but it means the $500
figure is a convention held by whoever writes `chia.env`, not an invariant. If
the owner wants it to be one, the fix is a committed maximum that `chia.env`
may lower but not raise.

## A trap that cost two harness runs, and is now written down

The harness was first run from a worktree in the agent scratchpad, under
`/tmp`. **Every candidate died at stage `import`** with
`ModuleNotFoundError: No module named 'allo.compose'` -- which reads as a
broken design and is nothing of the kind. `evaluate.sandboxed` gives each gate
`--tmpfs /tmp` and re-binds only the work directory and the composed tree, so
the checkout itself is invisible; `allo` is not composed into the tree, so the
import falls through to the env's editable install, which on this host points
at `/home/sk3463/allo` -- a tree with no `allo/compose.py`. Copying a prebuilt
`mlir/build` into a `/home` worktree does not fix it either: the bindings' RPATH
still points into the old build directory, so the sandbox raises
`ImportError: libAlloMLIRAggregateCAPI.so.22.0git`. The fix is a worktree under
`/home` with `mlir/` built in it. Recorded in `docs/source/extensions/chia.rst`
under "Limits and known failures".

## What only money can clear

- **`smoke.py`.** One real Vertex call, ~$0.02–0.05. Its first stage (the
  frozen gate on an unmodified spec) is free; the call, the ADC → Vertex
  handshake, opencode's Vertex provider taking the project from `loop.py`'s
  provider options, and the reply coming back through the MCP tool are not.
  **Nothing here proves a token has ever been billed to CHIA2026 by the current
  tree.**
- **A worker iteration driven by a real model**, and therefore: the 2400 s
  timeout path, a session title reaching opencode's database so a cap can find
  it, `spend.py run <dir>` on a live run, and a graded verdict from something
  other than the scripted model.
- **A cap firing against real spend.** Every refusal above was arithmetic over
  a fabricated or lowered figure. The arithmetic is now proven; the loop
  between a *billed* call, opencode's database and the cap closing has not been
  closed since run 3.
- **The remaining credit balance**, which neither gcloud nor the Billing API
  exposes: Cloud Console → Billing → CHIA2026 → Credits.
