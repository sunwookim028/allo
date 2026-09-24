"""Pre-flight: refuse to start a paid run unless it will charge the right account.

Run by `swarm.py`, `loop.py` and `smoke.py` before any worker or model call.
It makes no model call and costs nothing: three `gcloud ... describe/list`
reads, a file-existence check, and arithmetic. It refuses unless

  (a) the configured project (`GOOGLE_CLOUD_PROJECT`) is linked to the
      configured billing account (`CHIA_BILLING_ACCOUNT`, CHIA2026 by default)
      and billing is enabled on it,
  (b) `aiplatform.googleapis.com` is enabled on that project, and
  (c) a positive, finite spend cap has been given for the run, and
  (d) CHIA's cumulative spend on CHIA2026 so far plus this run's cap stays
      within `CHIA_TOTAL_CAP_USD` ($100 unless chia.env says otherwise). The
      cumulative figure is opencode's own record, attributed to CHIA2026 by
      the cutover time in `billing.json` -- see `spend.py` for why by time,

and then prints the account and project the run will charge, the spend so
far, what remains, and this run's cap.

Why the billing link is checked and not just the project name: Vertex AI bills
the project named in the request URL, which opencode takes from the provider
options `loop.py` sets from GOOGLE_CLOUD_PROJECT (see `vertex_provider`). A
project that exists but bills to another account -- `test-adrs` bills to a
general account shared with unrelated work -- would run perfectly and charge
the wrong people.

Nothing global is read or written: every gcloud call gets its project from the
command line and `CLOUDSDK_BILLING_QUOTA_PROJECT=LEGACY` in ITS environment
only, so neither the user's gcloud config (core/project = test-adrs) nor the
ADC quota project is consulted or changed.

The scripted test model (`TINYTPU_OPENCODE_BASE_URL` on loopback, set only by
`test_harness.py`) cannot reach Vertex and costs nothing, so (a), (b) and (d)
are skipped for it; (c) still applies.

    python preflight.py --budget-usd 15          # exit 0 and the summary, or 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: CHIA2026, the account created for this work (2026-09-19).
DEFAULT_BILLING_ACCOUNT = "01BF39-94AA3F-36BACB"
#: Cumulative cap across every CHIA run on CHIA2026; chia.env can raise it.
DEFAULT_TOTAL_CAP_USD = 100.0
GCLOUD_TIMEOUT = 60


class PreflightError(RuntimeError):
    pass


def _gcloud(args: list[str], project: str) -> dict | list:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        raise PreflightError("gcloud is not on PATH")
    # Per-process only: the project for this call, and gcloud's own client
    # project for quota, so the user's global core/project is never used.
    env = dict(os.environ, CLOUDSDK_CORE_PROJECT=project,
               CLOUDSDK_BILLING_QUOTA_PROJECT="LEGACY")
    try:
        p = subprocess.run([gcloud, *args, "--format=json"], env=env,
                           capture_output=True, text=True, timeout=GCLOUD_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PreflightError(f"gcloud {' '.join(args[:3])} timed out") from None
    if p.returncode:
        err = " ".join(p.stderr.split())[:400]
        raise PreflightError(f"gcloud {' '.join(args[:3])} failed: {err}")
    return json.loads(p.stdout or "null")


def _adc_present() -> bool:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or str(
        Path.home() / ".config/gcloud/application_default_credentials.json")
    return Path(path).is_file() and Path(path).stat().st_size > 0


def check(budget_usd, *, run_t0_ms: int | None = None, quiet: bool = False) -> dict:
    """Raise PreflightError, or return what the run will charge.

    `run_t0_ms` is when the run started (swarm.py passes it to its workers):
    "spent so far" counts only sessions before it, so a worker launched after
    its siblings have spent is judged against the same total as the run."""
    # (c) the spend cap, first: it needs no network.
    try:
        cap = float(budget_usd)
    except (TypeError, ValueError):
        cap = float("nan")
    if not (math.isfinite(cap) and cap > 0):
        raise PreflightError(f"no spend cap for this run (got {budget_usd!r}); pass "
                             f"--budget-usd with a positive dollar amount")
    model = os.environ.get("TINYTPU_OPENCODE_MODEL",
                           "google-vertex/gemini-3.1-pro-preview")
    base_url = os.environ.get("TINYTPU_OPENCODE_BASE_URL")
    if base_url:
        host = urlparse(base_url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise PreflightError(f"TINYTPU_OPENCODE_BASE_URL={base_url} is not "
                                 f"loopback; only the local scripted test model may "
                                 f"bypass the billing checks")
        info = {"mode": "test-model", "model": model, "base_url": base_url,
                "budget_usd": cap}
        if not quiet:
            print(f"pre-flight: scripted test model at {base_url}; no cloud model "
                  f"can be reached, billing checks skipped; spend cap ${cap:.2f}",
                  flush=True)
        return info

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    account = os.environ.get("CHIA_BILLING_ACCOUNT", DEFAULT_BILLING_ACCOUNT)
    if not project:
        raise PreflightError("GOOGLE_CLOUD_PROJECT is unset (source chia.env)")
    if not model.startswith("google-vertex/"):
        raise PreflightError(f"TINYTPU_OPENCODE_MODEL={model} is not a google-vertex "
                             f"model; this gate only knows how Vertex AI bills")
    # opencode's google-vertex provider takes the project from the provider
    # options first, then GOOGLE_VERTEX_PROJECT, then GOOGLE_CLOUD_PROJECT.
    # loop.py sets the option; a disagreeing variable is a config error.
    other = os.environ.get("GOOGLE_VERTEX_PROJECT")
    if other and other != project:
        raise PreflightError(f"GOOGLE_VERTEX_PROJECT={other} disagrees with "
                             f"GOOGLE_CLOUD_PROJECT={project}")
    if not _adc_present():
        raise PreflightError("no Application Default Credentials; run "
                             "`gcloud auth application-default login` (see gcp_setup.sh)")

    proj = _gcloud(["projects", "describe", project], project)
    if proj.get("lifecycleState") != "ACTIVE":
        raise PreflightError(f"project {project} is {proj.get('lifecycleState')}")
    # (a) the billing link.
    bill = _gcloud(["billing", "projects", "describe", project], project)
    linked = (bill.get("billingAccountName") or "").removeprefix("billingAccounts/")
    if linked != account:
        raise PreflightError(
            f"project {project} bills to billing account {linked or '(none)'}, not "
            f"the configured CHIA_BILLING_ACCOUNT {account}; refusing to charge it")
    if not bill.get("billingEnabled"):
        raise PreflightError(f"billing is not enabled on {project}")
    acct = _gcloud(["billing", "accounts", "describe", account], project)
    if not acct.get("open"):
        raise PreflightError(f"billing account {account} is closed")
    # (b) the API.
    apis = _gcloud(["services", "list", "--enabled", "--project", project,
                    "--filter=config.name=aiplatform.googleapis.com"], project)
    if not any(s.get("config", {}).get("name") == "aiplatform.googleapis.com"
               for s in apis or []):
        raise PreflightError(f"aiplatform.googleapis.com is not enabled on {project} "
                             f"(gcp_setup.sh --enable-apis)")
    # (d) the cumulative cap.
    from spend import billing, chia2026_spend
    if billing()["current"]["billing_account"] != account:
        raise PreflightError(f"billing.json attributes spend to "
                             f"{billing()['current']['billing_account']}, but "
                             f"CHIA_BILLING_ACCOUNT is {account}; record a new "
                             f"cutover in billing.json before switching accounts")
    total_cap = float(os.environ.get("CHIA_TOTAL_CAP_USD", DEFAULT_TOTAL_CAP_USD))
    so_far = chia2026_spend(run_t0_ms)
    remaining = total_cap - so_far["usd"]
    if so_far["usd"] + cap > total_cap:
        raise PreflightError(
            f"CHIA has spent ${so_far['usd']:.2f} on {account} "
            f"({so_far['sessions']} sessions since the cutover); this run's cap "
            f"${cap:.2f} would exceed CHIA_TOTAL_CAP_USD=${total_cap:.2f} "
            f"(remaining ${remaining:.2f}). Lower --budget-usd or raise the cap "
            f"in chia.env")
    info = {"mode": "vertex", "project": project,
            "project_number": proj.get("projectNumber"),
            "billing_account": account,
            "billing_account_name": acct.get("displayName"),
            "model": model,
            "location": os.environ.get("TINYTPU_VERTEX_LOCATION", "global"),
            "budget_usd": cap, "total_cap_usd": total_cap,
            "spent_so_far_usd": so_far["usd"], "sessions_so_far": so_far["sessions"],
            "remaining_usd": round(remaining, 4)}
    if not quiet:
        print(f"pre-flight OK: this run will charge billing account {account} "
              f"({info['billing_account_name']}) through project {project} "
              f"(#{info['project_number']}); model {model} @ {info['location']}\n"
              f"  CHIA spend on {account} so far ${so_far['usd']:.2f} "
              f"({so_far['sessions']} sessions, opencode's figures); cap "
              f"${total_cap:.2f}; remaining ${remaining:.2f}; this run's cap "
              f"${cap:.2f}", flush=True)
    return info


def require(budget_usd, run_t0_ms: int | None = None) -> dict:
    """check(), or print why not and exit 1."""
    try:
        return check(budget_usd, run_t0_ms=run_t0_ms)
    except PreflightError as e:
        print(f"pre-flight REFUSED: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-usd", default=os.environ.get("CHIA_BUDGET_USD"))
    a = ap.parse_args()
    print(json.dumps(require(a.budget_usd)))
