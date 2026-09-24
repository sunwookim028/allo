#!/usr/bin/env bash
# GCP setup and billing report for the CHIA loop. Idempotent; safe to re-run.
#
#   examples/tinytpu/chia_agent/gcp_setup.sh [--enable-apis] [--env FILE]
#
# What it does, in order, and what it never does:
#
#   1. gcloud login and Application Default Credentials: CHECKED. If either is
#      missing it prints the exact command to run and stops. It never runs an
#      interactive login itself.
#   2. The project (GOOGLE_CLOUD_PROJECT) exists and is ACTIVE, and is linked
#      to the billing account CHIA_BILLING_ACCOUNT with billing enabled.
#      A mismatch is reported, not fixed: relinking billing is a decision for
#      a person (`gcloud billing projects link PROJECT --billing-account=ID`).
#   3. aiplatform.googleapis.com is enabled on the project. With --enable-apis
#      it is enabled if it is not, together with billingbudgets.googleapis.com
#      (free; lets step 4 list the budgets with the CHIA project as the quota
#      project rather than the user's default project). Without the flag,
#      nothing is enabled.
#   4. Report: billing account id and name, project, the budget(s) on that
#      account, and CHIA's spend to date from opencode's session DB -- on
#      CHIA2026 since the cutover (against CHIA_TOTAL_CAP_USD) and, separately,
#      the historical spend that billed test-adrs.
#
# It never changes global gcloud configuration (`gcloud config set ...`) or the
# ADC quota project: other work on this host relies on both (core/project and
# the ADC quota project are test-adrs). Every gcloud call here gets its project
# on the command line or in its own environment (CLOUDSDK_CORE_PROJECT,
# CLOUDSDK_BILLING_QUOTA_PROJECT), which ends with the call. It never prints an
# access token or the contents of chia.env.
#
# Config comes from chia.env at the repository root (or --env FILE), read in
# this process only: GOOGLE_CLOUD_PROJECT, CHIA_BILLING_ACCOUNT,
# CHIA_TOTAL_CAP_USD.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
ENV_FILE="$REPO/chia.env"
ENABLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --enable-apis) ENABLE=1 ;;
    --env) ENV_FILE="$2"; shift ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "config: $ENV_FILE (values not printed except the project, account and cap)"
else
  echo "config: no $ENV_FILE; using the environment"
fi
PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
ACCOUNT="${CHIA_BILLING_ACCOUNT:-01BF39-94AA3F-36BACB}"
CAP="${CHIA_TOTAL_CAP_USD:-100}"
if [ -z "$PROJECT" ]; then
  echo "FAIL: GOOGLE_CLOUD_PROJECT is not set (copy chia.env.example to chia.env)" >&2
  exit 1
fi

# gcloud, scoped to this project for this call only; quota on gcloud's own
# client ("LEGACY"), so the user's default project is not involved.
g() { CLOUDSDK_CORE_PROJECT="$PROJECT" CLOUDSDK_BILLING_QUOTA_PROJECT=LEGACY gcloud "$@"; }
hr() { printf '%s\n' "------------------------------------------------------------------------"; }

command -v gcloud >/dev/null || { echo "FAIL: gcloud is not on PATH" >&2; exit 1; }

hr; echo "1. credentials"
ACTIVE="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if [ -z "$ACTIVE" ]; then
  echo "   gcloud: NOT logged in. Run, in a terminal (interactive, opens a browser):"
  echo "       gcloud auth login"
  exit 1
fi
echo "   gcloud account: $ACTIVE"
ADC="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
if [ ! -s "$ADC" ]; then
  echo "   ADC: MISSING ($ADC). Run, in a terminal (interactive, opens a browser):"
  echo "       gcloud auth application-default login"
  echo "   Do NOT run 'gcloud auth application-default set-quota-project' for CHIA:"
  echo "   that rewrites the quota project other work on this host uses. CHIA sets"
  echo "   GOOGLE_CLOUD_QUOTA_PROJECT per process in chia.env instead."
  exit 1
fi
# A token is minted to prove the ADC works; it is discarded, never printed.
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "   ADC: present at $ADC but cannot mint a token (expired?). Run:"
  echo "       gcloud auth application-default login"
  exit 1
fi
ADC_QUOTA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("quota_project_id") or "(none)")' "$ADC" 2>/dev/null || echo '?')"
echo "   ADC: $ADC (works; its quota project is $ADC_QUOTA, left as is --"
echo "        CHIA overrides it per process with GOOGLE_CLOUD_QUOTA_PROJECT=${GOOGLE_CLOUD_QUOTA_PROJECT:-unset})"

hr; echo "2. project and billing link"
STATE="$(g projects describe "$PROJECT" --format='value(lifecycleState)' 2>/dev/null || true)"
if [ "$STATE" != "ACTIVE" ]; then
  echo "   FAIL: project $PROJECT not found or not ACTIVE (state: ${STATE:-unknown})"
  exit 1
fi
PNUM="$(g projects describe "$PROJECT" --format='value(projectNumber)')"
PNAME="$(g projects describe "$PROJECT" --format='value(name)')"
echo "   project: $PROJECT (#$PNUM, \"$PNAME\"), ACTIVE"
LINKED="$(g billing projects describe "$PROJECT" --format='value(billingAccountName)')"
LINKED="${LINKED#billingAccounts/}"
BENABLED="$(g billing projects describe "$PROJECT" --format='value(billingEnabled)')"
ANAME="$(g billing accounts describe "$ACCOUNT" --format='value(displayName)' 2>/dev/null || echo '?')"
AOPEN="$(g billing accounts describe "$ACCOUNT" --format='value(open)' 2>/dev/null || echo '?')"
echo "   configured billing account: $ACCOUNT (\"$ANAME\", open=$AOPEN)"
if [ "$LINKED" != "$ACCOUNT" ]; then
  echo "   FAIL: $PROJECT bills to ${LINKED:-(no account)}, not $ACCOUNT."
  echo "         To relink (a deliberate decision, not done here):"
  echo "         gcloud billing projects link $PROJECT --billing-account=$ACCOUNT"
  exit 1
fi
[ "$BENABLED" = "True" ] || { echo "   FAIL: billing is not enabled on $PROJECT"; exit 1; }
echo "   linked: $PROJECT -> $ACCOUNT, billingEnabled=True"

hr; echo "3. APIs on $PROJECT"
enabled() { g services list --enabled --project "$PROJECT" --filter="config.name=$1" --format='value(config.name)' 2>/dev/null; }
for API in aiplatform.googleapis.com billingbudgets.googleapis.com; do
  if [ -n "$(enabled "$API")" ]; then
    echo "   $API: enabled"
  elif [ "$ENABLE" = 1 ]; then
    echo "   $API: not enabled; enabling (--enable-apis) ..."
    g services enable "$API" --project "$PROJECT"
    echo "   $API: enabled"
  elif [ "$API" = aiplatform.googleapis.com ]; then
    echo "   FAIL: $API is not enabled; re-run with --enable-apis"
    exit 1
  else
    echo "   $API: not enabled (only needed to list budgets from $PROJECT; --enable-apis)"
  fi
done

hr; echo "4. report"
echo "   billing account : $ACCOUNT  \"$ANAME\""
echo "   project         : $PROJECT  (#$PNUM)"
echo "   budgets on $ACCOUNT (alerts only -- a budget does not cap spend):"
if [ -n "$(enabled billingbudgets.googleapis.com)" ]; then
  BUDGETS_JSON="$(g billing budgets list --billing-account="$ACCOUNT" \
                   --billing-project="$PROJECT" --format=json)" python3 - <<'PY'
import json, os
bs = json.loads(os.environ["BUDGETS_JSON"] or "[]")
if not bs:
    print("     (none)")
for b in bs:
    amt = b.get("amount", {}).get("specifiedAmount", {})
    f = b.get("budgetFilter", {})
    th = ", ".join(f"{t['thresholdPercent']:.0%}" for t in b.get("thresholdRules", []))
    scope = ", ".join(f.get("projects", [])) or "the whole account"
    print(f"     \"{b.get('displayName')}\": {amt.get('units', '?')} "
          f"{amt.get('currencyCode', '')} per {f.get('calendarPeriod', 'custom period').lower()}, "
          f"credits {f.get('creditTypesTreatment', '?')}, scope {scope}, alerts at {th}")
    if f.get("creditTypesTreatment") == "EXCLUDE_ALL_CREDITS":
        print("       (EXCLUDE_ALL_CREDITS: measures spend BEFORE credits, i.e. how much"
              " credit is being consumed; it resets each period)")
PY
else
  echo "     (not listed: billingbudgets.googleapis.com is not enabled on $PROJECT;"
  echo "      re-run with --enable-apis to list them without using another project)"
fi
echo
CHIA_TOTAL_CAP_USD="$CAP" python3 "$HERE/spend.py" report "$REPO" | sed 's/^/   /'
echo
echo "   The remaining CREDIT BALANCE is not exposed by gcloud or the Cloud Billing"
echo "   API. Read it in the Cloud Console: Billing -> $ANAME ($ACCOUNT) -> Credits."
hr; echo "gcp_setup: OK (no global gcloud config or ADC setting was changed)"
