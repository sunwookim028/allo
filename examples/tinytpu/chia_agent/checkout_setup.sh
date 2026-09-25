#!/usr/bin/env bash
# Take a clean checkout to a runnable CHIA loop. Idempotent; safe to re-run; $0.
#
#   examples/tinytpu/chia_agent/checkout_setup.sh [--env FILE]
#
# The companion of `gcp_setup.sh`: that one checks the CLOUD side (login, ADC,
# project, billing, APIs); this one checks THE CHECKOUT, which is everything a
# fresh `git worktree` is missing. Run this first, then gcp_setup.sh.
#
# It exists because a clean checkout hit five separate blockers before a single
# model call could be made, one of them by hanging rather than erroring. Each
# step below is one of them:
#
#   1. chia.env      created from chia.env.example if ABSENT. An existing one
#                    is never overwritten, and one naming a different project
#                    stops the script -- that is the case that would bill the
#                    wrong account. Its contents are never printed.
#   2. opencode      `npm ci --prefix examples/tinytpu/chia_agent` if
#                    $OPENCODE_BIN/opencode is missing (~6 s warm, 725 MB).
#   3. .chia_scratch created; smoke.py and the evaluator mkdtemp inside it.
#   4. Ray           the address `ray.init(address="auto")` would really dial
#                    is alive (`preflight.py --check-ray`). A stale
#                    /tmp/ray/ray_current_cluster makes every driver BLOCK
#                    FOREVER instead of failing; it has cost three sessions.
#   5. mlir/build    reported, never built: an 8-minute build is not something
#                    a setup script should start behind your back. It prints
#                    `examples/tinytpu/reproduce.sh`, which builds it.
#   6. chia_env      the py3.10 env that runs the loop (ray + chialoops).
#
# It changes nothing outside the checkout: no global gcloud config, no conda
# env, no Ray process, no file under /tmp. Every step says what it is doing and
# what it found; every check runs even after an earlier one fails, so one
# invocation reports all of what is missing rather than one blocker per run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
ENV_FILE="$REPO/chia.env"
EXAMPLE="$HERE/chia.env.example"
while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_FILE="$2"; shift ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
FAILED=()
fail() { echo "   FAIL: $1"; FAILED+=("$2"); }
# The project a chia.env names, read in a subshell so nothing leaks into ours
# and nothing but the project name is ever printed.
project_in() { (set -a; . "$1"; set +a; printf '%s' "${GOOGLE_CLOUD_PROJECT:-}"); }

hr; echo "1. chia.env ($ENV_FILE)"
WANT="$(project_in "$EXAMPLE")"
if [ ! -f "$ENV_FILE" ]; then
  cp "$EXAMPLE" "$ENV_FILE"
  echo "   created from $(basename "$EXAMPLE") (gitignored; NEVER commit it)."
  echo "   It is complete for the \$0 path; a paid run also needs ADC, which"
  echo "   gcp_setup.sh checks. Its contents are not printed here."
  echo "   project: $WANT"
else
  HAVE="$(project_in "$ENV_FILE")"
  echo "   exists; left untouched (contents not printed)."
  if [ "$HAVE" != "$WANT" ]; then
    echo
    echo "   ####################################################################"
    echo "   ## STOP. $(basename "$ENV_FILE") names project '${HAVE:-(unset)}', not '$WANT'."
    echo "   ##"
    echo "   ## A run charges the project in this file. This is the case that"
    echo "   ## would bill the wrong account -- the host's old chia.env named"
    echo "   ## 'test-adrs', a general account shared with unrelated work."
    echo "   ## preflight.py would refuse the run, correctly; nothing here will"
    echo "   ## guess which one you meant."
    echo "   ##"
    echo "   ## Either edit $ENV_FILE, or move it aside and re-run this script"
    echo "   ## to get a fresh copy of $(basename "$EXAMPLE")."
    echo "   ####################################################################"
    exit 1
  fi
  echo "   project: $HAVE (matches $(basename "$EXAMPLE"))"
fi
set -a; . "$ENV_FILE"; set +a
echo "   variables set for the rest of this script (names only, no values):"
echo "     $(grep -oE '^export [A-Z_0-9]+' "$EXAMPLE" | cut -d' ' -f2 | tr '\n' ' ')"

hr; echo "2. opencode"
OPENCODE_BIN="${OPENCODE_BIN:-$HERE/node_modules/.bin}"
if [ -x "$OPENCODE_BIN/opencode" ]; then
  echo "   present: $OPENCODE_BIN/opencode ($("$OPENCODE_BIN/opencode" --version 2>/dev/null || echo '?'))"
else
  echo "   missing at $OPENCODE_BIN/opencode; it is a PER-WORKTREE install."
  if ! command -v npm >/dev/null; then
    fail "npm is not on PATH; install Node, then: npm ci --prefix $HERE" opencode
  else
    echo "   running: npm ci --prefix $HERE"
    if npm ci --prefix "$HERE" >/dev/null 2>&1 && [ -x "$OPENCODE_BIN/opencode" ]; then
      echo "   installed: $("$OPENCODE_BIN/opencode" --version 2>/dev/null || echo '?')"
    else
      fail "npm ci failed; re-run it by hand: npm ci --prefix $HERE" opencode
    fi
  fi
fi

hr; echo "3. .chia_scratch"
mkdir -p "$REPO/.chia_scratch"
echo "   ready: $REPO/.chia_scratch (gitignored; smoke.py and the evaluator"
echo "   mkdtemp inside it, and die at tempfile.mkdtemp if it is absent)"

hr; echo "4. Ray -- the blocker that HANGS instead of failing"
if RAY_OUT="$(python3 "$HERE/preflight.py" --check-ray 2>&1)"; then
  printf '%s\n' "$RAY_OUT" | grep -v '^{' | sed 's/^/   /'
else
  printf '%s\n' "$RAY_OUT" | sed 's/^/   /'
  fail "the Ray address a driver would dial is not usable (above)." ray
fi

hr; echo "5. mlir/build (this checkout's bindings)"
PY="${TINYTPU_ALLO_PYTHON:-$HOME/miniconda3/envs/allo/bin/python}"
if [ ! -x "$PY" ]; then
  fail "TINYTPU_ALLO_PYTHON=$PY is not executable (the allo env's python)" mlir
elif [ ! -f "$REPO/mlir/build/build.ninja" ]; then
  fail "no $REPO/mlir/build. Each worktree needs its own; build it with
         examples/tinytpu/reproduce.sh
       (~8 min cold, and it runs the functional gates too). Not started here:
       a setup script should not spend eight minutes without being asked." mlir
elif PYTHONPATH="$REPO" "$PY" -c 'import allo.dataflow' >/dev/null 2>&1; then
  echo "   present and importable: $REPO/mlir/build"
  echo "   ($("$PY" -c "import sys;sys.path.insert(0,'$REPO');import allo;print(allo.__file__)"))"
else
  fail "$REPO/mlir/build exists but \`import allo.dataflow\` fails against it --
       stale bindings (the WireConstructOp guard) or a different LLVM. Rebuild:
         examples/tinytpu/reproduce.sh" mlir
fi

hr; echo "6. chia_env (py3.10: ray + chialoops)"
CHIA_PY="$(conda env list 2>/dev/null | awk '$1=="chia_env"{print $NF"/bin/python"}')"
if [ -z "${CHIA_PY:-}" ] || [ ! -x "$CHIA_PY" ]; then
  fail "no chia_env. Create it:
         conda create -n chia_env python=3.10 -y
         conda run -n chia_env pip install -r $HERE/requirements.txt" chia_env
elif "$CHIA_PY" -c 'import ray, chia' >/dev/null 2>&1; then
  echo "   present: $CHIA_PY (ray $("$CHIA_PY" -c 'import ray;print(ray.__version__)'), chia importable)"
else
  fail "$CHIA_PY exists but \`import ray, chia\` fails. Install:
         conda run -n chia_env pip install -r $HERE/requirements.txt" chia_env
fi

hr
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "checkout_setup: NOT READY -- ${#FAILED[@]} blocker(s): ${FAILED[*]}"
  echo "Each is printed above with the exact command. Re-run this script after"
  echo "fixing them; everything it did already is idempotent."
  exit 1
fi
cat <<EOF
checkout_setup: OK -- this checkout can run the loop. Next, in order:

  examples/tinytpu/chia_agent/gcp_setup.sh   # the cloud side: auth, billing, \$0
  conda activate chia_env
  set -a; source chia.env; set +a            # BEFORE ray start: workers inherit
                                             # the raylet's environment
  ray start --head --temp-dir=/tmp/ray-<track> \\
            --resources='{"opencode_creds": 2}' --include-dashboard=false
  export RAY_ADDRESS=<what ray start printed>
  cd examples/tinytpu/chia_agent
  python test_harness.py --phases e,c        # ~2 min, \$0
  python preflight.py --budget-usd 5         # the gate alone, \$0

The whole procedure, including pre-registration, is the "Quick start, assuming
nothing" section of examples/tinytpu/chia_agent/README.md.
EOF
