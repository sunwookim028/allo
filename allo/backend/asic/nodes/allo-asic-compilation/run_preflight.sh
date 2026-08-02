#!/usr/bin/env bash
set -eo pipefail

python_bin=""
setup_script=""
forwarded=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python-bin)
      python_bin="$2"
      forwarded+=("$1" "$2")
      shift 2
      ;;
    --setup-script)
      setup_script="$2"
      forwarded+=("$1" "$2")
      shift 2
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

if [ -z "$python_bin" ]; then
  echo "ERROR: missing --python-bin" >&2
  exit 2
fi
if [ -z "$setup_script" ] || [ ! -f "$setup_script" ]; then
  echo "ERROR: Allo LLVM setup script does not exist: $setup_script" >&2
  exit 2
fi

# setup-llvm-main.sh unsets PYTHONPATH before appending to it and also prints
# optional variables such as LD_LIBRARY_PATH. It therefore must be sourced
# with Bash nounset disabled.
set +u
source "$setup_script"

exec "$python_bin" preflight.py "${forwarded[@]}"

