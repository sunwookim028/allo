#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CWD=$(dirname "$(realpath "$0")")

STAGE=${1:-all}
case "$STAGE" in
  all | format | pylint) ;;
  *)
    echo "usage: $0 [all|format|pylint]" >&2
    exit 2
    ;;
esac

if [ "$STAGE" = "all" ] || [ "$STAGE" = "format" ]; then
  echo "Check license header..."
  python3 $CWD/check_license_header.py all

  echo "Check Python formats using black..."
  python3 $CWD/check_python_format.py

  echo "Check C/C++ formats using clang-format..."
  python3 $CWD/check_cpp_format.py
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "pylint" ]; then
  echo "Running pylint on allo"
  python3 -m pylint allo --rcfile=$CWD/.pylintrc
fi
