#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the documentation and publish it to the fork's gh-pages branch, which
# GitHub serves at https://sunwookim028.github.io/allo/.
#
# Upstream builds its site in CI on a self-hosted runner that the fork does not
# have, so the fork's site is built locally and pushed. The site must match a
# commit that is on origin/main: the script refuses a dirty tree or an unpushed
# HEAD, and records the source commit in the gh-pages commit message.
#
# Prerequisites: the `allo` conda env active, LLVM_BUILD_DIR exported (see
# CLAUDE.md), and a Sphinx venv built from docs/requirements.txt:
#   python -m venv --system-site-packages ~/.cache/docs-tools/venv
#   ~/.cache/docs-tools/venv/bin/pip install -r docs/requirements.txt
#
# Usage: docs/publish.sh            build and publish
#        docs/publish.sh --dry-run  build only
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
VENV=${DOCS_VENV:-$HOME/.cache/docs-tools/venv}
REMOTE=${DOCS_REMOTE:-git@github.com:sunwookim028/allo.git}
URL=https://sunwookim028.github.io/allo/
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cd "$ROOT"
if [ -n "$(git status --porcelain)" ]; then
  echo "publish: working tree is not clean; the site must match a commit" >&2
  exit 1
fi
git fetch -q origin main
if [ "$DRY" = 0 ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "publish: HEAD is not origin/main; push first, then publish what is pushed" >&2
  exit 1
fi
SRC=$(git rev-parse --short HEAD)

# Tutorials are listed but not executed (sphinx-gallery's own switch).
(cd docs && rm -rf build &&
 make html SPHINXBUILD="$VENV/bin/sphinx-build" O="-D plot_gallery=0")
echo "publish: built docs/build/html from main@$SRC"
[ "$DRY" = 1 ] && exit 0

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
if ! git clone -q --depth 1 --branch gh-pages "$REMOTE" "$W" 2>/dev/null; then
  git init -q -b gh-pages "$W"
  git -C "$W" remote add origin "$REMOTE"
fi
find "$W" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r docs/build/html/. "$W"/
touch "$W/.nojekyll"   # directories such as _static must be served as-is
git -C "$W" add -A
if git -C "$W" diff --cached --quiet; then
  echo "publish: site unchanged, nothing to push"
  exit 0
fi
git -C "$W" commit -q -m "Publish docs from main@$SRC"
git -C "$W" push -q origin gh-pages
echo "publish: main@$SRC -> $URL (GitHub Pages rebuilds in about a minute)"
