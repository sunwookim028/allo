#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A pinned stand-in for Catapult's SystemC csim, for hosts without Catapult.
#
# target="systemc" emits SystemC for Siemens Catapult. On a Catapult host, csim
# compiles it against Catapult's own copies of SystemC, MatchLib Connections and
# ac_types under $MGC_HOME/shared. This script assembles open-source equivalents
# at pinned commits, in the same layout, so that the emitted design can be
# compiled and simulated functionally without a licence.
#
# It is a functional pre-check only. Nothing here synthesizes, schedules or
# produces RTL. The library versions match Catapult 2024.2's; the compiler does not.
# Catapult on a licence host is the authority.
#
# Usage:
#   scripts/systemc-csim-setup.sh            # fetch and assemble; print the env to export
#   eval "$(scripts/systemc-csim-setup.sh --env)"

set -euo pipefail

PREFIX="${ALLO_CSIM_HOME:-$HOME/.cache/allo/systemc-csim}"

# The versions Catapult 2024.2/1130128 bundles under $MGC_HOME/shared (read on
# zhang-21 on 2026-10-01), as hlslibs and Accellera tags, pinned to commits.
AC_TYPES_SHA=f542cd681bf388f98bc5676e9a8d12952c3e65db         # ac_types 4.9.0
AC_SIMUTILS_SHA=9aada6f55dc56b28ac74630ceb44f064deff427a      # ac_simutils 1.6.0
CONNECTIONS_SHA=6a3003b85c251c88dd2f02881bb98ce71f9aa42b      # matchlib_connections 2.2.0
SYSTEMC_SHA=38b8a2c61aafe40de44296bee0c9c28ac4e80b01          # Accellera SystemC 2.3.3
# Catapult builds its libsystemc with its own g++ 10.3.0; this uses the host's.
SYSTEMC_HOME="$PREFIX/systemc"

print_env() {
    echo "export MGC_HOME=$PREFIX/mgc"
    echo "export SYSTEMC_HOME=$SYSTEMC_HOME"
    echo "export ALLO_CXX_EXTRA=\"-DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM\""
}

if [ "${1:-}" = "--env" ]; then
    print_env
    exit 0
fi

fetch() {  # fetch <github org/repo> <sha> <dir>
    local repo=$1 sha=$2 dir=$3
    if [ ! -d "$dir/.git" ]; then
        git clone -q "https://github.com/$repo.git" "$dir"
    fi
    git -C "$dir" fetch -q origin "$sha" 2>/dev/null || git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q --detach "$sha"
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || { echo "error: $repo is not at $sha" >&2; exit 1; }
    echo "  $repo @ ${sha:0:7}"
}

mkdir -p "$PREFIX/src"
echo "== hlslibs (pinned)"
fetch hlslibs/ac_types "$AC_TYPES_SHA" "$PREFIX/src/ac_types"
fetch hlslibs/ac_simutils "$AC_SIMUTILS_SHA" "$PREFIX/src/ac_simutils"
fetch hlslibs/matchlib_connections "$CONNECTIONS_SHA" "$PREFIX/src/matchlib_connections"
fetch accellera-official/systemc "$SYSTEMC_SHA" "$PREFIX/src/systemc"

echo "== SystemC (built from source, C++17)"
stamp="$SYSTEMC_HOME/.built-$SYSTEMC_SHA"
if [ ! -f "$stamp" ]; then
    rm -rf "$PREFIX/build-systemc" "$SYSTEMC_HOME"
    step() {  # step <log> <cmd...>: run, and on failure show the log's tail
        local log=$1; shift
        "$@" > "$log" 2>&1 || { tail -20 "$log" >&2; echo "error: see $log" >&2; exit 1; }
    }
    # SystemC 2.3.3 declares cmake_minimum_required(3.1); CMake 4 needs the floor raised.
    step "$PREFIX/systemc-cmake.log" cmake -S "$PREFIX/src/systemc" -B "$PREFIX/build-systemc" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$SYSTEMC_HOME" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON
    step "$PREFIX/systemc-build.log" cmake --build "$PREFIX/build-systemc" -j "$(nproc)"
    step "$PREFIX/systemc-install.log" cmake --install "$PREFIX/build-systemc"
    touch "$stamp"
fi
echo "  $SYSTEMC_HOME"

# The layout Allo's csim expects: $MGC_HOME/shared/include holds the headers.
echo "== assembling $PREFIX/mgc"
inc="$PREFIX/mgc/shared/include"
rm -rf "$PREFIX/mgc"
mkdir -p "$inc" "$PREFIX/mgc/bin"
for d in ac_types ac_simutils matchlib_connections; do
    for f in "$PREFIX/src/$d/include/"*; do
        ln -sfn "$f" "$inc/$(basename "$f")"
    done
done
# csim only uses g++, but it still looks for a catapult binary first
# (hls.py, _find_catapult_binary). This stub fails loudly if anything runs it.
cat > "$PREFIX/mgc/bin/catapult" <<'EOF'
#!/bin/sh
echo "catapult stub from scripts/systemc-csim-setup.sh: this host has no Catapult" >&2
exit 1
EOF
chmod +x "$PREFIX/mgc/bin/catapult"

echo "== done. Export:"
print_env
