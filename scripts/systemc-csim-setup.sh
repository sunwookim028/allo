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
# produces RTL, and these library versions are not matched to Catapult's.
# Catapult on a licence host is the authority.
#
# Usage:
#   scripts/systemc-csim-setup.sh            # fetch and assemble; print the env to export
#   eval "$(scripts/systemc-csim-setup.sh --env)"

set -euo pipefail

PREFIX="${ALLO_CSIM_HOME:-$HOME/.cache/allo/systemc-csim}"

# hlslibs at the commits validated on 2026-10-01 (TinyTPU and EVA csim).
AC_TYPES_SHA=e9ed172a464e0a9b45a23c712ab526782c668952
AC_SIMUTILS_SHA=f1a3cc6d5e23830611def4ab238dcd602a507262
CONNECTIONS_SHA=fd79d73d0abad49cd813eee0aa05460ff0439083

# Accellera SystemC 2.3.1, as shipped inside Vitis 2023.2, pinned by checksum.
SYSTEMC_HOME_DEFAULT=/opt/xilinx/Vitis/2023.2/lnx64/tools/systemc
LIBSYSTEMC_SHA256=f868dbe5173e649f1192edcc50df16f49eb5d269b76f54e45d95cde2a49a095b
SYSTEMC_HOME="${SYSTEMC_HOME:-$SYSTEMC_HOME_DEFAULT}"

print_env() {
    echo "export MGC_HOME=$PREFIX/mgc"
    echo "export SYSTEMC_HOME=$SYSTEMC_HOME"
    echo "export ALLO_CXX_EXTRA=\"-DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM\""
}

if [ "${1:-}" = "--env" ]; then
    print_env
    exit 0
fi

fetch() {  # fetch <repo> <sha> <dir>
    local repo=$1 sha=$2 dir=$3
    if [ ! -d "$dir/.git" ]; then
        git clone -q "https://github.com/hlslibs/$repo.git" "$dir"
    fi
    git -C "$dir" fetch -q origin "$sha" 2>/dev/null || git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q --detach "$sha"
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || { echo "error: $repo is not at $sha" >&2; exit 1; }
    echo "  $repo @ ${sha:0:7}"
}

mkdir -p "$PREFIX/src"
echo "== hlslibs (pinned)"
fetch ac_types "$AC_TYPES_SHA" "$PREFIX/src/ac_types"
fetch ac_simutils "$AC_SIMUTILS_SHA" "$PREFIX/src/ac_simutils"
fetch matchlib_connections "$CONNECTIONS_SHA" "$PREFIX/src/matchlib_connections"

echo "== SystemC"
lib="$SYSTEMC_HOME/lib/libsystemc.a"
[ -f "$lib" ] || lib="$SYSTEMC_HOME/lib-linux64/libsystemc.a"
if [ ! -f "$lib" ]; then
    echo "error: no libsystemc.a under SYSTEMC_HOME=$SYSTEMC_HOME" >&2
    exit 1
fi
got=$(sha256sum "$lib" | cut -d' ' -f1)
if [ "$got" = "$LIBSYSTEMC_SHA256" ]; then
    echo "  $lib (sha256 matches the pin)"
else
    echo "  WARNING: $lib has sha256 $got, not the pinned $LIBSYSTEMC_SHA256" >&2
fi

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
