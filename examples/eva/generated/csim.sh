#!/bin/bash
# Standalone SystemC behavioral csim for the self-contained
# kernel.cpp (compile+run its sc_main tb with Catapult's g++).
set -e
: "${MGC_HOME:?set MGC_HOME to your Catapult Mgc_home}"
GXX="$MGC_HOME/bin/g++"
INC="$MGC_HOME/shared/include"
LIB=$(ls -d "$MGC_HOME"/shared/lib/Linux/gcc-*-64 2>/dev/null | head -1)
"$GXX" -std=c++11 -DSC_INCLUDE_DYNAMIC_PROCESSES -I"$INC" \
  kernel.cpp -o csim_sim -L"$LIB" -Wl,-rpath,"$LIB" -lsystemc
LD_LIBRARY_PATH="$MGC_HOME/lib:$LIB" ./csim_sim
