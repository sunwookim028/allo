#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# usage: u4d_bookings.sh: latency.check_booking on every pinned build's manifest, at its clock:
# the I/O pin (1) per kernel, and scalar_agu's D-20 booking S_LAT (Calendar.S_LAT, ControlGeometry).
S=${U4D_SCRATCH:-/work/shared/users/phd/sk3463/scratch/u4d}
export PYTHONPATH=$PWD
bk() { echo "== $1 (clock $2): $3"; $ALLO_PYTHON -m examples.minitpu.harness.latency --clock $2 --bookings "$3" $S/$1.prj 2>&1 | grep BOOKING; }
for c in 3p33:3.33 2p0:2.0; do k=${c%%:*}; p=${c##*:}
  bk issue_${k}_L1 $p '{"issue_0": 1}'
  bk wb_locked_${k}_L1 $p '{"wbk_0": 1}'
  bk fp_fq_${k}_L1 $p '{"fq_0": 1}'
  bk dec_${k}_L1 $p '{"decoder_0": 1}'
  bk agu_${k}_L1 $p '{"agu_0": 1}'
  bk vad_${k}_L1 $p '{"adapter_0": 1}'
  bk dag_bits_${k}_L1 $p '{"agu_0": 1}'
  bk loop_${k}_L1 $p '{"loop_0": 1}'
  bk desc_${k}_L1 $p '{"adapter_0": 1}'
  bk sagu2_${k}_L1 $p '{"agu_0": 1}'
done
bk dag_c1_3p33_L1 3.33 '{"gen_0": 1}'
for l in 1 2 3; do bk sagu${l}_3p33_L1 3.33 "{\"agu_0\": $l}"; done
