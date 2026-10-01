// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hls_stream.h>
#include <stdint.h>
void pulse_core(hls::stream<int32_t> &A, hls::stream<int32_t> &C) {
  C.write(A.read() + 1);
}
