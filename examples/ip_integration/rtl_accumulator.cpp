// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hls_stream.h>
#include <stdint.h>
void accumulator(hls::stream<int32_t> &A, hls::stream<int32_t> &C) {
  static uint32_t total = 0;
  total += static_cast<uint32_t>(A.read());
  C.write(static_cast<int32_t>(total));
}
