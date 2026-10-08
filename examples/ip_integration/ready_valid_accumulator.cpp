// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hls_stream.h>
#include <stdint.h>

void ready_valid_accumulator(hls::stream<int32_t> &A, hls::stream<int32_t> &B,
                             hls::stream<int32_t> &C) {
  static uint32_t total = 0;
  for (int i = 0; i < 4; ++i) {
    total += static_cast<uint32_t>(A.read());
    total += static_cast<uint32_t>(B.read());
    C.write(static_cast<int32_t>(total));
  }
}
