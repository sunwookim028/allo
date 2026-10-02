/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <hls_stream.h>
#include <stdint.h>

#define N 32

void vadd(hls::stream<bool> &enable_s, hls::stream<int32_t> &a_s,
          hls::stream<int32_t> &b_s, hls::stream<int32_t> &out_s) {
  int32_t a = 0;
  int32_t b = 0;
  int32_t out = 0;
  bool enable = enable_s.read();
  if (enable == 1) {
    for (int i = 0; i < N; i++) {
      a = a_s.read();
      b = b_s.read();
      out_s.write(a + b);
    }
  }
}