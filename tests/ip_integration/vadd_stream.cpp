/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A minimal HLS IP that communicates through hls::stream<T> FIFO ports.
// It reads one element from each input stream, adds them, and writes the
// result to the output stream, N times. Streams A and B are inputs (the IP
// reads them); stream C is an output (the IP writes it).
#include <hls_stream.h>
#include <stdint.h>

#define VADD_STREAM_N 32

extern "C" {

void vadd_stream(hls::stream<int32_t> &A, hls::stream<int32_t> &B,
                 hls::stream<int32_t> &C) {
  for (int i = 0; i < VADD_STREAM_N; ++i) {
#pragma HLS pipeline II = 1
    int32_t a = A.read();
    int32_t b = B.read();
    C.write(a + b);
  }
}

} // extern "C"
