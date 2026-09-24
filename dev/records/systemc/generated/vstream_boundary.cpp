
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;
void source_0(
  int32_t v0[8],
  hls::stream< int32_t >& v1
) {	// L2
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L3
    int32_t v3 = v0[i];	// L4
    v1.write(v3);	// L5
  }
}

void compute_0(
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5
) {	// L9
  l_S_i_0_i1: for (int i1 = 0; i1 < 8; i1++) {	// L11
    int32_t v7 = v4.read();	// L12
    ap_int<33> v8 = v7;	// L13
    ap_int<33> v9 = v8 + 1;	// L14
    v5.write(v9);	// L15
  }
}

void sink_0(
  int32_t v10[8],
  hls::stream< int32_t >& v11
) {	// L19
  l_S_i_0_i2: for (int i2 = 0; i2 < 8; i2++) {	// L20
    int32_t v13 = v11.read();	// L21
    v10[i2] = v13;	// L22
  }
}

/// This is top function.
void top(
  int32_t v14[8],
  int32_t v15[8]
) {	// L26
  #pragma HLS dataflow
  hls::stream< int32_t > v16;
  #pragma HLS stream variable=v16 depth=4	// L27
  hls::stream< int32_t > v17;
  #pragma HLS stream variable=v17 depth=4	// L28
  source_0(v14, v16);	// L29
  compute_0(v16, v17);	// L30
  sink_0(v15, v17);	// L31
}

