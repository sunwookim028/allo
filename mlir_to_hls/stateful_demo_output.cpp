
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
#include <math.h>
#include <stdint.h>
using namespace std;
static int32_t state_stateful_123456789 = {0};	// L2
void accumulator(
  int32_t v0,
  int32_t *v1
) {	// L3
  // placeholder for const int32_t state_stateful_123456789	// L4
  int32_t v3 = state_stateful_123456789;	// L5
  int32_t v4 = v3 + v0;	// L6
  state_stateful_123456789 = v4;	// L7
  *v1 = state_stateful_123456789;	// L8
}

