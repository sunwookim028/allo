
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
int32_t acc_stateful_5700927378943735518 = {0};	// L2
void test(
  int32_t v0,
  int32_t *v1
) {	// L3
  // placeholder for const int32_t acc_stateful_5700927378943735518	// L4
  int32_t v3 = acc_stateful_5700927378943735518;	// L5
  int32_t v4 = v3 + v0;	// L6
  acc_stateful_5700927378943735518 = v4;	// L7
  *v1 = acc_stateful_5700927378943735518;	// L8
}

