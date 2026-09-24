
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Catapult High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <ac_std_float.h>
#include <math.h>
#include <stdint.h>
using namespace std;
/// This is top function.
#pragma hls_design top
void mac16(
  int8_t v0[16],
  int8_t v1[16],
  int32_t *v2
) {	// L2
  int32_t acc;	// L5
  acc = 0;	// L6
  l_S_i_0_i: for (int i = 0; i < 16; i++) {	// L7
    int8_t v3 = v0[i];	// L8
    int8_t v4 = v1[i];	// L9
    int16_t v5 = v3;	// L10
    int16_t v6 = v4;	// L11
    int16_t v7 = v5 * v6;	// L12
    int32_t v8 = acc;	// L13
    ac_int<33, true> v9 = v8;	// L14
    ac_int<33, true> v10 = v7;	// L15
    ac_int<33, true> v11 = v9 + v10;	// L16
    int32_t v12 = v11;	// L17
    acc = v12;	// L18
  }
  *v2 = acc;	// L20
}

