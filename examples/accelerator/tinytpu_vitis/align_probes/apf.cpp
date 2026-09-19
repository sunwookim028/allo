// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Vitis 2023.2 ap_float: BF16 = ap_float<16,8>, MiniTPU acc24 = ap_float<24,8>; a subnormal
// operand is flushed to zero. Build: see docs/source/designs/alignment.rst (datatype section).
#include <ap_float.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
typedef ap_float<16,8> bf16; typedef ap_float<24,8> acc24; typedef ap_float<32,8> f32;
static float tof(const f32 &x){ float f; memcpy(&f, &x, 4); return f; }
static f32 fromf(float f){ f32 x; memcpy(&x, &f, 4); return x; }
int main(){
  // smallest normal f32 = 2^-126; half of it is subnormal
  f32 a = fromf(1.1754944e-38f), b = fromf(-1.0e-38f);   // a - |b| is subnormal
  f32 s = a; s += b;
  printf("f32 normal+normal->subnormal: %g (exact %g)\n", tof(s), 1.1754944e-38f - 1.0e-38f);
  f32 c = fromf(1.0f), d = fromf(3.0f); f32 e = c; e /= d;
  bf16 h(e); acc24 g(e); f32 back_h(h), back_g(g);
  printf("1/3 -> bf16 -> f32: %.9g ; -> acc24 -> f32: %.9g\n", tof(back_h), tof(back_g));
  return 0;
}
