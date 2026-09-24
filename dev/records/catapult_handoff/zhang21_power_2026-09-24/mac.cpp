#include <ac_int.h>
#include <mc_scverify.h>
#pragma hls_design top
void CCS_BLOCK(mac)(ac_int<8,true> a[16], ac_int<8,true> b[16], ac_int<24,true> &out) {
  ac_int<24,true> acc = 0;
  for (int i = 0; i < 16; ++i) acc += a[i] * b[i];
  out = acc;
}
