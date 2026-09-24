#include <ac_int.h>
#include <mc_scverify.h>
#include <cstdio>
void mac(ac_int<8,true> a[16], ac_int<8,true> b[16], ac_int<24,true> &out);
CCS_MAIN(int argc, char **argv) {
  ac_int<8,true> a[16], b[16]; ac_int<24,true> o; int errs = 0;
  for (int t = 0; t < 200; ++t) {
    int ref = 0;
    for (int i = 0; i < 16; ++i) { a[i] = (t * 7 + i * 13) % 256 - 128; b[i] = (t * 11 + i * 5) % 256 - 128; ref += a[i].to_int() * b[i].to_int(); }
    CCS_DESIGN(mac)(a, b, o);
    if (o.to_int() != ref) ++errs;
  }
  printf("MAC TB errors=%d\n", errs);
  CCS_RETURN(errs != 0);
}
