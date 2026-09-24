// SCVerify testbench for the Allo-emitted `mac16` kernel (kernel.cpp).
//
// `go switching` runs THIS against the pre-power RTL; its activity is the whole
// basis of the power number, so the stimulus is the measurement's workload. It is
// the same stimulus as the hand-run 16-tap MAC in ../zhang21_power_2026-09-24/
// (200 transactions, pseudo-random int8 operands over the full [-128, 127] range),
// so the two power figures are comparable.
//
// The design is `void mac16(int8_t a[16], int8_t b[16], int32_t *out)`, marked
// `#pragma hls_design top` by the Allo emitter. CCS_DESIGN() resolves to the DUT in
// csim and to the RTL wrapper under SCVerify.
#include <ac_int.h>
#include <mc_scverify.h>
#include <cstdint>
#include <cstdio>

void mac16(int8_t a[16], int8_t b[16], int32_t *out);

CCS_MAIN(int argc, char **argv) {
  int8_t a[16], b[16];
  int32_t o = 0;
  int errs = 0;
  for (int t = 0; t < 200; ++t) {
    int32_t ref = 0;
    for (int i = 0; i < 16; ++i) {
      a[i] = (int8_t)((t * 7 + i * 13) % 256 - 128);
      b[i] = (int8_t)((t * 11 + i * 5) % 256 - 128);
      ref += (int32_t)a[i] * (int32_t)b[i];
    }
    CCS_DESIGN(mac16)(a, b, &o);
    if (o != ref) ++errs;
  }
  printf("MAC16 TB errors=%d\n", errs);
  CCS_RETURN(errs != 0);
}
