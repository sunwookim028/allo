#include <ac_int.h>
#include <stdint.h>
#include <cstdio>
int main() {
  uint64_t v15 = 0x123456789ABCDEF0ULL;
  // fix for defect 1: ac_int + slc<W>(lo) instead of ap_int + (hi, lo)
  ac_int<64, false> t = v15;
  uint16_t v16 = t.slc<16>(0).to_uint();
  ac_int<6, false> v48 = t.slc<6>(0);
  // fix for defect 2: explicit to_int() on a >64-bit ac_int
  ac_int<65, true> v34 = 7; ac_int<65, true> v35 = v34 + 8; int v36 = v35.to_int();
  printf("%x %d %d\n", (unsigned)v16, v48.to_int(), v36);
#ifdef SHOW_DEFECT2
  int bad = v35;
#endif
  return !(v16 == 0xDEF0 && v48.to_int() == 0x30 && v36 == 15);
}
