// Mixed-reset proof (D-14 follow-up a): write all 32 words of the unreset
// storage `mem` and the reset storage `tag`, pulse reset mid-run, read back.
// Expect: mem keeps its contents across reset; tag reads 0; cnt restarts.
#include "Vrf_0.h"
#include "verilated.h"
#include <cstdio>
static Vrf_0 *m;
static long cyc = 0;
static void tick() { m->clk = 0; m->eval(); m->clk = 1; m->eval(); cyc++; }
static unsigned W(int i) { return (0x8000u + 0x111u * i) & 0xffff; }
int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  m = new Vrf_0;
  m->clk = 0; m->rst = 0; m->v12 = m->v13 = m->v14 = m->v15 = 0; m->eval();
  for (int i = 0; i < 4; i++) tick();
  m->rst = 1;
  for (int i = 0; i < 4; i++) tick();
  // phase 1: write mem[i] = tag[i&3] = W(i)
  for (int i = 0; i < 32; i++) { m->v13 = i; m->v14 = W(i); m->v15 = 1; m->v12 = 0; m->eval(); tick(); }
  m->v15 = 0; m->eval(); tick(); tick();
  int bad = 0, pre_tag_nz = 0; unsigned cnt_pre = 0;
  for (int i = 0; i < 32; i++) {
    m->v12 = i; m->eval();
    unsigned q = m->v16; if (q != W(i)) { bad++; printf("pre  mem[%d]=%04x want %04x\n", i, q, W(i)); }
    pre_tag_nz += (m->v17 != 0); cnt_pre = m->v18;
    tick();
  }
  printf("PRE-RESET  mem %d/32 ok, tag nonzero %d/32, cnt=%u\n", 32 - bad, pre_tag_nz, cnt_pre);
  // reset pulse mid-run (active low), inputs idle
  m->v15 = 0; m->rst = 0; m->eval();
  for (int i = 0; i < 3; i++) tick();
  m->rst = 1; m->eval();
  int bad2 = 0, tag_nz = 0; unsigned cmax = 0, cfirst = 0xffffffff;
  for (int i = 0; i < 32; i++) {
    m->v12 = i; m->eval();
    unsigned q = m->v16; if (q != W(i)) { bad2++; if (bad2 < 4) printf("post mem[%d]=%04x want %04x\n", i, q, W(i)); }
    if (i >= 2) tag_nz += (m->v17 != 0);
    if (i == 0) cfirst = m->v18;
    if (m->v18 > cmax) cmax = m->v18;
    tick();
  }
  printf("POST-RESET mem %d/32 kept, tag nonzero %d/30 (want 0), cnt first=%u max=%u (pre-reset %u)\n",
         32 - bad2, tag_nz, cfirst, cmax, cnt_pre);
  bool pass = bad == 0 && bad2 == 0 && tag_nz == 0 && pre_tag_nz > 0 && cmax <= 33 && cnt_pre > 60;
  printf("%s\n", pass ? "MIXED-RESET PASS" : "MIXED-RESET FAIL");
  delete m;
  return pass ? 0 : 1;
}
