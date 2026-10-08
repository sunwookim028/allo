// Constructs the standalone minitpu_core model, resets it, clocks 100 cycles; checks it elaborates and runs from cwd.
#include "Vminitpu_core.h"
#include "verilated.h"
#include <cstdio>
int main(int argc, char** argv) {
  VerilatedContext ctx; ctx.commandArgs(argc, argv);
  Vminitpu_core top(&ctx);
  top.clk = 0; top.rst_n = 0; top.start = 0; top.dm_req_ready = 1; top.dm_rsp_valid = 0;
  for (int c = 0; c < 200; ++c) {
    if (c == 8) top.rst_n = 1;
    top.clk = 0; top.eval(); ctx.timeInc(1);
    top.clk = 1; top.eval(); ctx.timeInc(1);
  }
  std::printf("SMOKE done=%d perf_cycles=%u illegal=%d\n", (int)top.done, (unsigned)top.perf_cnt_cycles_o, (int)top.illegal_op_o);
  top.final();
  return 0;
}
