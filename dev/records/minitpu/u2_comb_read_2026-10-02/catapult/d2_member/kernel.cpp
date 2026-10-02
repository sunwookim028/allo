// Form (d): hand-written SystemC regfile, the shape Allo's emitter would need.
// One clocked SC_THREAD owns the write; one SC_METHOD (no clock) drives the three
// read ports combinationally from the storage, which is an array of sc_signal so
// that two processes may share it (Catapult HIER-41 forbids a plain member).
#include <systemc.h>
#include <ac_int.h>
#include <mc_connections.h>
#ifndef W
#define W 16
#endif
typedef ac_int<5, false> addr_t;
typedef ac_int<W, false> data_t;

#pragma hls_design top
SC_MODULE(rf_comb) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_in<addr_t> ra, rb, rc, wa;
  sc_in<data_t> wd;
  sc_in<bool> we;
  sc_out<data_t> qa, qb, qc;
  data_t mem[32];  // plain member, as Allo emits @ Stateful

  SC_CTOR(rf_comb) {
    SC_THREAD(wr);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_METHOD(rd);
    sensitive << ra << rb << rc;
  }
  void rd() {  // combinational: this cycle's array contents
    qa.write(mem[ra.read().to_int()]);
    qb.write(mem[rb.read().to_int()]);
    qc.write(mem[rc.read().to_int()]);
  }
  void wr() {  // one sync write. vpu_regfile.sv never resets mem; Catapult
    // refuses an sc_signal not set in the reset action (CIN-233), so it is zeroed.
    for (int i = 0; i < 32; i++) mem[i] = 0;
    wait();
    while (1) {
      if (we.read()) mem[wa.read().to_int()] = wd.read();
      wait();
    }
  }
};
