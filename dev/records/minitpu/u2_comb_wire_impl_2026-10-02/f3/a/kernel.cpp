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
  sc_signal<data_t> mem[32];

  SC_CTOR(rf_comb) {
    SC_THREAD(wr);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_METHOD(rd);
    sensitive << ra << rb << rc;
    for (int i = 0; i < 32; i++) sensitive << mem[i];
  }
  void rd() {  // combinational: this cycle's array contents
    qa.write(mem[ra.read().to_int()].read());
    qb.write(mem[rb.read().to_int()].read());
    qc.write(mem[rc.read().to_int()].read());
  }
  void wr() {  // one sync write. vpu_regfile.sv never resets mem; Catapult
    // refuses an sc_signal not set in the reset action (CIN-233), so it is zeroed.
    // F3: storage NOT written in the reset action
    wait();
    while (1) {
      if (we.read()) mem[wa.read().to_int()].write(wd.read());
      wait();
    }
  }
};
