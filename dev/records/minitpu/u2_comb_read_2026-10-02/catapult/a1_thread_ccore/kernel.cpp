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

#pragma hls_design ccore
#pragma hls_ccore_type combinational
data_t rd_mux(data_t m[32], addr_t a) { return m[a.to_int()]; }

#pragma hls_design top
SC_MODULE(rf_comb) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_in<addr_t> ra, rb, rc, wa;
  sc_in<data_t> wd;
  sc_in<bool> we;
  sc_out<data_t> qa, qb, qc;
  data_t mem[32];

  SC_CTOR(rf_comb) {
    SC_THREAD(wr);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void wr() {  // Allo's one-thread shape: reads, port writes, then wait(); the mux is a CCORE
    qa.write(0); qb.write(0); qc.write(0);
    for (int i = 0; i < 32; i++) mem[i] = 0;
    wait();
    #pragma hls_pipeline_init_interval 1
    while (1) {
      qa.write(rd_mux(mem, ra.read()));
      qb.write(rd_mux(mem, rb.read()));
      qc.write(rd_mux(mem, rc.read()));
      if (we.read()) mem[wa.read().to_int()] = wd.read();
      wait();
    }
  }
};
