// Can a Vitis dataflow region have an on-chip array with ONE writer and TWO
// readers? MODE 0: plain.  MODE 1: `#pragma HLS stable variable=buf`.
// MODE 2: two readers of the same m_axi pointer (no local array).
#include <hls_stream.h>
#include <ap_int.h>
#ifndef MODE
#define MODE 0
#endif
typedef ap_uint<32> w_t;
static void wr(hls::stream<w_t>& in, w_t buf[64]) {
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II=1
    buf[i] = in.read();
  }
}
static void rd(w_t buf[64], hls::stream<w_t>& out, int off) {
  for (int i = 0; i < 32; i++) {
#pragma HLS pipeline II=1
    out.write(buf[off + i]);
  }
}
static void wrrd(hls::stream<w_t>& in, w_t buf[64], hls::stream<w_t>& out) {
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II=1
    buf[i] = in.read();
  }
  for (int i = 0; i < 32; i++) {
#pragma HLS pipeline II=1
    out.write(buf[i]);
  }
}
static void rdm(const w_t* p, hls::stream<w_t>& out, int off) {
  for (int i = 0; i < 32; i++) {
#pragma HLS pipeline II=1
    out.write(p[off + i]);
  }
}
static void src(const w_t* p, hls::stream<w_t>& o) {
  for (int i = 0; i < 64; i++) {
#pragma HLS pipeline II=1
    o.write(p[i]);
  }
}
static void snk(hls::stream<w_t>& a, hls::stream<w_t>& b, w_t* q) {
  for (int i = 0; i < 32; i++) {
#pragma HLS pipeline II=1
    q[i] = a.read() + b.read();
  }
}
extern "C" void top(const w_t* p, w_t* q) {
#pragma HLS interface m_axi port=p offset=slave bundle=gmem0 depth=64
#pragma HLS interface m_axi port=q offset=slave bundle=gmem1 depth=32
#pragma HLS dataflow
  hls::stream<w_t> s, a, b;
#if MODE == 2
  rdm(p, a, 0);
  rdm(p, b, 32);
#else
  w_t buf[64];
#if MODE == 1
#pragma HLS stable variable=buf
#endif
#if MODE == 3
#pragma HLS stream variable=buf type=shared
#endif
#if MODE == 4 || MODE == 5
#pragma HLS stream variable=buf type=unsync
#endif
  src(p, s);
#if MODE >= 5
  wrrd(s, buf, a);      // one process writes AND reads buf (port A)
  rd(buf, b, 32);       // a second process reads it       (port B)
#else
  wr(s, buf);
  rd(buf, a, 0);
  rd(buf, b, 32);
#endif
#endif
  snk(a, b, q);
}
