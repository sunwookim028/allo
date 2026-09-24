
## SystemC backend: single-shot kernel execution (EmitSystemC.cpp)
- Kernel run() body now executes ONCE then idles (`body; while(1) wait();`) instead
  of free-running `while(1){body}`. Fixes `both` (read-modify-write) memory
  accumulators that re-accumulated every pass.
- csim `__allo_done` counter (guarded out of __SYNTHESIS__): each kernel bumps it
  after its single pass; the memory-output testbench advances the clock until all
  kernel instances complete before reading memories, replacing a fixed cycle count.
- Net: +4 dataflow examples pass bit-exact (tiled_gemm, pingpong_gemm, hierachical,
  wrap_movement). Suite 28/28 unchanged. Synthesis-neutral (idle loop + guarded
  counter). NOTE: csynth already broken branch-wide by the ap_int subclass shim
  (2b1e66f) — unrelated.

## SystemC backend: fix csynth ac_int-subclass regression (EmitSystemC.cpp)
- The ap_int/ap_uint bit-slice shim (2b1e66f) was a `struct : ac_int<W>` subclass;
  Catapult rejected assignment to an ac_int-derived struct (ac_int.h:2259 CIN-15),
  breaking `go compile` on EVERY design.
- Fix: under `#ifdef __SYNTHESIS__` alias ap_(u)int to plain ac_int<W,S> (Catapult
  defines __SYNTHESIS__); keep the full-featured subclass (x(hi,lo) bit-range, >64
  narrowing) only for csim under `#else`.
- Verified: mem_port_reverse (AlloMem) synthesizes end-to-end to concat_sim_rtl.v;
  pure-stream kernels schedule cleanly; csim suite still 28/28. Synthesis loses the
  csim-only bit-range, so packed-stream designs fail locally at (hi,lo) until native
  ac_int .slc emission lands. AlloFifo (buffered stream) still unschedulable (separate).
