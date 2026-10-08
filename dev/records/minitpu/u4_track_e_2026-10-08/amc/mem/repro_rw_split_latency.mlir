// AMC finding E-A3 (tool bug): a static rw port whose read latency differs from its
// write latency -- legal at the type level (test/errors/dyn-rw-latency.mlir: "Static
// `rw` with split latencies is allowed") -- cannot be lowered:
//   amctool --memory-only --emit-verilog repro_rw_split_latency.mlir
//   error: failed to legalize operation 'seq.hlmem' that was explicitly marked illegal
// AmcToHW.cpp unpackPortType() folds the port to max(read, write) latency and
// lowerCreatePort passes that number to seq::WritePortOp (AmcToHW.cpp:359), which
// LowerSeqHLMem refuses ("only supports write ports with latency == 1", a match
// failure that never reaches the user). rw(1, 1) and r(2) + w(1) lower.
amc.memory @m() -> !amc.port<64xi32, static rw(2, 1)> {
  %0 = amc.alloc : !amc.ram<64xi32>
  %1 = amc.create_port(%0 : !amc.ram<64xi32>) : !amc.port<64xi32, static rw(2, 1)>
  amc.expose %1 : !amc.port<64xi32, static rw(2, 1)>
}
