// fetch f1_d12's IRAM (units/fetch_d12.py::IRAM): Memory("iram", UInt(128), rows=4096,
//   ports=(Port("host", "w", visible=1), Port("fetch", "r", latency=1)), collision="refuse", reset=False)
// as AMC's own memory declaration: one bank, one static write port (write latency 1),
// one static read port of read latency 1.
amc.memory @iram() -> (!amc.port<4096xi128, static w(1)>, !amc.port<4096xi128, static r(1)>) {
  %0 = amc.alloc : !amc.ram<4096xi128>
  %1 = amc.create_port(%0 : !amc.ram<4096xi128>) : !amc.port<4096xi128, static w(1)>
  %2 = amc.create_port(%0 : !amc.ram<4096xi128>) : !amc.port<4096xi128, static r(1)>
  amc.expose %1, %2 : !amc.port<4096xi128, static w(1)>, !amc.port<4096xi128, static r(1)>
}
