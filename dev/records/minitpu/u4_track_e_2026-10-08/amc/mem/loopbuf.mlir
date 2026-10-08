// loop_ctrl l1_d12's loop buffer (units/loop_ctrl_d12.py::LB): Memory("lb", UInt(128), rows=CAP=24,
//   ports=(Port("cap", "w", visible=1), Port("replay", "r", latency=0)), collision="refuse", reset=False)
// The replay port is ASYNCHRONOUS (latency 0): AMC's port type admits r(0) (d12_memory_ports).
amc.memory @loopbuf() -> (!amc.port<24xi128, static w(1)>, !amc.port<24xi128, static r(0)>) {
  %0 = amc.alloc : !amc.ram<24xi128>
  %1 = amc.create_port(%0 : !amc.ram<24xi128>) : !amc.port<24xi128, static w(1)>
  %2 = amc.create_port(%0 : !amc.ram<24xi128>) : !amc.port<24xi128, static r(0)>
  amc.expose %1, %2 : !amc.port<24xi128, static w(1)>, !amc.port<24xi128, static r(0)>
}
