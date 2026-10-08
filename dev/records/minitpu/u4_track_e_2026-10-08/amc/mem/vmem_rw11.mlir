// Second workaround probe: both D-12 rw ports at rw(1, 1) (the only rw latency AMC lowers);
// the extra read stages (2 for c, 1 for d) would have to be registers outside the memory.
amc.memory @vmem_rw11() -> (!amc.port<4096xi1024, static rw(1, 1)>, !amc.port<4096xi1024, static rw(1, 1)>) {
  %0 = amc.alloc : !amc.ram<4096xi1024>
  %1 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static rw(1, 1)>
  %2 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static rw(1, 1)>
  amc.expose %1, %2 : !amc.port<4096xi1024, static rw(1, 1)>, !amc.port<4096xi1024, static rw(1, 1)>
}
