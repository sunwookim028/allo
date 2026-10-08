// Workaround probe for vmem.mlir (E-A3): each D-12 rw port spelled as a read port of its
// latency plus a write port of latency 1 (c: r(3) + w(1), d: r(2) + w(1)). NOT the same
// hardware: a 2R2W memory where D-12 declares 2RW (each port reads OR writes per cycle).
amc.memory @vmem_split() -> (!amc.port<4096xi1024, static r(3)>, !amc.port<4096xi1024, static w(1)>,
                             !amc.port<4096xi1024, static r(2)>, !amc.port<4096xi1024, static w(1)>) {
  %0 = amc.alloc : !amc.ram<4096xi1024>
  %1 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static r(3)>
  %2 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static w(1)>
  %3 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static r(2)>
  %4 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static w(1)>
  amc.expose %1, %2, %3, %4 : !amc.port<4096xi1024, static r(3)>, !amc.port<4096xi1024, static w(1)>,
                              !amc.port<4096xi1024, static r(2)>, !amc.port<4096xi1024, static w(1)>
}
