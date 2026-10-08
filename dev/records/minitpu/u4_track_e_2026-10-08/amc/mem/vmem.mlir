// dma_vmem's VMEM (units/dma_unit.py::vmem_architecture, core): Memory("vmem", UInt(1024), rows=4096,
//   ports=(Port("c", "rw", latency=3, visible=1), Port("d", "rw", latency=2, visible=1)),
//   collision="obligation", reset=False)  -- H11: the DMA port beside the compute port.
amc.memory @vmem() -> (!amc.port<4096xi1024, static rw(3, 1)>, !amc.port<4096xi1024, static rw(2, 1)>) {
  %0 = amc.alloc : !amc.ram<4096xi1024>
  %1 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static rw(3, 1)>
  %2 = amc.create_port(%0 : !amc.ram<4096xi1024>) : !amc.port<4096xi1024, static rw(2, 1)>
  amc.expose %1, %2 : !amc.port<4096xi1024, static rw(3, 1)>, !amc.port<4096xi1024, static rw(2, 1)>
}
