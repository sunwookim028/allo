#include "bram_defs.h"
#include "instructions.h"

static int32_t shared_bram[mini_tpu::BRAM_WORDS];

static void KernelLoad(const mini_tpu::Command& cmd, int32_t* host_mem) {
  const std::uint32_t host_addr = cmd.addr0;
  const std::uint32_t bram_addr = cmd.addr1;
  for (std::uint32_t i = 0; i < cmd.length; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
    shared_bram[bram_addr + i] = host_mem[host_addr + i];
  }
}

static void KernelStore(const mini_tpu::Command& cmd, int32_t* host_mem) {
  const std::uint32_t bram_addr = cmd.addr0;
  const std::uint32_t host_addr = cmd.addr1;
  for (std::uint32_t i = 0; i < cmd.length; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
    host_mem[host_addr + i] = shared_bram[bram_addr + i];
  }
}

static void KernelMatMul(const mini_tpu::Command& cmd) {
  const std::uint32_t a = cmd.addr0;
  const std::uint32_t b = cmd.addr1;
  const std::uint32_t c = cmd.addr2;

  const int32_t a00 = shared_bram[a + 0];
  const int32_t a01 = shared_bram[a + 1];
  const int32_t a10 = shared_bram[a + 2];
  const int32_t a11 = shared_bram[a + 3];

  const int32_t b00 = shared_bram[b + 0];
  const int32_t b01 = shared_bram[b + 1];
  const int32_t b10 = shared_bram[b + 2];
  const int32_t b11 = shared_bram[b + 3];

  shared_bram[c + 0] = a00 * b00 + a01 * b10;
  shared_bram[c + 1] = a00 * b01 + a01 * b11;
  shared_bram[c + 2] = a10 * b00 + a11 * b10;
  shared_bram[c + 3] = a10 * b01 + a11 * b11;
}

static void KernelVecAdd(const mini_tpu::Command& cmd) {
  const std::uint32_t a = cmd.addr0;
  const std::uint32_t b = cmd.addr1;
  const std::uint32_t out = cmd.addr2;
  for (std::uint32_t i = 0; i < cmd.length; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
    shared_bram[out + i] = shared_bram[a + i] + shared_bram[b + i];
  }
}

static void KernelScale(const mini_tpu::Command& cmd) {
  const std::uint32_t in = cmd.addr0;
  const std::uint32_t out = cmd.addr1;
  const int32_t scalar = cmd.scalar;
  for (std::uint32_t i = 0; i < cmd.length; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II=1
#endif
    shared_bram[out + i] = shared_bram[in + i] * scalar;
  }
}

extern "C" void top_module(const mini_tpu::Command commands[mini_tpu::MAX_COMMANDS],
                           int num_commands,
                           int32_t host_mem[mini_tpu::HOST_MEM_WORDS]) {
#ifdef __SYNTHESIS__
#pragma HLS RESOURCE variable=shared_bram core=RAM_1P_BRAM
#pragma HLS INTERFACE m_axi port=commands offset=slave bundle=gmem0 depth=mini_tpu::MAX_COMMANDS
#pragma HLS INTERFACE m_axi port=host_mem offset=slave bundle=gmem1 depth=mini_tpu::HOST_MEM_WORDS
#pragma HLS INTERFACE s_axilite port=num_commands bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE bram port=shared_bram
#endif

  const int clamped_commands =
      (num_commands < static_cast<int>(mini_tpu::MAX_COMMANDS))
          ? num_commands
          : static_cast<int>(mini_tpu::MAX_COMMANDS);

  for (int idx = 0; idx < clamped_commands; ++idx) {
#ifdef __SYNTHESIS__
#pragma HLS LOOP_TRIPCOUNT min=1 max=mini_tpu::MAX_COMMANDS
#endif
    const mini_tpu::Command cmd = commands[idx];

    switch (cmd.opcode) {
      case mini_tpu::OP_LOAD:
        KernelLoad(cmd, host_mem);
        break;
      case mini_tpu::OP_STORE:
        KernelStore(cmd, host_mem);
        break;
      case mini_tpu::OP_MATMUL:
        KernelMatMul(cmd);
        break;
      case mini_tpu::OP_VECADD:
        KernelVecAdd(cmd);
        break;
      case mini_tpu::OP_SCALE:
        KernelScale(cmd);
        break;
      default:
        break;
    }
  }
}


