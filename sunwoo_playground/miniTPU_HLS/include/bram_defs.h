#pragma once

#include <cstdint>

namespace mini_tpu {

// BRAM configuration.
constexpr std::size_t BRAM_WORDS = 256;
constexpr std::size_t BRAM_WORD_BYTES = 4;

// Shared BRAM address map (word offsets).
constexpr std::uint32_t BRAM_ADDR_A = 0;
constexpr std::uint32_t BRAM_ADDR_B = 16;
constexpr std::uint32_t BRAM_ADDR_S = 64;
constexpr std::uint32_t BRAM_ADDR_C = 80;
constexpr std::uint32_t BRAM_ADDR_D = 96;
constexpr std::uint32_t BRAM_ADDR_E = 112;

constexpr std::uint32_t MATRIX_WORDS = 4;   // 2x2 matrix
constexpr std::uint32_t SCALAR_WORDS = 1;   // single 32-bit integer

// Host-side memory emulation size (supports multiple regions).
constexpr std::size_t HOST_MEM_WORDS = 256;
constexpr std::size_t MAX_COMMANDS = 64;

constexpr std::uint32_t HOST_ADDR_A = 0;
constexpr std::uint32_t HOST_ADDR_B = 32;
constexpr std::uint32_t HOST_ADDR_S = 64;
constexpr std::uint32_t HOST_ADDR_E = 96;

// Command opcodes.
enum Opcode : std::uint32_t {
  OP_NOP = 0,
  OP_LOAD = 1,
  OP_STORE = 2,
  OP_MATMUL = 3,
  OP_VECADD = 4,
  OP_SCALE = 5,
};

// Generic command encoding interpreted according to opcode.
struct Command {
  std::uint32_t opcode;
  std::uint32_t addr0;
  std::uint32_t addr1;
  std::uint32_t addr2;
  std::uint32_t length;
  std::int32_t scalar;
};

}  // namespace mini_tpu


