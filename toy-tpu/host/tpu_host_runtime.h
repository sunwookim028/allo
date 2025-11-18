#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bram_defs.h"
#include "instructions.h"

extern "C" void top_module(
    const mini_tpu::Command commands[mini_tpu::MAX_COMMANDS], int num_commands,
    int32_t host_mem[mini_tpu::HOST_MEM_WORDS]);

namespace mini_tpu {

enum class MemcpyDirection { HostToTpu, TpuToHost };

class TpuSession {
 public:
  void reset() {
    host_mem_.fill(0);
    program_.clear();
    pending_reads_.clear();
  }

  void memcpy(std::uint32_t bram_addr, const int32_t* buffer,
              std::uint32_t words, MemcpyDirection dir) {
    if (dir != MemcpyDirection::HostToTpu) {
      throw std::invalid_argument(
          "Const buffer may only be used with HostToTpu direction");
    }
    const std::uint32_t host_addr = map_host_addr(bram_addr);
    ensure_range(host_addr, words);
    for (std::uint32_t i = 0; i < words; ++i) {
      host_mem_[host_addr + i] = buffer[i];
    }
    program_.push_back(MakeLoad(host_addr, bram_addr, words));
  }

  void memcpy(std::uint32_t bram_addr, int32_t* buffer, std::uint32_t words,
              MemcpyDirection dir) {
    const std::uint32_t host_addr = map_host_addr(bram_addr);
    ensure_range(host_addr, words);
    if (dir == MemcpyDirection::HostToTpu) {
      for (std::uint32_t i = 0; i < words; ++i) {
        host_mem_[host_addr + i] = buffer[i];
      }
      program_.push_back(MakeLoad(host_addr, bram_addr, words));
    } else {
      program_.push_back(MakeStore(bram_addr, host_addr, words));
      pending_reads_.push_back({host_addr, buffer, words});
    }
  }

  std::vector<Command>& program() { return program_; }

  void run() {
    if (program_.size() > MAX_COMMANDS) {
      throw std::runtime_error("Command list exceeds MAX_COMMANDS");
    }
    std::array<Command, MAX_COMMANDS> command_buffer{};
    std::copy(program_.begin(), program_.end(), command_buffer.begin());
    top_module(command_buffer.data(),
               static_cast<int>(program_.size()), host_mem_.data());
    for (const auto& read : pending_reads_) {
      ensure_range(read.host_addr, read.words);
      for (std::uint32_t i = 0; i < read.words; ++i) {
        read.buffer[i] = host_mem_[read.host_addr + i];
      }
    }
    program_.clear();
    pending_reads_.clear();
  }

 private:
  struct PendingRead {
    std::uint32_t host_addr;
    int32_t* buffer;
    std::uint32_t words;
  };

  static constexpr std::uint32_t map_host_addr(std::uint32_t bram_addr) {
    return bram_addr;
  }

  void ensure_range(std::uint32_t host_addr, std::uint32_t words) const {
    if (host_addr + words > HOST_MEM_WORDS) {
      throw std::out_of_range("Host memory access exceeds bounds");
    }
  }

  std::array<int32_t, HOST_MEM_WORDS> host_mem_{};
  std::vector<Command> program_;
  std::vector<PendingRead> pending_reads_;
};

}  // namespace mini_tpu


