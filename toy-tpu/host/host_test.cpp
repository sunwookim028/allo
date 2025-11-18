#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bram_defs.h"
#include "instructions.h"
#include "tpu_host_runtime.h"

namespace {

using Matrix2x2 = std::array<int32_t, mini_tpu::MATRIX_WORDS>;

Matrix2x2 MatMul(const Matrix2x2& a, const Matrix2x2& b) {
  Matrix2x2 c{};
  c[0] = a[0] * b[0] + a[1] * b[2];
  c[1] = a[0] * b[1] + a[1] * b[3];
  c[2] = a[2] * b[0] + a[3] * b[2];
  c[3] = a[2] * b[1] + a[3] * b[3];
  return c;
}

Matrix2x2 VecAdd(const Matrix2x2& a, const Matrix2x2& b) {
  Matrix2x2 c{};
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

Matrix2x2 Scale(const Matrix2x2& in, int32_t scalar) {
  Matrix2x2 out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = in[i] * scalar;
  }
  return out;
}

struct Options {
  int trials = 10;
  std::uint32_t seed = 0x12345678;
};

Options ParseOptions(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--trials" && i + 1 < argc) {
      opts.trials = std::atoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      opts.seed = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    }
  }
  return opts;
}

void PrintMatrix(const std::string& name, const Matrix2x2& m) {
  std::cout << name << " = [["
            << std::setw(6) << m[0] << ", " << std::setw(6) << m[1] << "], ["
            << std::setw(6) << m[2] << ", " << std::setw(6) << m[3] << "]]\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = ParseOptions(argc, argv);

  std::mt19937 rng(opts.seed);
  std::uniform_int_distribution<int32_t> mat_dist(-8, 8);
  std::uniform_int_distribution<int32_t> scalar_dist(-5, 5);

  bool all_passed = true;

  for (int trial = 0; trial < opts.trials; ++trial) {
    Matrix2x2 A{};
    Matrix2x2 B{};
    for (std::size_t i = 0; i < A.size(); ++i) {
      A[i] = mat_dist(rng);
      B[i] = mat_dist(rng);
    }
    const int32_t s = scalar_dist(rng);

    const Matrix2x2 C_host = MatMul(A, B);
    const Matrix2x2 D_host = VecAdd(C_host, A);
    const Matrix2x2 E_host = Scale(D_host, s);

    mini_tpu::TpuSession session;
    session.reset();

    session.memcpy(mini_tpu::BRAM_ADDR_A, A.data(), mini_tpu::MATRIX_WORDS,
                   mini_tpu::MemcpyDirection::HostToTpu);
    session.memcpy(mini_tpu::BRAM_ADDR_B, B.data(), mini_tpu::MATRIX_WORDS,
                   mini_tpu::MemcpyDirection::HostToTpu);
    int32_t scalar_copy = s;
    session.memcpy(mini_tpu::BRAM_ADDR_S, &scalar_copy,
                   mini_tpu::SCALAR_WORDS,
                   mini_tpu::MemcpyDirection::HostToTpu);

    auto& program = session.program();
    program.push_back(mini_tpu::MakeMatMul(mini_tpu::BRAM_ADDR_A,
                                           mini_tpu::BRAM_ADDR_B,
                                           mini_tpu::BRAM_ADDR_C));
    program.push_back(mini_tpu::MakeVecAdd(mini_tpu::BRAM_ADDR_C,
                                           mini_tpu::BRAM_ADDR_A,
                                           mini_tpu::BRAM_ADDR_D,
                                           mini_tpu::MATRIX_WORDS));
    program.push_back(mini_tpu::MakeScale(mini_tpu::BRAM_ADDR_D,
                                          mini_tpu::BRAM_ADDR_E,
                                          mini_tpu::MATRIX_WORDS, s));

    Matrix2x2 E_hw{};
    session.memcpy(mini_tpu::BRAM_ADDR_E, E_hw.data(), mini_tpu::MATRIX_WORDS,
                   mini_tpu::MemcpyDirection::TpuToHost);

    session.run();

    bool trial_pass = true;
    for (std::size_t i = 0; i < E_hw.size(); ++i) {
      if (E_hw[i] != E_host[i]) {
        trial_pass = false;
        break;
      }
    }

    if (!trial_pass) {
      all_passed = false;
      std::cout << "Trial " << trial << " FAILED\n";
      PrintMatrix("A", A);
      PrintMatrix("B", B);
      std::cout << "s = " << s << "\n";
      PrintMatrix("E_host", E_host);
      PrintMatrix("E_hw", E_hw);
      break;
    }
  }

  if (all_passed) {
    std::cout << "All " << opts.trials << " trials passed.\n";
    return 0;
  }
  return 1;
}


