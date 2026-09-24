// Cycle-count benchmark matched to the Allo TinyTPU RTL-sim shapes.
//
// The stock matmul tests are correctness tests: they never call read_cycles,
// so they yield no timing. This runs the same GEMM and ReLU-MLP shapes the
// TinyTPU testbench runs, brackets each with rdcycle, and prints the count --
// end to end, including RoCC decode, mvin/mvout and the tiling loop on Rocket,
// which is what makes it comparable to TinyTPU's start-to-done count.
//
// Data-type agnostic: written against `elem_t` and filled with values in
// [-4, 4], so the same source builds against either config. Two are relevant:
//
//   FPGemminiRocketConfig        DIM=4,  elem_t=float   (fp32, 4x4 mesh)
//   Int8Dim4GemminiRocketConfig  DIM=4,  elem_t=int8_t  (int8/int32, 4x4 mesh)
//
// The int8 one is the matched baseline for the Allo TinyTPU: Gemmini's own
// default data type at the array size TinyTPU generates. The stock
// GemminiRocketConfig is int8 but 16x16, so it differs in mesh size; the FP32
// config is 4x4 but fp32, so it differs in data type. Neither alone is a fair
// comparison, which is why the DIM=4 int8 config was added
// (gemmini.GemminiCustomConfigs.int8Dim4Config).

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define MAXDIM 16
// Tall-skinny shapes: the same K and N, a longer M panel. TinyTPU's `mm`
// streams one panel per instruction against a stationary weight tile, so a
// longer panel amortizes its per-instruction fixed cost; these shapes measure
// whether Gemmini's advantage over it is that fixed cost or something
// structural. Separate arrays so MAXDIM (and the existing shapes) are
// untouched.
#define TALLM 64

static elem_t A[MAXDIM][MAXDIM] row_align(1);
static elem_t B[MAXDIM][MAXDIM] row_align(1);
static elem_t C[MAXDIM][MAXDIM] row_align(1);
static elem_t H[MAXDIM][MAXDIM] row_align(1);
static elem_t TA[TALLM][MAXDIM] row_align(1);
static elem_t TC[TALLM][MAXDIM] row_align(1);

// Self-contained LCG: the baremetal environment has no rand()/srand(), and
// the values only need to be varied, not statistically good.
static uint32_t rng_state = 1;
static uint32_t nextrand(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return (rng_state >> 16) & 0x7fff;
}

static void fill(size_t r, size_t c, elem_t m[MAXDIM][MAXDIM]) {
  for (size_t i = 0; i < r; i++)
    for (size_t j = 0; j < c; j++)
      m[i][j] = (elem_t)((int)(nextrand() % 9) - 4);
}

static uint64_t gemm(size_t I, size_t K, size_t J, int act) {
  fill(I, K, A);
  fill(K, J, B);
  uint64_t t0 = read_cycles();
  tiled_matmul_auto(I, J, K,
                    (elem_t*)A, (elem_t*)B, NULL, (elem_t*)C,
                    MAXDIM, MAXDIM, 0, MAXDIM,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    act, ACC_SCALE_IDENTITY, 0, false,
                    false, false, false, false, 0, WS);
  uint64_t t1 = read_cycles();
  return t1 - t0;
}

// x[M,K] @ W1[K,H] -> relu -> @ W2[H,N] -> relu, both layers fused-activated,
// mirroring the TinyTPU MLP program.
static uint64_t mlp(size_t M, size_t K, size_t Hd, size_t N) {
  fill(M, K, A);
  fill(K, Hd, B);
  uint64_t t0 = read_cycles();
  tiled_matmul_auto(M, Hd, K, (elem_t*)A, (elem_t*)B, NULL, (elem_t*)H,
                    MAXDIM, MAXDIM, 0, MAXDIM,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    RELU, ACC_SCALE_IDENTITY, 0, false,
                    false, false, false, false, 0, WS);
  tiled_matmul_auto(M, N, Hd, (elem_t*)H, (elem_t*)B, NULL, (elem_t*)C,
                    MAXDIM, MAXDIM, 0, MAXDIM,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    RELU, ACC_SCALE_IDENTITY, 0, false,
                    false, false, false, false, 0, WS);
  uint64_t t1 = read_cycles();
  return t1 - t0;
}

static void fill_tall(size_t r, size_t c, elem_t m[TALLM][MAXDIM]) {
  for (size_t i = 0; i < r; i++)
    for (size_t j = 0; j < c; j++)
      m[i][j] = (elem_t)((int)(nextrand() % 9) - 4);
}

static uint64_t gemm_tall(size_t I, size_t K, size_t J) {
  fill_tall(I, K, TA);
  fill(K, J, B);
  uint64_t t0 = read_cycles();
  tiled_matmul_auto(I, J, K,
                    (elem_t*)TA, (elem_t*)B, NULL, (elem_t*)TC,
                    MAXDIM, MAXDIM, 0, MAXDIM,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false, false, false, 0, WS);
  uint64_t t1 = read_cycles();
  return t1 - t0;
}

int main() {
  gemmini_flush(0);

  printf("GEMMINI DIM=%d elem_t_bytes=%d\n", DIM, (int)sizeof(elem_t));

  // warm the accelerator so the first timed run is not paying cold-start
  gemm(4, 4, 4, NO_ACTIVATION);

  printf("GEMM 4x4x4 %llu\n",    (unsigned long long)gemm(4, 4, 4, NO_ACTIVATION));
  printf("GEMM 8x8x8 %llu\n",    (unsigned long long)gemm(8, 8, 8, NO_ACTIVATION));
  printf("GEMM 12x12x12 %llu\n", (unsigned long long)gemm(12, 12, 12, NO_ACTIVATION));
  printf("GEMM 16x16x8 %llu\n",  (unsigned long long)gemm(16, 16, 8, NO_ACTIVATION));
  printf("GEMM 16x16x16 %llu\n", (unsigned long long)gemm(16, 16, 16, NO_ACTIVATION));
  printf("MLP 4.8.8.4 %llu\n",   (unsigned long long)mlp(4, 8, 8, 4));
  printf("MLP 8.8.8.8 %llu\n",   (unsigned long long)mlp(8, 8, 8, 8));
  printf("MLP 8.16.16.8 %llu\n", (unsigned long long)mlp(8, 16, 16, 8));
  printf("GEMM 32x16x16 %llu\n", (unsigned long long)gemm_tall(32, 16, 16));
  printf("GEMM 64x16x16 %llu\n", (unsigned long long)gemm_tall(64, 16, 16));
  printf("DONE\n");
  return 0;
}
