// Gemmini's accelerator-plus-RoCC-dispatch window at the STEADY-STATE shapes,
// and at the five latency shapes on the same build so the two sets are
// directly comparable. See docs/source/designs/benchmarks.rst.
//
// Identical methodology to allo_bare5.c, which this file extends rather than
// replaces (allo_bare5.c stays as the provenance of the published
// 161/220/347/391/593):
//   fill A,B from the CPU  ->  rdcycle  ->  5 configs, one loop_ws, fence  ->  rdcycle
//
// WHY MAXDIM IS A -D AND NOT A CONSTANT. MAXDIM is the DRAM row stride of
// every operand, on BOTH machines: allo_bare5.c passes it as the loop_ws
// stride, and TinyTPU-isa's `dma_ld` addresses `A[row * MAXDIM + col]`. So the
// same logical shape costs differently on a build with a bigger MAXDIM --
// more cache lines touched here, a longer burst span there -- and a
// comparison is only matched if the two sides use the SAME MAXDIM. The
// steady-state set needs 64; the published five were measured at 16. Both are
// built from this one file:
//     -DMAXDIM=64   the steady-state set (and the latency set beside it)
//     -DMAXDIM=16   reproduces allo_bare5.c's five, as a control
//
// Every shape below is ONE hardware loop_ws: the driver's own tiling search
// (`tiles()`, replicated from gemmini.h as in allo_bare5.c) is run and printed
// for each, so the call count is proven at runtime rather than assumed. The
// binding limit is the double-buffered accumulator, tI*tJ <= ACC_ROWS/2/DIM;
// at DIM=4 that is tI*tJ <= 512, so 64x64x64 (16,16) fits with room and the
// first shape that would split is around 96x96x96.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef MAXDIM
#define MAXDIM 64
#endif

static elem_t A[MAXDIM][MAXDIM] row_align(1);
static elem_t B[MAXDIM][MAXDIM] row_align(1);
static elem_t C[MAXDIM][MAXDIM] row_align(1);
static uint32_t rs=1; static uint32_t nr(void){rs=rs*1103515245u+12345u;return (rs>>16)&0x7fff;}
static void fill(size_t r,size_t c,elem_t m[MAXDIM][MAXDIM]){
  for(size_t i=0;i<r;i++)for(size_t j=0;j<c;j++)m[i][j]=(elem_t)((int)(nr()%9)-4);}

#define CFG5 \
  gemmini_extended_config_ex(WS, NO_ACTIVATION & 3, 0, 1, false, false); \
  gemmini_extended_config_st(MAXDIM*sizeof(elem_t), NO_ACTIVATION & 3, ACC_SCALE_IDENTITY); \
  gemmini_extended3_config_ld(MAXDIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0); \
  gemmini_extended3_config_ld(MAXDIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); \
  gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);

// one hardware loop_ws covering tile (ti,tj,tk); spad ids 1,1 as
// tiled_matmul_outer passes them when I0==J0==K0==1 (a_reuse && b_reuse).
#define BARE(ti,tj,tk) ({ uint64_t _a=read_cycles(); \
  CFG5 \
  gemmini_loop_ws(ti,tj,tk, 0,0,0, (elem_t*)A,(elem_t*)B,NULL,(elem_t*)C, \
     MAXDIM,MAXDIM,0,MAXDIM, false,false,false,false,true, NO_ACTIVATION, 1,1, false); \
  gemmini_fence(); uint64_t _b=read_cycles(); _b-_a; })

// replicate tiled_matmul_auto's tiling search to PROVE the loop_ws call count
static void tiles(size_t dI,size_t dJ,size_t dK,const char*nm){
#define mspad (BANK_NUM*BANK_ROWS/2)
#define macc  (ACC_ROWS/2)
  size_t pI=(dI/DIM+(dI%DIM!=0))*DIM, pJ=(dJ/DIM+(dJ%DIM!=0))*DIM, pK=(dK/DIM+(dK%DIM!=0))*DIM;
  size_t dbij=(size_t)sqrt(((ACC_ROWS/2)/DIM)), dbk=(((BANK_NUM*BANK_ROWS/2)/2)/DIM)/dbij;
  size_t tI=pI/DIM<dbij?pI/DIM:dbij, tJ=pJ/DIM<dbij?pJ/DIM:dbij, tK=pK/DIM<dbk?pK/DIM:dbk;
  while(1){ int inc=0;
    if((tI*tK+tK*(tJ+1))*DIM<=mspad && (tI*(tJ+1))*DIM<=macc && (tJ+1)*DIM<=pJ){tJ++;inc=1;}
    if(((tI+1)*tK+tK*tJ)*DIM<=mspad && ((tI+1)*tJ)*DIM<=macc && (tI+1)*DIM<=pI){tI++;inc=1;}
    if((tI*(tK+1)+(tK+1)*tJ)*DIM<=mspad && (tK+1)*DIM<=pK){tK++;inc=1;}
    if(!inc)break; }
  size_t I0=pI/(tI*DIM)+(pI%(tI*DIM)!=0), J0=pJ/(tJ*DIM)+(pJ%(tJ*DIM)!=0), K0=pK/(tK*DIM)+(pK%(tK*DIM)!=0);
  printf("TILES %s tile=%d,%d,%d  I0J0K0=%d,%d,%d  loop_ws_calls=%d\n",
    nm,(int)tI,(int)tJ,(int)tK,(int)I0,(int)J0,(int)K0,(int)(I0*J0*K0));
}

// THE ex_accumulate BIT, AND WHY BOTH ARE MEASURED.
// `BARE` above passes gemmini_loop_ws's 5th bool -- ex_accumulate -- as
// literal `true`, inherited from allo_bare5.c. The real driver does not:
// `sp_tiled_matmul_ws` computes it as `!no_bias || D == NULL`
// (gemmini.h:701), and for a no-bias matmul `tiled_matmul_outer` has already
// replaced the NULL D with a dummy `(void*)1` (gemmini.h:742-744), so
// `D == NULL` is false and the driver passes **false**. Setting it asks the
// accumulator to accumulate onto whatever is resident instead of
// overwriting, which adds no RoCC commands but does turn the k=0 accumulator
// writes into read-modify-writes, so it is not guaranteed cycle-neutral.
// Both are therefore measured at every shape and the difference is printed.
// If it is zero the published 161/220/347/391/593 stand as they are; if it is
// not, they need a footnote and this file says so.
#define BAREF(ti,tj,tk) ({ uint64_t _a=read_cycles(); \
  CFG5 \
  gemmini_loop_ws(ti,tj,tk, 0,0,0, (elem_t*)A,(elem_t*)B,NULL,(elem_t*)C, \
     MAXDIM,MAXDIM,0,MAXDIM, false,false,false,false,false, NO_ACTIVATION, 1,1, false); \
  gemmini_fence(); uint64_t _b=read_cycles(); _b-_a; })

// One shape: "MxKxN" is gemm(I=M, K=K, J=N), so tiles(dI=M,dJ=N,dK=K) and
// BARE(M/DIM, N/DIM, K/DIM). Two trials, as allo_bare5.c does, because the
// 4x4x4 point was the one that moved (161 vs 144).
#define SHAPE(M,K,N) do { \
  fill(M,K,A); fill(K,N,B); \
  printf("BARE %dx%dx%d %llu\n", M,K,N, (unsigned long long)BARE((M)/DIM,(N)/DIM,(K)/DIM)); \
  fill(M,K,A); fill(K,N,B); \
  printf("BARE %dx%dx%d %llu\n", M,K,N, (unsigned long long)BARE((M)/DIM,(N)/DIM,(K)/DIM)); \
  fill(M,K,A); fill(K,N,B); \
  printf("BAREF %dx%dx%d %llu\n", M,K,N, (unsigned long long)BAREF((M)/DIM,(N)/DIM,(K)/DIM)); \
  fill(M,K,A); fill(K,N,B); \
  printf("BAREF %dx%dx%d %llu\n", M,K,N, (unsigned long long)BAREF((M)/DIM,(N)/DIM,(K)/DIM)); \
} while (0)

int main(){
  gemmini_flush(0);
  printf("GEMMINI DIM=%d elem_t_bytes=%d MAXDIM=%d\n",DIM,(int)sizeof(elem_t),MAXDIM);
  // The tiling search, printed for every shape this build runs.
  // A shape is only runnable when every dimension is a multiple of DIM: one
  // `loop_ws` tile argument is `dim / DIM`, so 4x4x4 at DIM=8 would ask for
  // zero tiles. Same rule as our own `runnable()` (bench_isa.py), which
  // filters on T.
#if DIM <= 4
  tiles(4,4,4,"4x4x4"); tiles(12,12,12,"12x12x12");
#endif
#if DIM <= 8
  tiles(8,8,8,"8x8x8"); tiles(16,8,16,"16x16x8");
#endif
  tiles(16,16,16,"16x16x16");
#if MAXDIM >= 64
  tiles(32,32,32,"32x32x32"); tiles(48,48,48,"48x48x48");
  tiles(64,64,64,"64x64x64"); tiles(64,64,32,"64x32x64");
  tiles(32,32,64,"32x64x32");
#endif
  // Warm at the largest shape this build runs, as allo_bare5.c warms at 4,4,4.
  fill(MAXDIM,MAXDIM,A); fill(MAXDIM,MAXDIM,B);
  { volatile uint64_t w; w=BARE(1,1,1); w=BARE(MAXDIM/DIM,MAXDIM/DIM,MAXDIM/DIM); (void)w; }

  // --- the latency set (the published five) ---
#if DIM <= 4
  SHAPE(4,4,4); SHAPE(12,12,12);
#endif
#if DIM <= 8
  SHAPE(8,8,8);
  SHAPE(16,16,8);            // M=16, K=16, N=8
#endif
  SHAPE(16,16,16);
#if MAXDIM >= 64
  // --- the steady-state set ---
  SHAPE(32,32,32); SHAPE(48,48,48); SHAPE(64,64,64);
  SHAPE(64,32,64);           // M=64, K=32, N=64
  SHAPE(32,64,32);           // M=32, K=64, N=32
#endif
  printf("DONE\n"); return 0;
}
