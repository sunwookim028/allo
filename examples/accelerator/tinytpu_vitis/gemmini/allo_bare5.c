// Accelerator-plus-RoCC-dispatch window at all five shapes.
// Identical methodology to the 161-cycle 4x4x4 measurement:
//   fill A,B from the CPU  ->  rdcycle  ->  5 configs, one loop_ws, fence  ->  rdcycle
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#define MAXDIM 16
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

// one hardware loop_ws covering tile (ti,tj,tk); spad ids 1,1 as tiled_matmul_outer
// passes them when I0==J0==K0==1 (a_reuse && b_reuse both true).
#define BARE(ti,tj,tk) ({ uint64_t _a=read_cycles(); \
  CFG5 \
  gemmini_loop_ws(ti,tj,tk, 0,0,0, (elem_t*)A,(elem_t*)B,NULL,(elem_t*)C, \
     MAXDIM,MAXDIM,0,MAXDIM, false,false,false,false,true, NO_ACTIVATION, 1,1, false); \
  gemmini_fence(); uint64_t _b=read_cycles(); _b-_a; })

// the 4x4x4 point re-measured with spad ids 0,0 -- the variant used earlier
#define BARE00(ti,tj,tk) ({ uint64_t _a=read_cycles(); \
  CFG5 \
  gemmini_loop_ws(ti,tj,tk, 0,0,0, (elem_t*)A,(elem_t*)B,NULL,(elem_t*)C, \
     MAXDIM,MAXDIM,0,MAXDIM, false,false,false,false,true, NO_ACTIVATION, 0,0, false); \
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

int main(){
  gemmini_flush(0);
  printf("GEMMINI DIM=%d elem_t_bytes=%d\n",DIM,(int)sizeof(elem_t));
  tiles(4,4,4,"4x4x4"); tiles(8,8,8,"8x8x8"); tiles(12,12,12,"12x12x12");
  tiles(16,8,16,"16x16x8"); tiles(16,16,16,"16x16x16");
  fill(16,16,A); fill(16,16,B);
  { volatile uint64_t w; w=BARE(1,1,1); w=BARE(4,4,4); (void)w; }   // warm
  // gemm(4,4,4): tiled_matmul_auto(dim_I=4,dim_J=4,dim_K=4)
  fill(4,4,A); fill(4,4,B);   printf("BARE 4x4x4 %llu\n",   (unsigned long long)BARE(1,1,1));
  fill(4,4,A); fill(4,4,B);   printf("BARE 4x4x4 %llu\n",   (unsigned long long)BARE(1,1,1));
  fill(4,4,A); fill(4,4,B);   printf("BARE00 4x4x4 %llu\n", (unsigned long long)BARE00(1,1,1));
  // gemm(8,8,8)
  fill(8,8,A); fill(8,8,B);   printf("BARE 8x8x8 %llu\n",   (unsigned long long)BARE(2,2,2));
  fill(8,8,A); fill(8,8,B);   printf("BARE 8x8x8 %llu\n",   (unsigned long long)BARE(2,2,2));
  // gemm(12,12,12)
  fill(12,12,A); fill(12,12,B); printf("BARE 12x12x12 %llu\n",(unsigned long long)BARE(3,3,3));
  fill(12,12,A); fill(12,12,B); printf("BARE 12x12x12 %llu\n",(unsigned long long)BARE(3,3,3));
  // gemm(16,16,8): I=16,K=16,J=8 -> tiled_matmul_auto(dim_I=16,dim_J=8,dim_K=16)
  fill(16,16,A); fill(16,8,B);  printf("BARE 16x16x8 %llu\n", (unsigned long long)BARE(4,2,4));
  fill(16,16,A); fill(16,8,B);  printf("BARE 16x16x8 %llu\n", (unsigned long long)BARE(4,2,4));
  // gemm(16,16,16)
  fill(16,16,A); fill(16,16,B); printf("BARE 16x16x16 %llu\n",(unsigned long long)BARE(4,4,4));
  fill(16,16,A); fill(16,16,B); printf("BARE 16x16x16 %llu\n",(unsigned long long)BARE(4,4,4));
  printf("DONE\n"); return 0;
}
