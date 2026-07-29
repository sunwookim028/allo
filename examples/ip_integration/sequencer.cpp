/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <hls_stream.h>
#include <stdint.h>

#define N 32        // Vector length
#define IMEM_SIZE 4 // Length of program
#define INSTR_W 4   // Size of one instruction
#define NEXT 2      // Number of external memory slots
#define NIBUF 2     // Number of operands/input buffer size

void sequencer(
    int8_t ctrl[1], int32_t d_addr[1],
    int8_t prog_in[4], // TODO: Add support for #define in parameter list
    int32_t ext_in[64], int32_t ext_out[32], hls::stream<bool> &enable_s,
    hls::stream<int32_t> &a_s, hls::stream<int32_t> &b_s,
    hls::stream<int32_t> &out_s) {
  static int8_t imem[IMEM_SIZE * INSTR_W] = {};
  int32_t in_buf[NIBUF * N] = {};
  int32_t out_buf[N] = {};
  int8_t op = 0;
  int8_t f1 = 0;
  int8_t f2 = 0;
  int8_t f3 = 0;

  if (ctrl[0] == 1) {
    int32_t base = d_addr[0] * INSTR_W;
    enable_s.write(0);
    for (int w = 0; w < INSTR_W; w++) {
      imem[base + w] = prog_in[w];
    }
  } else {
    enable_s.write(1);
    for (int pc = 0; pc < IMEM_SIZE; pc++) {
      // Decode the instruction at pc
      int32_t base2 = pc * 4;
      op = imem[base2];
      f1 = imem[base2 + 1];
      f2 = imem[base2 + 2];
      f3 = imem[base2 + 3];

      // Execute the decoded instruction
      if (op == 0) {
        for (int i = 0; i < N; i++) {
          in_buf[f1 * N + i] = ext_in[f2 * N + i];
        }
      } else if (op == 1) {
        for (int i = 0; i < N; i++) {
          ext_out[i] = out_buf[i];
        }
      } else if (op == 2) {
        for (int i = 0; i < N; i++) {
          a_s.write(in_buf[i]);
          b_s.write(in_buf[i + N]);
        }
        for (int j = 0; j < N; j++) {
          out_buf[j] = out_s.read();
        }
      }
    }
  }
}