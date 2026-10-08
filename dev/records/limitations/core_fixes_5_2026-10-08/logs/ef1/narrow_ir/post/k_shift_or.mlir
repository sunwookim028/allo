module {
  func.func @k_shift_or(%arg0: memref<8xi32>, %arg1: memref<8xi32>) attributes {itypes = "ss", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a"} : memref<8xi32>
      %alloc = memref.alloc() {name = "x"} : memref<i32>
      affine.store %0, %alloc[] {to = "x"} : memref<i32>
      %1 = affine.load %alloc[] {from = "x"} : memref<i32>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_0 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %c4_i32_1 = arith.constant 4 : i32
      %2 = arith.shli %c1_i32_0, %c4_i32_1 : i32
      %3 = arith.ori %1, %2 : i32
      %4 = affine.load %alloc[] {from = "x"} : memref<i32>
      %c2_i32 = arith.constant 2 : i32
      %c2_i32_2 = arith.constant 2 : i32
      %5 = arith.shrsi %4, %c2_i32_2 : i32
      %6 = arith.xori %3, %5 : i32
      affine.store %6, %arg1[%arg2] {to = "o"} : memref<8xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
