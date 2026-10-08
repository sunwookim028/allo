module {
  func.func @k_wide_small(%arg0: memref<8xi64>, %arg1: memref<8xi64>) attributes {itypes = "uu", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a", unsigned} : memref<8xi64>
      %alloc = memref.alloc() {name = "x", unsigned} : memref<i64>
      affine.store %0, %alloc[] {to = "x"} : memref<i64>
      %1 = affine.load %alloc[] {from = "x", unsigned} : memref<i64>
      %c5_i32 = arith.constant 5 : i32
      %c5_i32_0 = arith.constant 5 : i32
      %2 = arith.extsi %c5_i32_0 : i32 to i64
      %3 = arith.ori %1, %2 : i64
      %4 = affine.load %alloc[] {from = "x", unsigned} : memref<i64>
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_1 = arith.constant 255 : i32
      %5 = arith.extsi %c255_i32_1 : i32 to i64
      %6 = arith.andi %4, %5 : i64
      %7 = arith.extsi %3 : i64 to i65
      %8 = arith.extsi %6 : i64 to i65
      %9 = arith.addi %7, %8 : i65
      %10 = arith.extsi %9 : i65 to i66
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_2 = arith.constant 1 : i32
      %11 = arith.extsi %c1_i32_2 : i32 to i66
      %12 = arith.subi %10, %11 : i66
      %13 = arith.trunci %12 {unsigned} : i66 to i64
      affine.store %13, %arg1[%arg2] {to = "o"} : memref<8xi64>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
