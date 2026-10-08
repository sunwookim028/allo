module {
  func.func @k_aug(%arg0: memref<8xi16>, %arg1: memref<8xi16>) attributes {itypes = "uu", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a", unsigned} : memref<8xi16>
      %alloc = memref.alloc() {name = "x", unsigned} : memref<i16>
      affine.store %0, %alloc[] {to = "x"} : memref<i16>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_0 = arith.constant 1 : i32
      %c15_i32 = arith.constant 15 : i32
      %c15_i32_1 = arith.constant 15 : i32
      %1 = arith.shli %c1_i32_0, %c15_i32_1 : i32
      %2 = affine.load %alloc[] {from = "x", unsigned} : memref<i16>
      %3 = arith.extui %2 : i16 to i32
      %4 = arith.ori %3, %1 : i32
      %5 = arith.trunci %4 {unsigned} : i32 to i16
      affine.store %5, %alloc[] {to = "x"} : memref<i16>
      %6 = affine.load %alloc[] {from = "x", unsigned} : memref<i16>
      %7 = arith.extui %6 : i16 to i33
      %c70000_i32 = arith.constant 70000 : i32
      %c70000_i32_2 = arith.constant 70000 : i32
      %8 = arith.extsi %c70000_i32_2 : i32 to i33
      %9 = arith.addi %7, %8 : i33
      %10 = arith.trunci %9 {unsigned} : i33 to i16
      affine.store %10, %alloc[] {to = "x"} : memref<i16>
      %11 = affine.load %alloc[] {from = "x", unsigned} : memref<i16>
      affine.store %11, %arg1[%arg2] {to = "o"} : memref<8xi16>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
