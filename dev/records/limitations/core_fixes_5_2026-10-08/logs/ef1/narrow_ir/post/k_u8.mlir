module {
  func.func @k_u8(%arg0: memref<8xi8>, %arg1: memref<8xi8>) attributes {itypes = "uu", otypes = ""} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "a", unsigned} : memref<8xi8>
      %alloc = memref.alloc() {name = "x", unsigned} : memref<i8>
      affine.store %0, %alloc[] {to = "x"} : memref<i8>
      %1 = affine.load %alloc[] {from = "x", unsigned} : memref<i8>
      %2 = arith.extui %1 : i8 to i33
      %c300_i32 = arith.constant 300 : i32
      %c300_i32_0 = arith.constant 300 : i32
      %3 = arith.extsi %c300_i32_0 : i32 to i33
      %4 = arith.addi %2, %3 : i33
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_1 = arith.constant 255 : i32
      %5 = arith.extsi %c255_i32_1 : i32 to i33
      %6 = arith.andi %4, %5 : i33
      %7 = arith.trunci %6 {unsigned} : i33 to i8
      %alloc_2 = memref.alloc() {name = "y", unsigned} : memref<i8>
      affine.store %7, %alloc_2[] {to = "y"} : memref<i8>
      %8 = affine.load %alloc_2[] {from = "y", unsigned} : memref<i8>
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_3 = arith.constant 1 : i32
      %c7_i32 = arith.constant 7 : i32
      %c7_i32_4 = arith.constant 7 : i32
      %9 = arith.shli %c1_i32_3, %c7_i32_4 : i32
      %10 = arith.extui %8 : i8 to i32
      %11 = arith.ori %10, %9 : i32
      %12 = arith.trunci %11 {unsigned} : i32 to i8
      affine.store %12, %arg1[%arg2] {to = "o"} : memref<8xi8>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
