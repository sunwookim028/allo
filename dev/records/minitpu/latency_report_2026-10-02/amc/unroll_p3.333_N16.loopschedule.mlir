module {
  amc.memory @mem0() -> !amc.port<16xi16, static rw(1, 1)> {
    %0 = amc.alloc on bram : !amc.ram<16xi16>
    %1 = amc.create_port(%0 : !amc.ram<16xi16>) : !amc.port<16xi16, static rw(1, 1)>
    amc.expose %1 : !amc.port<16xi16, static rw(1, 1)>
  }
  amc.memory @mem1() -> !amc.port<16xi16, static rw(1, 1)> {
    %0 = amc.alloc on bram : !amc.ram<16xi16>
    %1 = amc.create_port(%0 : !amc.ram<16xi16>) : !amc.port<16xi16, static rw(1, 1)>
    amc.expose %1 : !amc.port<16xi16, static rw(1, 1)>
  }
  amc.memory @mem2() -> !amc.port<16xi16, static rw(1, 1)> {
    %0 = amc.alloc on bram : !amc.ram<16xi16>
    %1 = amc.create_port(%0 : !amc.ram<16xi16>) : !amc.port<16xi16, static rw(1, 1)>
    amc.expose %1 : !amc.port<16xi16, static rw(1, 1)>
  }
  oplib.library @bf16_add_bits_amc_library {
    oplib.operator @i6_addi_l0 latency<0>, incDelay<0.054261173967600773>, outDelay<0.054261173967600773> {
      oplib.target @arith_addi_i6(%arg0: i6, %arg1: i6) -> i6 {
        %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i6, i6) : i6
        oplib.output %0 : i6
      }
      oplib.hw_match(@arith_addi_i6 : (i6, i6) -> i6) produce (in %arg0 : i6, in %arg1 : i6) {
        %0 = comb.add %arg0, %arg1 : i6
        oplib.hw_return %0 : i6
      }
    }
    oplib.operator @i8_addi_l0 latency<0>, incDelay<1.400000e-01>, outDelay<1.400000e-01> {
      oplib.target @arith_addi_i8(%arg0: i8, %arg1: i8) -> i8 {
        %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i8, i8) : i8
        oplib.output %0 : i8
      }
      oplib.hw_match(@arith_addi_i8 : (i8, i8) -> i8) produce (in %arg0 : i8, in %arg1 : i8) {
        %0 = comb.add %arg0, %arg1 : i8
        oplib.hw_return %0 : i8
      }
    }
    oplib.operator @i9_addi_l0 latency<0>, incDelay<0.1489032533688972>, outDelay<0.1489032533688972> {
      oplib.target @arith_addi_i9(%arg0: i9, %arg1: i9) -> i9 {
        %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i9, i9) : i9
        oplib.output %0 : i9
      }
      oplib.hw_match(@arith_addi_i9 : (i9, i9) -> i9) produce (in %arg0 : i9, in %arg1 : i9) {
        %0 = comb.add %arg0, %arg1 : i9
        oplib.hw_return %0 : i9
      }
    }
    oplib.operator @i18_addi_l0 latency<0>, incDelay<0.31721862290553227>, outDelay<0.31721862290553227> {
      oplib.target @arith_addi_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_addi_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.add %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i5_subi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_subi_i5(%arg0: i5, %arg1: i5) -> i5 {
        %0 = oplib.operation "arith.subi"(%arg0, %arg1 : i5, i5) : i5
        oplib.output %0 : i5
      }
      oplib.hw_match(@arith_subi_i5 : (i5, i5) -> i5) produce (in %arg0 : i5, in %arg1 : i5) {
        %0 = comb.sub %arg0, %arg1 : i5
        oplib.hw_return %0 : i5
      }
    }
    oplib.operator @i9_subi_l0 latency<0>, incDelay<0.14890325318995579>, outDelay<0.14890325318995579> {
      oplib.target @arith_subi_i9(%arg0: i9, %arg1: i9) -> i9 {
        %0 = oplib.operation "arith.subi"(%arg0, %arg1 : i9, i9) : i9
        oplib.output %0 : i9
      }
      oplib.hw_match(@arith_subi_i9 : (i9, i9) -> i9) produce (in %arg0 : i9, in %arg1 : i9) {
        %0 = comb.sub %arg0, %arg1 : i9
        oplib.hw_return %0 : i9
      }
    }
    oplib.operator @i18_subi_l0 latency<0>, incDelay<0.31721862259056055>, outDelay<0.31721862259056055> {
      oplib.target @arith_subi_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.subi"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_subi_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.sub %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i1_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i1(%arg0: i1, %arg1: i1) -> i1 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i1, i1) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_andi_i1 : (i1, i1) -> i1) produce (in %arg0 : i1, in %arg1 : i1) {
        %0 = comb.and %arg0, %arg1 : i1
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i2_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i2(%arg0: i2, %arg1: i2) -> i2 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i2, i2) : i2
        oplib.output %0 : i2
      }
      oplib.hw_match(@arith_andi_i2 : (i2, i2) -> i2) produce (in %arg0 : i2, in %arg1 : i2) {
        %0 = comb.and %arg0, %arg1 : i2
        oplib.hw_return %0 : i2
      }
    }
    oplib.operator @i3_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i3(%arg0: i3, %arg1: i3) -> i3 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i3, i3) : i3
        oplib.output %0 : i3
      }
      oplib.hw_match(@arith_andi_i3 : (i3, i3) -> i3) produce (in %arg0 : i3, in %arg1 : i3) {
        %0 = comb.and %arg0, %arg1 : i3
        oplib.hw_return %0 : i3
      }
    }
    oplib.operator @i4_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i4(%arg0: i4, %arg1: i4) -> i4 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i4, i4) : i4
        oplib.output %0 : i4
      }
      oplib.hw_match(@arith_andi_i4 : (i4, i4) -> i4) produce (in %arg0 : i4, in %arg1 : i4) {
        %0 = comb.and %arg0, %arg1 : i4
        oplib.hw_return %0 : i4
      }
    }
    oplib.operator @i5_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i5(%arg0: i5, %arg1: i5) -> i5 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i5, i5) : i5
        oplib.output %0 : i5
      }
      oplib.hw_match(@arith_andi_i5 : (i5, i5) -> i5) produce (in %arg0 : i5, in %arg1 : i5) {
        %0 = comb.and %arg0, %arg1 : i5
        oplib.hw_return %0 : i5
      }
    }
    oplib.operator @i6_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i6(%arg0: i6, %arg1: i6) -> i6 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i6, i6) : i6
        oplib.output %0 : i6
      }
      oplib.hw_match(@arith_andi_i6 : (i6, i6) -> i6) produce (in %arg0 : i6, in %arg1 : i6) {
        %0 = comb.and %arg0, %arg1 : i6
        oplib.hw_return %0 : i6
      }
    }
    oplib.operator @i7_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i7(%arg0: i7, %arg1: i7) -> i7 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i7, i7) : i7
        oplib.output %0 : i7
      }
      oplib.hw_match(@arith_andi_i7 : (i7, i7) -> i7) produce (in %arg0 : i7, in %arg1 : i7) {
        %0 = comb.and %arg0, %arg1 : i7
        oplib.hw_return %0 : i7
      }
    }
    oplib.operator @i8_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i8(%arg0: i8, %arg1: i8) -> i8 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i8, i8) : i8
        oplib.output %0 : i8
      }
      oplib.hw_match(@arith_andi_i8 : (i8, i8) -> i8) produce (in %arg0 : i8, in %arg1 : i8) {
        %0 = comb.and %arg0, %arg1 : i8
        oplib.hw_return %0 : i8
      }
    }
    oplib.operator @i9_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i9(%arg0: i9, %arg1: i9) -> i9 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i9, i9) : i9
        oplib.output %0 : i9
      }
      oplib.hw_match(@arith_andi_i9 : (i9, i9) -> i9) produce (in %arg0 : i9, in %arg1 : i9) {
        %0 = comb.and %arg0, %arg1 : i9
        oplib.hw_return %0 : i9
      }
    }
    oplib.operator @i10_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i10(%arg0: i10, %arg1: i10) -> i10 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i10, i10) : i10
        oplib.output %0 : i10
      }
      oplib.hw_match(@arith_andi_i10 : (i10, i10) -> i10) produce (in %arg0 : i10, in %arg1 : i10) {
        %0 = comb.and %arg0, %arg1 : i10
        oplib.hw_return %0 : i10
      }
    }
    oplib.operator @i11_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i11(%arg0: i11, %arg1: i11) -> i11 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i11, i11) : i11
        oplib.output %0 : i11
      }
      oplib.hw_match(@arith_andi_i11 : (i11, i11) -> i11) produce (in %arg0 : i11, in %arg1 : i11) {
        %0 = comb.and %arg0, %arg1 : i11
        oplib.hw_return %0 : i11
      }
    }
    oplib.operator @i12_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i12(%arg0: i12, %arg1: i12) -> i12 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i12, i12) : i12
        oplib.output %0 : i12
      }
      oplib.hw_match(@arith_andi_i12 : (i12, i12) -> i12) produce (in %arg0 : i12, in %arg1 : i12) {
        %0 = comb.and %arg0, %arg1 : i12
        oplib.hw_return %0 : i12
      }
    }
    oplib.operator @i13_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i13(%arg0: i13, %arg1: i13) -> i13 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i13, i13) : i13
        oplib.output %0 : i13
      }
      oplib.hw_match(@arith_andi_i13 : (i13, i13) -> i13) produce (in %arg0 : i13, in %arg1 : i13) {
        %0 = comb.and %arg0, %arg1 : i13
        oplib.hw_return %0 : i13
      }
    }
    oplib.operator @i14_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i14(%arg0: i14, %arg1: i14) -> i14 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i14, i14) : i14
        oplib.output %0 : i14
      }
      oplib.hw_match(@arith_andi_i14 : (i14, i14) -> i14) produce (in %arg0 : i14, in %arg1 : i14) {
        %0 = comb.and %arg0, %arg1 : i14
        oplib.hw_return %0 : i14
      }
    }
    oplib.operator @i15_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i15(%arg0: i15, %arg1: i15) -> i15 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i15, i15) : i15
        oplib.output %0 : i15
      }
      oplib.hw_match(@arith_andi_i15 : (i15, i15) -> i15) produce (in %arg0 : i15, in %arg1 : i15) {
        %0 = comb.and %arg0, %arg1 : i15
        oplib.hw_return %0 : i15
      }
    }
    oplib.operator @i16_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i16(%arg0: i16, %arg1: i16) -> i16 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i16, i16) : i16
        oplib.output %0 : i16
      }
      oplib.hw_match(@arith_andi_i16 : (i16, i16) -> i16) produce (in %arg0 : i16, in %arg1 : i16) {
        %0 = comb.and %arg0, %arg1 : i16
        oplib.hw_return %0 : i16
      }
    }
    oplib.operator @i17_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i17(%arg0: i17, %arg1: i17) -> i17 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i17, i17) : i17
        oplib.output %0 : i17
      }
      oplib.hw_match(@arith_andi_i17 : (i17, i17) -> i17) produce (in %arg0 : i17, in %arg1 : i17) {
        %0 = comb.and %arg0, %arg1 : i17
        oplib.hw_return %0 : i17
      }
    }
    oplib.operator @i18_andi_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_andi_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.andi"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_andi_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.and %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i1_ori_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_ori_i1(%arg0: i1, %arg1: i1) -> i1 {
        %0 = oplib.operation "arith.ori"(%arg0, %arg1 : i1, i1) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_ori_i1 : (i1, i1) -> i1) produce (in %arg0 : i1, in %arg1 : i1) {
        %0 = comb.or %arg0, %arg1 : i1
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i16_ori_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_ori_i16(%arg0: i16, %arg1: i16) -> i16 {
        %0 = oplib.operation "arith.ori"(%arg0, %arg1 : i16, i16) : i16
        oplib.output %0 : i16
      }
      oplib.hw_match(@arith_ori_i16 : (i16, i16) -> i16) produce (in %arg0 : i16, in %arg1 : i16) {
        %0 = comb.or %arg0, %arg1 : i16
        oplib.hw_return %0 : i16
      }
    }
    oplib.operator @i17_ori_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_ori_i17(%arg0: i17, %arg1: i17) -> i17 {
        %0 = oplib.operation "arith.ori"(%arg0, %arg1 : i17, i17) : i17
        oplib.output %0 : i17
      }
      oplib.hw_match(@arith_ori_i17 : (i17, i17) -> i17) produce (in %arg0 : i17, in %arg1 : i17) {
        %0 = comb.or %arg0, %arg1 : i17
        oplib.hw_return %0 : i17
      }
    }
    oplib.operator @i18_ori_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_ori_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.ori"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_ori_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.or %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i25_ori_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_ori_i25(%arg0: i25, %arg1: i25) -> i25 {
        %0 = oplib.operation "arith.ori"(%arg0, %arg1 : i25, i25) : i25
        oplib.output %0 : i25
      }
      oplib.hw_match(@arith_ori_i25 : (i25, i25) -> i25) produce (in %arg0 : i25, in %arg1 : i25) {
        %0 = comb.or %arg0, %arg1 : i25
        oplib.hw_return %0 : i25
      }
    }
    oplib.operator @i2_shli_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_shli_i2(%arg0: i2, %arg1: i2) -> i2 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i2, i2) : i2
        oplib.output %0 : i2
      }
      oplib.hw_match(@arith_shli_i2 : (i2, i2) -> i2) produce (in %arg0 : i2, in %arg1 : i2) {
        %0 = comb.shl %arg0, %arg1 : i2
        oplib.hw_return %0 : i2
      }
    }
    oplib.operator @i8_shli_l0 latency<0>, incDelay<0.054999999999999938>, outDelay<0.054999999999999938> {
      oplib.target @arith_shli_i8(%arg0: i8, %arg1: i8) -> i8 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i8, i8) : i8
        oplib.output %0 : i8
      }
      oplib.hw_match(@arith_shli_i8 : (i8, i8) -> i8) produce (in %arg0 : i8, in %arg1 : i8) {
        %0 = comb.shl %arg0, %arg1 : i8
        oplib.hw_return %0 : i8
      }
    }
    oplib.operator @i9_shli_l0 latency<0>, incDelay<0.061037459833031549>, outDelay<0.061037459833031549> {
      oplib.target @arith_shli_i9(%arg0: i9, %arg1: i9) -> i9 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i9, i9) : i9
        oplib.output %0 : i9
      }
      oplib.hw_match(@arith_shli_i9 : (i9, i9) -> i9) produce (in %arg0 : i9, in %arg1 : i9) {
        %0 = comb.shl %arg0, %arg1 : i9
        oplib.hw_return %0 : i9
      }
    }
    oplib.operator @i15_shli_l0 latency<0>, incDelay<0.16513562725182018>, outDelay<0.16513562725182018> {
      oplib.target @arith_shli_i15(%arg0: i15, %arg1: i15) -> i15 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i15, i15) : i15
        oplib.output %0 : i15
      }
      oplib.hw_match(@arith_shli_i15 : (i15, i15) -> i15) produce (in %arg0 : i15, in %arg1 : i15) {
        %0 = comb.shl %arg0, %arg1 : i15
        oplib.hw_return %0 : i15
      }
    }
    oplib.operator @i16_shli_l0 latency<0>, incDelay<0.16899999999999993>, outDelay<0.16899999999999993> {
      oplib.target @arith_shli_i16(%arg0: i16, %arg1: i16) -> i16 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i16, i16) : i16
        oplib.output %0 : i16
      }
      oplib.hw_match(@arith_shli_i16 : (i16, i16) -> i16) produce (in %arg0 : i16, in %arg1 : i16) {
        %0 = comb.shl %arg0, %arg1 : i16
        oplib.hw_return %0 : i16
      }
    }
    oplib.operator @i17_shli_l0 latency<0>, incDelay<0.19140539073557383>, outDelay<0.19140539073557383> {
      oplib.target @arith_shli_i17(%arg0: i17, %arg1: i17) -> i17 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i17, i17) : i17
        oplib.output %0 : i17
      }
      oplib.hw_match(@arith_shli_i17 : (i17, i17) -> i17) produce (in %arg0 : i17, in %arg1 : i17) {
        %0 = comb.shl %arg0, %arg1 : i17
        oplib.hw_return %0 : i17
      }
    }
    oplib.operator @i18_shli_l0 latency<0>, incDelay<0.20352690302106635>, outDelay<0.20352690302106635> {
      oplib.target @arith_shli_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_shli_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.shl %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i25_shli_l0 latency<0>, incDelay<0.27499743400098053>, outDelay<0.27499743400098053> {
      oplib.target @arith_shli_i25(%arg0: i25, %arg1: i25) -> i25 {
        %0 = oplib.operation "arith.shli"(%arg0, %arg1 : i25, i25) : i25
        oplib.output %0 : i25
      }
      oplib.hw_match(@arith_shli_i25 : (i25, i25) -> i25) produce (in %arg0 : i25, in %arg1 : i25) {
        %0 = comb.shl %arg0, %arg1 : i25
        oplib.hw_return %0 : i25
      }
    }
    oplib.operator @i16_shrui_l0 latency<0>, incDelay<0.16899999999999993>, outDelay<0.16899999999999993> {
      oplib.target @arith_shrui_i16(%arg0: i16, %arg1: i16) -> i16 {
        %0 = oplib.operation "arith.shrui"(%arg0, %arg1 : i16, i16) : i16
        oplib.output %0 : i16
      }
      oplib.hw_match(@arith_shrui_i16 : (i16, i16) -> i16) produce (in %arg0 : i16, in %arg1 : i16) {
        %0 = comb.shru %arg0, %arg1 : i16
        oplib.hw_return %0 : i16
      }
    }
    oplib.operator @i17_shrui_l0 latency<0>, incDelay<0.19140539626242303>, outDelay<0.19140539626242303> {
      oplib.target @arith_shrui_i17(%arg0: i17, %arg1: i17) -> i17 {
        %0 = oplib.operation "arith.shrui"(%arg0, %arg1 : i17, i17) : i17
        oplib.output %0 : i17
      }
      oplib.hw_match(@arith_shrui_i17 : (i17, i17) -> i17) produce (in %arg0 : i17, in %arg1 : i17) {
        %0 = comb.shru %arg0, %arg1 : i17
        oplib.hw_return %0 : i17
      }
    }
    oplib.operator @i18_shrui_l0 latency<0>, incDelay<0.20352690863730449>, outDelay<0.20352690863730449> {
      oplib.target @arith_shrui_i18(%arg0: i18, %arg1: i18) -> i18 {
        %0 = oplib.operation "arith.shrui"(%arg0, %arg1 : i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_shrui_i18 : (i18, i18) -> i18) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.shru %arg0, %arg1 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i1_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i1(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i1, i1) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_select_i1 : (i1, i1, i1) -> i1) produce (in %arg0 : i1, in %arg1 : i1, in %arg2 : i1) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i1
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i4_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i4(%arg0: i1, %arg1: i4, %arg2: i4) -> i4 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i4, i4) : i4
        oplib.output %0 : i4
      }
      oplib.hw_match(@arith_select_i4 : (i1, i4, i4) -> i4) produce (in %arg0 : i1, in %arg1 : i4, in %arg2 : i4) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i4
        oplib.hw_return %0 : i4
      }
    }
    oplib.operator @i5_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i5(%arg0: i1, %arg1: i5, %arg2: i5) -> i5 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i5, i5) : i5
        oplib.output %0 : i5
      }
      oplib.hw_match(@arith_select_i5 : (i1, i5, i5) -> i5) produce (in %arg0 : i1, in %arg1 : i5, in %arg2 : i5) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i5
        oplib.hw_return %0 : i5
      }
    }
    oplib.operator @i8_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i8(%arg0: i1, %arg1: i8, %arg2: i8) -> i8 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i8, i8) : i8
        oplib.output %0 : i8
      }
      oplib.hw_match(@arith_select_i8 : (i1, i8, i8) -> i8) produce (in %arg0 : i1, in %arg1 : i8, in %arg2 : i8) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i8
        oplib.hw_return %0 : i8
      }
    }
    oplib.operator @i9_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i9(%arg0: i1, %arg1: i9, %arg2: i9) -> i9 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i9, i9) : i9
        oplib.output %0 : i9
      }
      oplib.hw_match(@arith_select_i9 : (i1, i9, i9) -> i9) produce (in %arg0 : i1, in %arg1 : i9, in %arg2 : i9) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i9
        oplib.hw_return %0 : i9
      }
    }
    oplib.operator @i16_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i16(%arg0: i1, %arg1: i16, %arg2: i16) -> i16 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i16, i16) : i16
        oplib.output %0 : i16
      }
      oplib.hw_match(@arith_select_i16 : (i1, i16, i16) -> i16) produce (in %arg0 : i1, in %arg1 : i16, in %arg2 : i16) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i16
        oplib.hw_return %0 : i16
      }
    }
    oplib.operator @i17_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i17(%arg0: i1, %arg1: i17, %arg2: i17) -> i17 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i17, i17) : i17
        oplib.output %0 : i17
      }
      oplib.hw_match(@arith_select_i17 : (i1, i17, i17) -> i17) produce (in %arg0 : i1, in %arg1 : i17, in %arg2 : i17) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i17
        oplib.hw_return %0 : i17
      }
    }
    oplib.operator @i18_select_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_select_i18(%arg0: i1, %arg1: i18, %arg2: i18) -> i18 {
        %0 = oplib.operation "arith.select"(%arg0, %arg1, %arg2 : i1, i18, i18) : i18
        oplib.output %0 : i18
      }
      oplib.hw_match(@arith_select_i18 : (i1, i18, i18) -> i18) produce (in %arg0 : i1, in %arg1 : i18, in %arg2 : i18) {
        %0 = comb.mux %arg0, %arg1, %arg2 : i18
        oplib.hw_return %0 : i18
      }
    }
    oplib.operator @i1_cmpi_eq_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_eq_i1(%arg0: i1, %arg1: i1) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%arg0, %arg1 : i1, i1) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_eq_i1 : (i1, i1) -> i1) produce (in %arg0 : i1, in %arg1 : i1) {
        %0 = comb.icmp eq %arg0, %arg1 : i1
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i5_cmpi_eq_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_eq_i5(%arg0: i5, %arg1: i5) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%arg0, %arg1 : i5, i5) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_eq_i5 : (i5, i5) -> i1) produce (in %arg0 : i5, in %arg1 : i5) {
        %0 = comb.icmp eq %arg0, %arg1 : i5
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i8_cmpi_eq_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_eq_i8(%arg0: i8, %arg1: i8) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%arg0, %arg1 : i8, i8) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_eq_i8 : (i8, i8) -> i1) produce (in %arg0 : i8, in %arg1 : i8) {
        %0 = comb.icmp eq %arg0, %arg1 : i8
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i15_cmpi_eq_l0 latency<0>, incDelay<0.12968108842533266>, outDelay<0.12968108842533266> {
      oplib.target @arith_cmpi_eq_i15(%arg0: i15, %arg1: i15) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%arg0, %arg1 : i15, i15) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_eq_i15 : (i15, i15) -> i1) produce (in %arg0 : i15, in %arg1 : i15) {
        %0 = comb.icmp eq %arg0, %arg1 : i15
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i18_cmpi_eq_l0 latency<0>, incDelay<0.15753352143987398>, outDelay<0.15753352143987398> {
      oplib.target @arith_cmpi_eq_i18(%arg0: i18, %arg1: i18) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%arg0, %arg1 : i18, i18) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_eq_i18 : (i18, i18) -> i1) produce (in %arg0 : i18, in %arg1 : i18) {
        %0 = comb.icmp eq %arg0, %arg1 : i18
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i1_cmpi_ne_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_ne_i1(%arg0: i1, %arg1: i1) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 1 : i64}(%arg0, %arg1 : i1, i1) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_ne_i1 : (i1, i1) -> i1) produce (in %arg0 : i1, in %arg1 : i1) {
        %0 = comb.icmp ne %arg0, %arg1 : i1
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i7_cmpi_ne_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_ne_i7(%arg0: i7, %arg1: i7) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 1 : i64}(%arg0, %arg1 : i7, i7) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_ne_i7 : (i7, i7) -> i1) produce (in %arg0 : i7, in %arg1 : i7) {
        %0 = comb.icmp ne %arg0, %arg1 : i7
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i8_cmpi_ne_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_ne_i8(%arg0: i8, %arg1: i8) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 1 : i64}(%arg0, %arg1 : i8, i8) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_ne_i8 : (i8, i8) -> i1) produce (in %arg0 : i8, in %arg1 : i8) {
        %0 = comb.icmp ne %arg0, %arg1 : i8
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i6_cmpi_slt_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_slt_i6(%arg0: i6, %arg1: i6) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 2 : i64}(%arg0, %arg1 : i6, i6) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_slt_i6 : (i6, i6) -> i1) produce (in %arg0 : i6, in %arg1 : i6) {
        %0 = comb.icmp slt %arg0, %arg1 : i6
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i10_cmpi_sle_l0 latency<0>, incDelay<0.050118742153790752>, outDelay<0.050118742153790752> {
      oplib.target @arith_cmpi_sle_i10(%arg0: i10, %arg1: i10) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 3 : i64}(%arg0, %arg1 : i10, i10) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_sle_i10 : (i10, i10) -> i1) produce (in %arg0 : i10, in %arg1 : i10) {
        %0 = comb.icmp sle %arg0, %arg1 : i10
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i9_cmpi_sgt_l0 latency<0>, incDelay<5.000000e-02>, outDelay<5.000000e-02> {
      oplib.target @arith_cmpi_sgt_i9(%arg0: i9, %arg1: i9) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 4 : i64}(%arg0, %arg1 : i9, i9) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_sgt_i9 : (i9, i9) -> i1) produce (in %arg0 : i9, in %arg1 : i9) {
        %0 = comb.icmp sgt %arg0, %arg1 : i9
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i10_cmpi_sge_l0 latency<0>, incDelay<0.050118742153790752>, outDelay<0.050118742153790752> {
      oplib.target @arith_cmpi_sge_i10(%arg0: i10, %arg1: i10) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 5 : i64}(%arg0, %arg1 : i10, i10) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_sge_i10 : (i10, i10) -> i1) produce (in %arg0 : i10, in %arg1 : i10) {
        %0 = comb.icmp sge %arg0, %arg1 : i10
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @i26_cmpi_sge_l0 latency<0>, incDelay<0.18657046967264712>, outDelay<0.18657046967264712> {
      oplib.target @arith_cmpi_sge_i26(%arg0: i26, %arg1: i26) -> i1 {
        %0 = oplib.operation "arith.cmpi" with {predicate = 5 : i64}(%arg0, %arg1 : i26, i26) : i1
        oplib.output %0 : i1
      }
      oplib.hw_match(@arith_cmpi_sge_i26 : (i26, i26) -> i1) produce (in %arg0 : i26, in %arg1 : i26) {
        %0 = comb.icmp sge %arg0, %arg1 : i26
        oplib.hw_return %0 : i1
      }
    }
    oplib.operator @amcmem_bf16_add_bits_amc_d16_w16_l1 latency<1>, incDelay<1.000000e-01>, outDelay<0.50399999999999989> {
    }
  }
  loopschedule.func_sequential @bf16_add_bits_amc(%arg0: !amc.memory_ref<@mem0>, %arg1: !amc.memory_ref<@mem1>, %arg2: !amc.memory_ref<@mem2>) attributes {itypes = "uuu", oplib.library = @bf16_add_bits_amc_library, otypes = "", top} {
    %c7_i15 = arith.constant 7 : i15
    %c255_i10 = arith.constant 255 : i10
    %c17_i9 = arith.constant 17 : i9
    %c1_i15 = arith.constant 1 : i15
    %c1_i14 = arith.constant 1 : i14
    %c1_i13 = arith.constant 1 : i13
    %c1_i12 = arith.constant 1 : i12
    %c1_i11 = arith.constant 1 : i11
    %c1_i10 = arith.constant 1 : i10
    %c1_i7 = arith.constant 1 : i7
    %c1_i4 = arith.constant 1 : i4
    %c1_i3 = arith.constant 1 : i3
    %false = arith.constant false
    %c1_i2 = arith.constant 1 : i2
    %c0_i18 = arith.constant 0 : i18
    %c0_i15 = arith.constant 0 : i15
    %c32640_i16 = arith.constant 32640 : i16
    %c0_i7 = arith.constant 0 : i7
    %c-1_i8 = arith.constant -1 : i8
    %c10_i10 = arith.constant 10 : i10
    %c17_i25 = arith.constant 17 : i25
    %c1_i6 = arith.constant 1 : i6
    %c16_i6 = arith.constant 16 : i6
    %c0_i6 = arith.constant 0 : i6
    %c16_i17 = arith.constant 16 : i17
    %c0_i8 = arith.constant 0 : i8
    %c0_i16 = arith.constant 0 : i16
    %c32704_i16 = arith.constant 32704 : i16
    %c15_i16 = arith.constant 15 : i16
    %c1_i18 = arith.constant 1 : i18
    %c1_i9 = arith.constant 1 : i9
    %c-15_i5 = arith.constant -15 : i5
    %c1_i16 = arith.constant 1 : i16
    %c9_i16 = arith.constant 9 : i16
    %c5_i9 = arith.constant 5 : i9
    %c8_i18 = arith.constant 8 : i18
    %c9_i18 = arith.constant 9 : i18
    %c10_i18 = arith.constant 10 : i18
    %c11_i18 = arith.constant 11 : i18
    %c2_i18 = arith.constant 2 : i18
    %c1_i8 = arith.constant 1 : i8
    %c-6_i4 = arith.constant -6 : i4
    %c1_i5 = arith.constant 1 : i5
    %c-16_i5 = arith.constant -16 : i5
    %c0_i5 = arith.constant 0 : i5
    %c2_i5 = arith.constant 2 : i5
    %c3_i5 = arith.constant 3 : i5
    %c4_i5 = arith.constant 4 : i5
    %c5_i5 = arith.constant 5 : i5
    %c6_i5 = arith.constant 6 : i5
    %c7_i5 = arith.constant 7 : i5
    %c8_i5 = arith.constant 8 : i5
    %c9_i5 = arith.constant 9 : i5
    %c10_i5 = arith.constant 10 : i5
    %c11_i5 = arith.constant 11 : i5
    %c12_i5 = arith.constant 12 : i5
    %c13_i5 = arith.constant 13 : i5
    %c14_i5 = arith.constant 14 : i5
    %c1_i17 = arith.constant 1 : i17
    %c15_i5 = arith.constant 15 : i5
    %0 = amc.expand_ref(%arg2 : !amc.memory_ref<@mem2>) : !amc.port<16xi16, static rw(1, 1)>
    %1 = amc.expand_ref(%arg1 : !amc.memory_ref<@mem1>) : !amc.port<16xi16, static rw(1, 1)>
    %2 = amc.expand_ref(%arg0 : !amc.memory_ref<@mem0>) : !amc.port<16xi16, static rw(1, 1)>
    %3 = loopschedule.frame -> (!loopschedule.handle) {
      %4 = loopschedule.at 0 -> !loopschedule.handle {
        %5 = loopschedule.launch : !loopschedule.handle {
          %6 = loopschedule.pipeline II = 1 trip_count = 16 latency = 4 iter_args(%arg3 = %c0_i6) : (i6) -> i6 {
            %7:5 = loopschedule.at 0 -> (i6, i5, i16, i16, i1) {
              %11 = arith.cmpi slt, %arg3, %c16_i6 {loopschedule.operator = @i6_cmpi_slt_l0} : i6
              %12 = arith.addi %arg3, %c1_i6 {loopschedule.operator = @i6_addi_l0} : i6
              %13 = arith.trunci %arg3 : i6 to i5
              %14 = amc.load %2[%13 : i5] {loopschedule.operator = @amcmem_bf16_add_bits_amc_d16_w16_l1} : !amc.port<16xi16, static rw(1, 1)>
              %15 = amc.load %1[%13 : i5] {loopschedule.operator = @amcmem_bf16_add_bits_amc_d16_w16_l1} : !amc.port<16xi16, static rw(1, 1)>
              loopschedule.iter_arg_update %arg3 = %12 : i6
              loopschedule.yield %12, %13, %14, %15, %11 : i6, i5, i16, i16, i1
            }
            %8:39 = loopschedule.at 1 -> (i6, i5, i16, i16, i9, i1, i18, i1, i1, i1, i16, i16, i1, i1, i16, i1, i1, i18, i9, i1, i5, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i5, i6, i16, i16) {
              %11 = comb.extract %7#2 from 15 : (i16) -> i1
              %12 = comb.extract %7#3 from 15 : (i16) -> i1
              %13 = arith.shli %7#2, %c1_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %14 = arith.shrui %13, %c1_i16 {loopschedule.operator = @i16_shrui_l0} : i16
              %15 = comb.extract %14 from 7 : (i16) -> i8
              %16 = arith.shli %7#3, %c1_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %17 = arith.shrui %16, %c1_i16 {loopschedule.operator = @i16_shrui_l0} : i16
              %18 = comb.extract %17 from 7 : (i16) -> i8
              %19 = arith.shli %7#2, %c9_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %20 = comb.extract %19 from 9 : (i16) -> i7
              %21 = arith.shli %7#3, %c9_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %22 = comb.extract %21 from 9 : (i16) -> i7
              %23 = arith.cmpi ne, %15, %c0_i8 {loopschedule.operator = @i8_cmpi_ne_l0} : i8
              %24 = arith.extui %23 : i1 to i17
              %25 = arith.cmpi ne, %18, %c0_i8 {loopschedule.operator = @i8_cmpi_ne_l0} : i8
              %26 = arith.extui %25 : i1 to i17
              %27 = arith.shli %24, %c16_i17 {loopschedule.operator = @i17_shli_l0} : i17
              %28 = arith.extui %20 : i7 to i16
              %29 = arith.shli %28, %c9_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %30 = arith.extui %29 : i16 to i17
              %31 = arith.ori %27, %30 {loopschedule.operator = @i17_ori_l0} : i17
              %32 = arith.shli %26, %c16_i17 {loopschedule.operator = @i17_shli_l0} : i17
              %33 = arith.extui %22 : i7 to i16
              %34 = arith.shli %33, %c9_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %35 = arith.extui %34 : i16 to i17
              %36 = arith.ori %32, %35 {loopschedule.operator = @i17_ori_l0} : i17
              %37 = arith.cmpi eq, %11, %12 {loopschedule.operator = @i1_cmpi_eq_l0} : i1
              %38 = arith.extui %15 : i8 to i25
              %39 = arith.shli %38, %c17_i25 {loopschedule.operator = @i25_shli_l0} : i25
              %40 = arith.extui %31 : i17 to i25
              %41 = arith.ori %39, %40 {loopschedule.operator = @i25_ori_l0} : i25
              %42 = arith.extui %41 : i25 to i26
              %43 = arith.extui %18 : i8 to i25
              %44 = arith.shli %43, %c17_i25 {loopschedule.operator = @i25_shli_l0} : i25
              %45 = arith.extui %36 : i17 to i25
              %46 = arith.ori %44, %45 {loopschedule.operator = @i25_ori_l0} : i25
              %47 = arith.extui %46 : i25 to i26
              %48 = arith.cmpi sge, %42, %47 {loopschedule.operator = @i26_cmpi_sge_l0} : i26
              %49 = arith.cmpi eq, %15, %c0_i8 {loopschedule.operator = @i8_cmpi_eq_l0} : i8
              %50 = arith.extui %15 : i8 to i9
              %51 = arith.trunci %15 : i8 to i5
              %52 = arith.select %49, %c1_i5, %51 {loopschedule.operator = @i5_select_l0} : i5
              %53 = arith.select %49, %c1_i9, %50 {loopschedule.operator = @i9_select_l0} : i9
              %54 = arith.cmpi eq, %18, %c0_i8 {loopschedule.operator = @i8_cmpi_eq_l0} : i8
              %55 = arith.extui %18 : i8 to i9
              %56 = arith.trunci %18 : i8 to i5
              %57 = arith.select %54, %c1_i5, %56 {loopschedule.operator = @i5_select_l0} : i5
              %58 = arith.select %54, %c1_i9, %55 {loopschedule.operator = @i9_select_l0} : i9
              %59 = arith.select %48, %11, %12 {loopschedule.operator = @i1_select_l0} : i1
              %60 = arith.select %48, %52, %57 {loopschedule.operator = @i5_select_l0} : i5
              %61 = arith.select %48, %53, %58 {loopschedule.operator = @i9_select_l0} : i9
              %62 = arith.select %48, %58, %53 {loopschedule.operator = @i9_select_l0} : i9
              %63 = arith.select %48, %31, %36 {loopschedule.operator = @i17_select_l0} : i17
              %64 = arith.select %48, %36, %31 {loopschedule.operator = @i17_select_l0} : i17
              %65 = arith.subi %61, %62 {loopschedule.operator = @i9_subi_l0} : i9
              %66 = arith.extui %65 : i9 to i10
              %67 = arith.cmpi sge, %66, %c10_i10 {loopschedule.operator = @i10_cmpi_sge_l0} : i10
              %68 = arith.shli %65, %c5_i9 {loopschedule.operator = @i9_shli_l0} : i9
              %69 = comb.extract %68 from 5 : (i9) -> i4
              %70 = arith.select %67, %c-6_i4, %69 {loopschedule.operator = @i4_select_l0} : i4
              %71 = arith.extui %70 {unsigned} : i4 to i17
              %72 = arith.shrui %64, %71 {loopschedule.operator = @i17_shrui_l0} : i17
              %73 = arith.extui %63 {unsigned} : i17 to i18
              %74 = arith.extui %72 {unsigned} : i17 to i18
              %75 = arith.addi %73, %74 {loopschedule.operator = @i18_addi_l0} : i18
              %76 = arith.subi %73, %74 {loopschedule.operator = @i18_subi_l0} : i18
              %77 = arith.select %37, %75, %76 {loopschedule.operator = @i18_select_l0} : i18
              %78 = arith.extui %11 {unsigned} : i1 to i16
              %79 = arith.extui %12 {unsigned} : i1 to i16
              %80 = arith.extui %59 {unsigned} : i1 to i16
              %81 = arith.cmpi eq, %15, %c-1_i8 {loopschedule.operator = @i8_cmpi_eq_l0} : i8
              %82 = arith.cmpi ne, %20, %c0_i7 {loopschedule.operator = @i7_cmpi_ne_l0} : i7
              %83 = arith.andi %81, %82 {loopschedule.operator = @i1_andi_l0} : i1
              %84 = arith.cmpi eq, %18, %c-1_i8 {loopschedule.operator = @i8_cmpi_eq_l0} : i8
              %85 = arith.cmpi ne, %22, %c0_i7 {loopschedule.operator = @i7_cmpi_ne_l0} : i7
              %86 = arith.andi %84, %85 {loopschedule.operator = @i1_andi_l0} : i1
              %87 = arith.ori %83, %86 {loopschedule.operator = @i1_ori_l0} : i1
              %88 = arith.cmpi ne, %11, %12 {loopschedule.operator = @i1_cmpi_ne_l0} : i1
              %89 = arith.andi %84, %88 {loopschedule.operator = @i1_andi_l0} : i1
              %90 = arith.andi %81, %89 {loopschedule.operator = @i1_andi_l0} : i1
              %91 = arith.ori %87, %90 {loopschedule.operator = @i1_ori_l0} : i1
              %92 = arith.shli %78, %c15_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %93 = arith.ori %92, %c32640_i16 {loopschedule.operator = @i16_ori_l0} : i16
              %94 = arith.shli %79, %c15_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %95 = arith.ori %94, %c32640_i16 {loopschedule.operator = @i16_ori_l0} : i16
              %96 = comb.extract %13 from 1 : (i16) -> i15
              %97 = arith.cmpi eq, %96, %c0_i15 {loopschedule.operator = @i15_cmpi_eq_l0} : i15
              %98 = comb.extract %16 from 1 : (i16) -> i15
              %99 = arith.cmpi eq, %98, %c0_i15 {loopschedule.operator = @i15_cmpi_eq_l0} : i15
              %100 = arith.select %48, %7#2, %7#3 {loopschedule.operator = @i16_select_l0} : i16
              %101 = arith.cmpi eq, %77, %c0_i18 {loopschedule.operator = @i18_cmpi_eq_l0} : i18
              %102 = comb.extract %77 from 17 : (i18) -> i1
              %103 = arith.andi %77, %c1_i18 {loopschedule.operator = @i18_andi_l0} : i18
              %104 = arith.trunci %103 : i18 to i2
              %105 = arith.shli %104, %c1_i2 {loopschedule.operator = @i2_shli_l0} : i2
              %106 = arith.extui %105 : i2 to i18
              %107 = arith.ori %77, %106 {loopschedule.operator = @i18_ori_l0} : i18
              %108 = arith.shrui %107, %c1_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %109 = arith.addi %61, %c1_i9 {loopschedule.operator = @i9_addi_l0} : i9
              %110 = arith.shli %77, %c1_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %111 = arith.shrui %110, %c1_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %112 = comb.extract %111 from 16 : (i18) -> i1
              %113 = arith.cmpi eq, %112, %false {loopschedule.operator = @i1_cmpi_eq_l0} : i1
              %114 = comb.extract %110 from 1 : (i18) -> i17
              %115 = comb.extract %110 from 17 : (i18) -> i1
              %116 = arith.select %115, %c0_i5, %c-15_i5 {loopschedule.operator = @i5_select_l0} : i5
              %117 = arith.cmpi eq, %116, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %118 = comb.extract %110 from 16 : (i18) -> i2
              %119 = arith.andi %118, %c1_i2 {loopschedule.operator = @i2_andi_l0} : i2
              %120 = arith.trunci %119 : i2 to i1
              %121 = arith.andi %117, %120 {loopschedule.operator = @i1_andi_l0} : i1
              %122 = arith.select %121, %c1_i5, %116 {loopschedule.operator = @i5_select_l0} : i5
              %123 = arith.cmpi eq, %122, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %124 = comb.extract %110 from 15 : (i18) -> i3
              %125 = arith.andi %124, %c1_i3 {loopschedule.operator = @i3_andi_l0} : i3
              %126 = arith.trunci %125 : i3 to i1
              %127 = arith.andi %123, %126 {loopschedule.operator = @i1_andi_l0} : i1
              %128 = arith.select %127, %c2_i5, %122 {loopschedule.operator = @i5_select_l0} : i5
              %129 = arith.cmpi eq, %128, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %130 = comb.extract %110 from 14 : (i18) -> i4
              %131 = arith.andi %130, %c1_i4 {loopschedule.operator = @i4_andi_l0} : i4
              %132 = arith.trunci %131 : i4 to i1
              %133 = arith.andi %129, %132 {loopschedule.operator = @i1_andi_l0} : i1
              %134 = arith.select %133, %c3_i5, %128 {loopschedule.operator = @i5_select_l0} : i5
              %135 = arith.cmpi eq, %134, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %136 = comb.extract %110 from 13 : (i18) -> i5
              %137 = arith.andi %136, %c1_i5 {loopschedule.operator = @i5_andi_l0} : i5
              %138 = arith.trunci %137 : i5 to i1
              %139 = comb.extract %110 from 12 : (i18) -> i6
              %140 = arith.andi %139, %c1_i6 {loopschedule.operator = @i6_andi_l0} : i6
              %141 = arith.trunci %140 : i6 to i1
              %142 = comb.extract %110 from 11 : (i18) -> i7
              %143 = arith.andi %142, %c1_i7 {loopschedule.operator = @i7_andi_l0} : i7
              %144 = arith.trunci %143 : i7 to i1
              %145 = comb.extract %110 from 10 : (i18) -> i8
              %146 = arith.andi %145, %c1_i8 {loopschedule.operator = @i8_andi_l0} : i8
              %147 = arith.trunci %146 : i8 to i1
              %148 = comb.extract %110 from 9 : (i18) -> i9
              %149 = arith.andi %148, %c1_i9 {loopschedule.operator = @i9_andi_l0} : i9
              %150 = arith.trunci %149 : i9 to i1
              %151 = comb.extract %110 from 8 : (i18) -> i10
              %152 = arith.andi %151, %c1_i10 {loopschedule.operator = @i10_andi_l0} : i10
              %153 = arith.trunci %152 : i10 to i1
              %154 = comb.extract %110 from 7 : (i18) -> i11
              %155 = arith.andi %154, %c1_i11 {loopschedule.operator = @i11_andi_l0} : i11
              %156 = arith.trunci %155 : i11 to i1
              %157 = comb.extract %110 from 6 : (i18) -> i12
              %158 = arith.andi %157, %c1_i12 {loopschedule.operator = @i12_andi_l0} : i12
              %159 = arith.trunci %158 : i12 to i1
              %160 = comb.extract %110 from 5 : (i18) -> i13
              %161 = arith.andi %160, %c1_i13 {loopschedule.operator = @i13_andi_l0} : i13
              %162 = arith.trunci %161 : i13 to i1
              %163 = comb.extract %110 from 4 : (i18) -> i14
              %164 = arith.andi %163, %c1_i14 {loopschedule.operator = @i14_andi_l0} : i14
              %165 = arith.trunci %164 : i14 to i1
              %166 = comb.extract %110 from 3 : (i18) -> i15
              %167 = arith.andi %166, %c1_i15 {loopschedule.operator = @i15_andi_l0} : i15
              %168 = arith.trunci %167 : i15 to i1
              %169 = comb.extract %110 from 2 : (i18) -> i16
              %170 = arith.andi %169, %c1_i16 {loopschedule.operator = @i16_andi_l0} : i16
              %171 = arith.trunci %170 : i16 to i1
              %172 = arith.andi %114, %c1_i17 {loopschedule.operator = @i17_andi_l0} : i17
              %173 = arith.trunci %172 : i17 to i1
              %174 = arith.cmpi sgt, %61, %c17_i9 {loopschedule.operator = @i9_cmpi_sgt_l0} : i9
              %175 = arith.subi %60, %c1_i5 {loopschedule.operator = @i5_subi_l0} : i5
              %176 = arith.select %174, %c-16_i5, %175 {loopschedule.operator = @i5_select_l0} : i5
              %177 = arith.extui %176 {unsigned} : i5 to i6
              %178 = arith.shli %80, %c15_i16 {loopschedule.operator = @i16_shli_l0} : i16
              %179 = arith.ori %178, %c32640_i16 {loopschedule.operator = @i16_ori_l0} : i16
              loopschedule.yield %7#0, %7#1, %7#2, %7#3, %61, %67, %77, %81, %84, %91, %93, %95, %97, %99, %100, %101, %102, %108, %109, %113, %134, %135, %138, %141, %144, %147, %150, %153, %156, %159, %162, %165, %168, %171, %173, %176, %177, %178, %179 : i6, i5, i16, i16, i9, i1, i18, i1, i1, i1, i16, i16, i1, i1, i16, i1, i1, i18, i9, i1, i5, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i5, i6, i16, i16
            }
            %9:21 = loopschedule.at 2 -> (i6, i5, i16, i16, i1, i1, i1, i1, i16, i16, i1, i1, i16, i1, i16, i16, i9, i8, i1, i9, i1) {
              %11 = arith.andi %8#21, %8#22 {loopschedule.operator = @i1_andi_l0} : i1
              %12 = arith.select %11, %c4_i5, %8#20 {loopschedule.operator = @i5_select_l0} : i5
              %13 = arith.cmpi eq, %12, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %14 = arith.andi %13, %8#23 {loopschedule.operator = @i1_andi_l0} : i1
              %15 = arith.select %14, %c5_i5, %12 {loopschedule.operator = @i5_select_l0} : i5
              %16 = arith.cmpi eq, %15, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %17 = arith.andi %16, %8#24 {loopschedule.operator = @i1_andi_l0} : i1
              %18 = arith.select %17, %c6_i5, %15 {loopschedule.operator = @i5_select_l0} : i5
              %19 = arith.cmpi eq, %18, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %20 = arith.andi %19, %8#25 {loopschedule.operator = @i1_andi_l0} : i1
              %21 = arith.select %20, %c7_i5, %18 {loopschedule.operator = @i5_select_l0} : i5
              %22 = arith.cmpi eq, %21, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %23 = arith.andi %22, %8#26 {loopschedule.operator = @i1_andi_l0} : i1
              %24 = arith.select %23, %c8_i5, %21 {loopschedule.operator = @i5_select_l0} : i5
              %25 = arith.cmpi eq, %24, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %26 = arith.andi %25, %8#27 {loopschedule.operator = @i1_andi_l0} : i1
              %27 = arith.select %26, %c9_i5, %24 {loopschedule.operator = @i5_select_l0} : i5
              %28 = arith.cmpi eq, %27, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %29 = arith.andi %28, %8#28 {loopschedule.operator = @i1_andi_l0} : i1
              %30 = arith.select %29, %c10_i5, %27 {loopschedule.operator = @i5_select_l0} : i5
              %31 = arith.cmpi eq, %30, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %32 = arith.andi %31, %8#29 {loopschedule.operator = @i1_andi_l0} : i1
              %33 = arith.select %32, %c11_i5, %30 {loopschedule.operator = @i5_select_l0} : i5
              %34 = arith.cmpi eq, %33, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %35 = arith.andi %34, %8#30 {loopschedule.operator = @i1_andi_l0} : i1
              %36 = arith.select %35, %c12_i5, %33 {loopschedule.operator = @i5_select_l0} : i5
              %37 = arith.cmpi eq, %36, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %38 = arith.andi %37, %8#31 {loopschedule.operator = @i1_andi_l0} : i1
              %39 = arith.select %38, %c13_i5, %36 {loopschedule.operator = @i5_select_l0} : i5
              %40 = arith.cmpi eq, %39, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %41 = arith.andi %40, %8#32 {loopschedule.operator = @i1_andi_l0} : i1
              %42 = arith.select %41, %c14_i5, %39 {loopschedule.operator = @i5_select_l0} : i5
              %43 = arith.cmpi eq, %42, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %44 = arith.andi %43, %8#33 {loopschedule.operator = @i1_andi_l0} : i1
              %45 = arith.select %44, %c15_i5, %42 {loopschedule.operator = @i5_select_l0} : i5
              %46 = arith.cmpi eq, %45, %c-15_i5 {loopschedule.operator = @i5_cmpi_eq_l0} : i5
              %47 = arith.andi %46, %8#34 {loopschedule.operator = @i1_andi_l0} : i1
              %48 = arith.select %47, %c-16_i5, %45 {loopschedule.operator = @i5_select_l0} : i5
              %49 = arith.extui %48 {unsigned} : i5 to i6
              %50 = arith.cmpi slt, %49, %8#36 {loopschedule.operator = @i6_cmpi_slt_l0} : i6
              %51 = arith.select %50, %48, %8#35 {loopschedule.operator = @i5_select_l0} : i5
              %52 = arith.extui %51 {unsigned} : i5 to i18
              %53 = arith.shli %8#6, %52 {loopschedule.operator = @i18_shli_l0} : i18
              %54 = arith.extui %51 {unsigned} : i5 to i9
              %55 = arith.subi %8#4, %54 {loopschedule.operator = @i9_subi_l0} : i9
              %56 = arith.select %8#19, %55, %8#4 {loopschedule.operator = @i9_select_l0} : i9
              %57 = arith.select %8#19, %53, %8#6 {loopschedule.operator = @i18_select_l0} : i18
              %58 = arith.select %8#16, %8#18, %56 {loopschedule.operator = @i9_select_l0} : i9
              %59 = arith.select %8#16, %8#17, %57 {loopschedule.operator = @i18_select_l0} : i18
              %60 = arith.shli %59, %c9_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %61 = arith.shrui %60, %c9_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %62 = comb.extract %61 from 8 : (i18) -> i1
              %63 = arith.shli %59, %c10_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %64 = arith.shrui %63, %c10_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %65 = comb.extract %64 from 7 : (i18) -> i1
              %66 = arith.shli %59, %c11_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %67 = comb.extract %66 from 11 : (i18) -> i7
              %68 = arith.cmpi ne, %67, %c0_i7 {loopschedule.operator = @i7_cmpi_ne_l0} : i7
              %69 = arith.ori %65, %68 {loopschedule.operator = @i1_ori_l0} : i1
              %70 = arith.shli %59, %c8_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %71 = arith.shrui %70, %c8_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %72 = comb.extract %71 from 9 : (i18) -> i1
              %73 = arith.ori %69, %72 {loopschedule.operator = @i1_ori_l0} : i1
              %74 = arith.andi %62, %73 {loopschedule.operator = @i1_andi_l0} : i1
              %75 = arith.shli %59, %c2_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %76 = arith.shrui %75, %c2_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %77 = comb.extract %76 from 9 : (i18) -> i7
              %78 = arith.extui %77 {unsigned} : i7 to i8
              %79 = arith.extui %74 {unsigned} : i1 to i8
              %80 = arith.addi %78, %79 {loopschedule.operator = @i8_addi_l0} : i8
              %81 = comb.extract %80 from 7 : (i8) -> i1
              %82 = arith.addi %58, %c1_i9 {loopschedule.operator = @i9_addi_l0} : i9
              %83 = arith.shli %59, %c1_i18 {loopschedule.operator = @i18_shli_l0} : i18
              %84 = arith.shrui %83, %c1_i18 {loopschedule.operator = @i18_shrui_l0} : i18
              %85 = comb.extract %84 from 16 : (i18) -> i1
              %86 = arith.cmpi eq, %85, %false {loopschedule.operator = @i1_cmpi_eq_l0} : i1
              loopschedule.yield %8#0, %8#1, %8#2, %8#3, %8#5, %8#7, %8#8, %8#9, %8#10, %8#11, %8#12, %8#13, %8#14, %8#15, %8#37, %8#38, %58, %80, %81, %82, %86 : i6, i5, i16, i16, i1, i1, i1, i1, i16, i16, i1, i1, i16, i1, i16, i16, i9, i8, i1, i9, i1
            }
            %10 = loopschedule.at 3 -> i6 {
              %11 = arith.select %9#18, %9#19, %9#16 {loopschedule.operator = @i9_select_l0} : i9
              %12 = arith.select %9#18, %c0_i8, %9#17 {loopschedule.operator = @i8_select_l0} : i8
              %13 = arith.extui %11 : i9 to i10
              %14 = arith.cmpi sge, %13, %c255_i10 {loopschedule.operator = @i10_cmpi_sge_l0} : i10
              %15 = arith.cmpi sle, %13, %c1_i10 {loopschedule.operator = @i10_cmpi_sle_l0} : i10
              %16 = arith.andi %15, %9#20 {loopschedule.operator = @i1_andi_l0} : i1
              %17 = arith.shli %12, %c1_i8 {loopschedule.operator = @i8_shli_l0} : i8
              %18 = comb.extract %17 from 1 : (i8) -> i7
              %19 = arith.extui %18 {unsigned} : i7 to i16
              %20 = arith.ori %9#14, %19 {loopschedule.operator = @i16_ori_l0} : i16
              %21 = arith.shli %11, %c1_i9 {loopschedule.operator = @i9_shli_l0} : i9
              %22 = comb.extract %21 from 1 : (i9) -> i8
              %23 = arith.extui %22 : i8 to i15
              %24 = arith.shli %23, %c7_i15 {loopschedule.operator = @i15_shli_l0} : i15
              %25 = arith.extui %24 : i15 to i16
              %26 = arith.ori %9#14, %25 {loopschedule.operator = @i16_ori_l0} : i16
              %27 = arith.ori %26, %19 {loopschedule.operator = @i16_ori_l0} : i16
              %28 = arith.select %16, %20, %27 {loopschedule.operator = @i16_select_l0} : i16
              %29 = arith.select %14, %9#15, %28 {loopschedule.operator = @i16_select_l0} : i16
              %30 = arith.select %9#13, %c0_i16, %29 {loopschedule.operator = @i16_select_l0} : i16
              %31 = arith.select %9#4, %9#12, %30 {loopschedule.operator = @i16_select_l0} : i16
              %32 = arith.select %9#11, %9#2, %31 {loopschedule.operator = @i16_select_l0} : i16
              %33 = arith.select %9#10, %9#3, %32 {loopschedule.operator = @i16_select_l0} : i16
              %34 = arith.select %9#6, %9#9, %33 {loopschedule.operator = @i16_select_l0} : i16
              %35 = arith.select %9#5, %9#8, %34 {loopschedule.operator = @i16_select_l0} : i16
              %36 = arith.select %9#7, %c32704_i16, %35 {loopschedule.operator = @i16_select_l0} : i16
              amc.store %36, %0[%9#1 : i5] {loopschedule.operator = @amcmem_bf16_add_bits_amc_d16_w16_l1} : !amc.port<16xi16, static rw(1, 1)>
              loopschedule.yield %9#0 : i6
            }
            loopschedule.terminator condition(%7#4), results(%7#0) : i6
          }
          loopschedule.yield %6 : i6
        }
        loopschedule.yield %5 : !loopschedule.handle
      }
      loopschedule.yield %4 : !loopschedule.handle
    }
    loopschedule.frame await {
      loopschedule.await %3
    }
    loopschedule.return
  }
}
