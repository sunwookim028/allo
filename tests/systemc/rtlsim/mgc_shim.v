// Behavioral stand-in for the one Catapult library cell the archived netlists use.
// mgc_shift_r_v5: z = a >> s (logical when signd_a==0, arithmetic otherwise).
module mgc_shift_r_v5 (a, s, z);
  parameter integer width_a = 8;
  parameter integer signd_a = 0;
  parameter integer width_s = 8;
  parameter integer signd_s = 0;
  parameter integer width_z = 8;
  input  [width_a-1:0] a;
  input  [width_s-1:0] s;
  output [width_z-1:0] z;
  wire signed [width_a-1:0] a_s = a;
  assign z = signd_a ? (a_s >>> s) : (a >> s);
endmodule
