// CCORE probe: Allo's emitted add_bits (vpu_bf16_add, `bits`) as a Catapult
// combinational CCORE top. The body below is copied verbatim from Allo's
// SystemC emission of examples/minitpu/units/bf16_add.py::bits (unrolled LZC).
#include <ac_int.h>
#include <stdint.h>
#include <algorithm>
// The emitter's synthesis-side ap_int shim (kernel.cpp of the SystemC flow).
template <int W> using ap_int = ac_int<W, true>;
template <int W> using ap_uint = ac_int<W, false>;
using std::max;
using std::min;
void leading_zeros17(
  ac_int<17, false> v0,
  ac_int<5, false> *v1
) {	// L2
  ac_int<5, false> lz;	// L7
  lz = -15;	// L8
  bool found;	// L9
  found = 0;	// L10
  #pragma hls_unroll
  l_S_offset_0_offset: for (int offset = 0; offset < 17; offset++) {	// L11
    bool v2 = found;	// L12
    bool v3 = v2 == 0;	// L13
    ac_int<34, true> v4 = offset;	// L14
    ac_int<34, true> v5 = 16 - v4;	// L15
    int v6 = v5;	// L16
    bool v7;
    ac_int<17, true> _bs_v7 = v0;
    v7 = _bs_v7[v6];	// L17
    bool v8 = v3 & v7;	// L18
    if (v8) {	// L19
      ac_int<5, false> v9 = offset;	// L20
      lz = v9;	// L21
      found = 1;	// L22
    }
  }
  *v1 = lz;	// L25
}

void add_bits(
  uint16_t v10,
  uint16_t v11,
  uint16_t *v12
) {	// L28
  bool v13;
  ac_int<16, true> _bs_v13 = v10;
  v13 = _bs_v13[15];	// L60
  bool sign_a;	// L61
  sign_a = v13;	// L62
  bool v14;
  ac_int<16, true> _bs_v14 = v11;
  v14 = _bs_v14[15];	// L63
  bool sign_b;	// L64
  sign_b = v14;	// L65
  uint8_t v15;
  ac_int<16, true> _bs_v15 = v10;
  v15 = _bs_v15.slc<8>(7);	// L66
  uint8_t exp_a;	// L67
  exp_a = v15;	// L68
  uint8_t v16;
  ac_int<16, true> _bs_v16 = v11;
  v16 = _bs_v16.slc<8>(7);	// L69
  uint8_t exp_b;	// L70
  exp_b = v16;	// L71
  ac_int<7, false> v17;
  ac_int<16, true> _bs_v17 = v10;
  v17 = _bs_v17.slc<7>(0);	// L72
  ac_int<7, false> frac_a;	// L73
  frac_a = v17;	// L74
  ac_int<7, false> v18;
  ac_int<16, true> _bs_v18 = v11;
  v18 = _bs_v18.slc<7>(0);	// L75
  ac_int<7, false> frac_b;	// L76
  frac_b = v18;	// L77
  ac_int<17, false> mant_a;	// L78
  mant_a = 0;	// L79
  uint8_t v19 = exp_a;	// L80
  int32_t v20 = v19;	// L81
  bool v21 = v20 != 0;	// L82
  ac_int<17, false> v22 = mant_a;	// L83
  ac_int<17, true> v23;
  ac_int<17, true> _bs_v23 = v22;
  _bs_v23[16] = v21;
  v23 = _bs_v23;	// L84
  mant_a = v23;	// L85
  ac_int<7, false> v24 = frac_a;	// L86
  ac_int<17, false> v25 = mant_a;	// L87
  ac_int<17, true> v26;
  ac_int<17, true> _bs_v26 = v25;
  _bs_v26.set_slc(9, ac_int<7, false>(v24));
  v26 = _bs_v26;	// L88
  mant_a = v26;	// L89
  ac_int<17, false> mant_b;	// L90
  mant_b = 0;	// L91
  uint8_t v27 = exp_b;	// L92
  int32_t v28 = v27;	// L93
  bool v29 = v28 != 0;	// L94
  ac_int<17, false> v30 = mant_b;	// L95
  ac_int<17, true> v31;
  ac_int<17, true> _bs_v31 = v30;
  _bs_v31[16] = v29;
  v31 = _bs_v31;	// L96
  mant_b = v31;	// L97
  ac_int<7, false> v32 = frac_b;	// L98
  ac_int<17, false> v33 = mant_b;	// L99
  ac_int<17, true> v34;
  ac_int<17, true> _bs_v34 = v33;
  _bs_v34.set_slc(9, ac_int<7, false>(v32));
  v34 = _bs_v34;	// L100
  mant_b = v34;	// L101
  bool v35 = sign_a;	// L102
  bool v36 = sign_b;	// L103
  bool v37 = v35 == v36;	// L104
  bool same_sign;	// L105
  same_sign = v37;	// L106
  bool sign_large;	// L107
  sign_large = 0;	// L108
  ac_int<9, false> exp_large;	// L109
  exp_large = 0;	// L110
  ac_int<9, false> exp_small;	// L111
  exp_small = 0;	// L112
  ac_int<9, false> exp_result;	// L113
  exp_result = 0;	// L114
  ac_int<9, false> exp_diff;	// L115
  exp_diff = 0;	// L116
  ac_int<17, false> mant_large;	// L117
  mant_large = 0;	// L118
  ac_int<17, false> mant_small;	// L119
  mant_small = 0;	// L120
  ac_int<17, false> small_aligned;	// L121
  small_aligned = 0;	// L122
  ac_int<18, false> magnitude;	// L123
  magnitude = 0;	// L124
  ac_int<4, false> align_shift;	// L125
  align_shift = 0;	// L126
  ac_int<5, false> leading_zeros;	// L127
  leading_zeros = 0;	// L128
  ac_int<5, false> normalize_shift;	// L129
  normalize_shift = 0;	// L130
  ac_int<5, false> max_normalize_shift;	// L131
  max_normalize_shift = 0;	// L132
  bool guard_bit;	// L133
  guard_bit = 0;	// L134
  bool round_bit;	// L135
  round_bit = 0;	// L136
  bool sticky_bit;	// L137
  sticky_bit = 0;	// L138
  bool round_up;	// L139
  round_up = 0;	// L140
  uint8_t rounded;	// L141
  rounded = 0;	// L142
  uint16_t result_o;	// L143
  result_o = 0;	// L144
  ac_int<26, false> key_a;	// L145
  key_a = 0;	// L146
  ac_int<17, false> v38 = mant_a;	// L147
  ac_int<26, false> v39 = key_a;	// L148
  ac_int<26, true> v40;
  ac_int<26, true> _bs_v40 = v39;
  _bs_v40.set_slc(0, ac_int<17, false>(v38));
  v40 = _bs_v40;	// L149
  key_a = v40;	// L150
  uint8_t v41 = exp_a;	// L151
  ac_int<26, false> v42 = key_a;	// L152
  ac_int<26, true> v43;
  ac_int<26, true> _bs_v43 = v42;
  _bs_v43.set_slc(17, ac_int<8, false>(v41));
  v43 = _bs_v43;	// L153
  key_a = v43;	// L154
  ac_int<26, false> key_b;	// L155
  key_b = 0;	// L156
  ac_int<17, false> v44 = mant_b;	// L157
  ac_int<26, false> v45 = key_b;	// L158
  ac_int<26, true> v46;
  ac_int<26, true> _bs_v46 = v45;
  _bs_v46.set_slc(0, ac_int<17, false>(v44));
  v46 = _bs_v46;	// L159
  key_b = v46;	// L160
  uint8_t v47 = exp_b;	// L161
  ac_int<26, false> v48 = key_b;	// L162
  ac_int<26, true> v49;
  ac_int<26, true> _bs_v49 = v48;
  _bs_v49.set_slc(17, ac_int<8, false>(v47));
  v49 = _bs_v49;	// L163
  key_b = v49;	// L164
  ac_int<26, false> v50 = key_a;	// L165
  ac_int<26, false> v51 = key_b;	// L166
  bool v52 = v50 >= v51;	// L167
  bool a_is_large;	// L168
  a_is_large = v52;	// L169
  bool v53 = a_is_large;	// L170
  if (v53) {	// L171
    bool v54 = sign_a;	// L172
    sign_large = v54;	// L173
    uint8_t v55 = exp_a;	// L174
    int32_t v56 = v55;	// L175
    bool v57 = v56 == 0;	// L176
    int32_t v58 = v57 ? (int32_t)1 : (int32_t)v56;	// L177
    ac_int<9, false> v59 = v58;	// L178
    exp_large = v59;	// L179
    uint8_t v60 = exp_b;	// L180
    int32_t v61 = v60;	// L181
    bool v62 = v61 == 0;	// L182
    int32_t v63 = v62 ? (int32_t)1 : (int32_t)v61;	// L183
    ac_int<9, false> v64 = v63;	// L184
    exp_small = v64;	// L185
    ac_int<17, false> v65 = mant_a;	// L186
    mant_large = v65;	// L187
    ac_int<17, false> v66 = mant_b;	// L188
    mant_small = v66;	// L189
  } else {
    bool v67 = sign_b;	// L191
    sign_large = v67;	// L192
    uint8_t v68 = exp_b;	// L193
    int32_t v69 = v68;	// L194
    bool v70 = v69 == 0;	// L195
    int32_t v71 = v70 ? (int32_t)1 : (int32_t)v69;	// L196
    ac_int<9, false> v72 = v71;	// L197
    exp_large = v72;	// L198
    uint8_t v73 = exp_a;	// L199
    int32_t v74 = v73;	// L200
    bool v75 = v74 == 0;	// L201
    int32_t v76 = v75 ? (int32_t)1 : (int32_t)v74;	// L202
    ac_int<9, false> v77 = v76;	// L203
    exp_small = v77;	// L204
    ac_int<17, false> v78 = mant_b;	// L205
    mant_large = v78;	// L206
    ac_int<17, false> v79 = mant_a;	// L207
    mant_small = v79;	// L208
  }
  ac_int<9, false> v80 = exp_large;	// L210
  ac_int<9, false> v81 = exp_small;	// L211
  ac_int<10, false> v82 = v80;	// L212
  ac_int<10, false> v83 = v81;	// L213
  ac_int<10, false> v84 = v82 - v83;	// L214
  ac_int<9, false> v85 = v84;	// L215
  exp_diff = v85;	// L216
  ac_int<9, false> v86 = exp_diff;	// L217
  int32_t v87 = v86;	// L218
  bool v88 = v87 >= 10;	// L219
  ac_int<4, false> v89;
  ac_int<9, true> _bs_v89 = v86;
  v89 = _bs_v89.slc<4>(0);	// L220
  int32_t v90 = v89;	// L221
  int32_t v91 = v88 ? (int32_t)10 : (int32_t)v90;	// L222
  ac_int<4, false> v92 = v91;	// L223
  align_shift = v92;	// L224
  ac_int<17, false> v93 = mant_small;	// L225
  ac_int<4, false> v94 = align_shift;	// L226
  ac_int<17, false> v95 = v94;	// L227
  ac_int<17, false> v96 = v93 >> v95;	// L228
  small_aligned = v96;	// L229
  bool v97 = same_sign;	// L230
  ac_int<17, false> v98 = mant_large;	// L231
  ac_int<17, false> v99 = small_aligned;	// L232
  ac_int<18, false> v100 = v98;	// L233
  ac_int<18, false> v101 = v99;	// L234
  ac_int<18, false> v102 = v100 + v101;	// L235
  ac_int<18, false> v103 = v100 - v101;	// L236
  ac_int<18, true> v104 = v97 ? (ap_uint<18>)v102 : (ap_uint<18>)v103;	// L237
  magnitude = v104;	// L238
  ac_int<9, false> v105 = exp_large;	// L239
  exp_result = v105;	// L240
  uint8_t v106 = exp_a;	// L241
  int32_t v107 = v106;	// L242
  bool v108 = v107 == 255;	// L243
  ac_int<7, false> v109 = frac_a;	// L244
  int32_t v110 = v109;	// L245
  bool v111 = v110 != 0;	// L246
  bool v112 = v108 & v111;	// L247
  uint8_t v113 = exp_b;	// L248
  int32_t v114 = v113;	// L249
  bool v115 = v114 == 255;	// L250
  ac_int<7, false> v116 = frac_b;	// L251
  int32_t v117 = v116;	// L252
  bool v118 = v117 != 0;	// L253
  bool v119 = v115 & v118;	// L254
  bool v120 = sign_a;	// L255
  bool v121 = sign_b;	// L256
  bool v122 = v120 != v121;	// L257
  bool v123 = v108 & v115;	// L258
  bool v124 = v123 & v122;	// L259
  bool v125 = v112 | v119;	// L260
  bool v126 = v125 | v124;	// L261
  if (v126) {	// L262
    result_o = 32704;	// L263
  } else {
    uint8_t v127 = exp_a;	// L265
    int32_t v128 = v127;	// L266
    bool v129 = v128 == 255;	// L267
    if (v129) {	// L268
      bool v130 = sign_a;	// L269
      uint16_t v131 = result_o;	// L270
      int16_t v132;
      ac_int<16, true> _bs_v132 = v131;
      _bs_v132[15] = v130;
      v132 = _bs_v132;	// L271
      result_o = v132;	// L272
      uint16_t v133 = result_o;	// L273
      int16_t v134;
      ac_int<16, true> _bs_v134 = v133;
      _bs_v134.set_slc(7, ac_int<8, false>(-1));
      v134 = _bs_v134;	// L274
      result_o = v134;	// L275
    } else {
      uint8_t v135 = exp_b;	// L277
      int32_t v136 = v135;	// L278
      bool v137 = v136 == 255;	// L279
      if (v137) {	// L280
        bool v138 = sign_b;	// L281
        uint16_t v139 = result_o;	// L282
        int16_t v140;
        ac_int<16, true> _bs_v140 = v139;
        _bs_v140[15] = v138;
        v140 = _bs_v140;	// L283
        result_o = v140;	// L284
        uint16_t v141 = result_o;	// L285
        int16_t v142;
        ac_int<16, true> _bs_v142 = v141;
        _bs_v142.set_slc(7, ac_int<8, false>(-1));
        v142 = _bs_v142;	// L286
        result_o = v142;	// L287
      } else {
        ac_int<15, false> v143;
        ac_int<16, true> _bs_v143 = v10;
        v143 = _bs_v143.slc<15>(0);	// L289
        int32_t v144 = v143;	// L290
        bool v145 = v144 == 0;	// L291
        if (v145) {	// L292
          result_o = v11;	// L293
        } else {
          ac_int<15, false> v146;
          ac_int<16, true> _bs_v146 = v11;
          v146 = _bs_v146.slc<15>(0);	// L295
          int32_t v147 = v146;	// L296
          bool v148 = v147 == 0;	// L297
          if (v148) {	// L298
            result_o = v10;	// L299
          } else {
            ac_int<9, false> v149 = exp_diff;	// L301
            int32_t v150 = v149;	// L302
            bool v151 = v150 >= 10;	// L303
            if (v151) {	// L304
              bool v152 = a_is_large;	// L305
              int16_t v153 = v152 ? (uint16_t)v10 : (uint16_t)v11;	// L306
              result_o = v153;	// L307
            } else {
              ac_int<18, false> v154 = magnitude;	// L309
              int32_t v155 = v154;	// L310
              bool v156 = v155 == 0;	// L311
              if (v156) {	// L312
                result_o = 0;	// L313
              } else {
                ac_int<18, false> v157 = magnitude;	// L315
                bool v158;
                ac_int<18, true> _bs_v158 = v157;
                v158 = _bs_v158[17];	// L316
                if (v158) {	// L317
                  ac_int<18, false> v159 = magnitude;	// L318
                  bool v160;
                  ac_int<18, true> _bs_v160 = v159;
                  v160 = _bs_v160[1];	// L319
                  ac_int<18, false> v161 = magnitude;	// L320
                  bool v162;
                  ac_int<18, true> _bs_v162 = v161;
                  v162 = _bs_v162[0];	// L321
                  bool v163 = v160 | v162;	// L322
                  ac_int<18, false> v164 = magnitude;	// L323
                  ac_int<18, true> v165;
                  ac_int<18, true> _bs_v165 = v164;
                  _bs_v165[1] = v163;
                  v165 = _bs_v165;	// L324
                  magnitude = v165;	// L325
                  ac_int<18, false> v166 = magnitude;	// L326
                  ac_int<18, false> v167 = v166 >> 1;	// L327
                  magnitude = v167;	// L328
                  ac_int<9, false> v168 = exp_result;	// L329
                  ac_int<33, true> v169 = v168;	// L330
                  ac_int<33, true> v170 = v169 + 1;	// L331
                  ac_int<9, false> v171 = v170;	// L332
                  exp_result = v171;	// L333
                } else {
                  ac_int<18, false> v172 = magnitude;	// L335
                  bool v173;
                  ac_int<18, true> _bs_v173 = v172;
                  v173 = _bs_v173[16];	// L336
                  bool v174 = v173 == 0;	// L337
                  if (v174) {	// L338
                    ac_int<18, false> v175 = magnitude;	// L339
                    ac_int<17, false> v176;
                    ac_int<18, true> _bs_v176 = v175;
                    v176 = _bs_v176.slc<17>(0);	// L340
                    ac_int<5, false> v177;
                    leading_zeros17(v176, &v177);	// L341
                    leading_zeros = v177;	// L342
                    ac_int<9, false> v178 = exp_result;	// L343
                    int32_t v179 = v178;	// L344
                    bool v180 = v179 > 17;	// L345
                    ac_int<33, true> v181 = v178;	// L346
                    ac_int<33, true> v182 = v181 - 1;	// L347
                    ac_int<33, true> v183 = v180 ? (ap_int<33>)16 : (ap_int<33>)v182;	// L348
                    ac_int<5, false> v184 = v183;	// L349
                    max_normalize_shift = v184;	// L350
                    ac_int<5, false> v185 = leading_zeros;	// L351
                    ac_int<6, false> v186 = v185;	// L352
                    ac_int<6, false> lz6;	// L353
                    lz6 = v186;	// L354
                    ac_int<5, false> v187 = max_normalize_shift;	// L355
                    ac_int<6, false> v188 = v187;	// L356
                    ac_int<6, false> max6;	// L357
                    max6 = v188;	// L358
                    ac_int<6, false> v189 = lz6;	// L359
                    ac_int<6, false> v190 = max6;	// L360
                    bool v191 = v189 < v190;	// L361
                    ac_int<5, false> v192 = leading_zeros;	// L362
                    ac_int<5, false> v193 = max_normalize_shift;	// L363
                    ac_int<5, true> v194 = v191 ? (ap_uint<5>)v192 : (ap_uint<5>)v193;	// L364
                    normalize_shift = v194;	// L365
                    ac_int<5, false> v195 = normalize_shift;	// L366
                    ac_int<18, false> v196 = magnitude;	// L367
                    ac_int<18, false> v197 = v195;	// L368
                    ac_int<18, false> v198 = v196 << v197;	// L369
                    magnitude = v198;	// L370
                    ac_int<5, false> v199 = normalize_shift;	// L371
                    ac_int<9, false> v200 = exp_result;	// L372
                    ac_int<10, false> v201 = v200;	// L373
                    ac_int<10, false> v202 = v199;	// L374
                    ac_int<10, false> v203 = v201 - v202;	// L375
                    ac_int<9, false> v204 = v203;	// L376
                    exp_result = v204;	// L377
                  }
                }
                ac_int<18, false> v205 = magnitude;	// L380
                bool v206;
                ac_int<18, true> _bs_v206 = v205;
                v206 = _bs_v206[8];	// L381
                guard_bit = v206;	// L382
                ac_int<18, false> v207 = magnitude;	// L383
                bool v208;
                ac_int<18, true> _bs_v208 = v207;
                v208 = _bs_v208[7];	// L384
                round_bit = v208;	// L385
                ac_int<18, false> v209 = magnitude;	// L386
                ac_int<7, false> v210;
                ac_int<18, true> _bs_v210 = v209;
                v210 = _bs_v210.slc<7>(0);	// L387
                int32_t v211 = v210;	// L388
                bool v212 = v211 != 0;	// L389
                sticky_bit = v212;	// L390
                bool v213 = guard_bit;	// L391
                bool v214 = round_bit;	// L392
                bool v215 = sticky_bit;	// L393
                bool v216 = v214 | v215;	// L394
                ac_int<18, false> v217 = magnitude;	// L395
                bool v218;
                ac_int<18, true> _bs_v218 = v217;
                v218 = _bs_v218[9];	// L396
                bool v219 = v216 | v218;	// L397
                bool v220 = v213 & v219;	// L398
                round_up = v220;	// L399
                ac_int<18, false> v221 = magnitude;	// L400
                ac_int<7, false> v222;
                ac_int<18, true> _bs_v222 = v221;
                v222 = _bs_v222.slc<7>(9);	// L401
                bool v223 = round_up;	// L402
                uint8_t v224 = v222;	// L403
                uint8_t v225 = v223;	// L404
                uint8_t v226 = v224 + v225;	// L405
                rounded = v226;	// L406
                uint8_t v227 = rounded;	// L407
                bool v228;
                ac_int<8, true> _bs_v228 = v227;
                v228 = _bs_v228[7];	// L408
                if (v228) {	// L409
                  rounded = 0;	// L410
                  ac_int<9, false> v229 = exp_result;	// L411
                  ac_int<33, true> v230 = v229;	// L412
                  ac_int<33, true> v231 = v230 + 1;	// L413
                  ac_int<9, false> v232 = v231;	// L414
                  exp_result = v232;	// L415
                }
                ac_int<9, false> v233 = exp_result;	// L417
                int32_t v234 = v233;	// L418
                bool v235 = v234 >= 255;	// L419
                if (v235) {	// L420
                  bool v236 = sign_large;	// L421
                  uint16_t v237 = result_o;	// L422
                  int16_t v238;
                  ac_int<16, true> _bs_v238 = v237;
                  _bs_v238[15] = v236;
                  v238 = _bs_v238;	// L423
                  result_o = v238;	// L424
                  uint16_t v239 = result_o;	// L425
                  int16_t v240;
                  ac_int<16, true> _bs_v240 = v239;
                  _bs_v240.set_slc(7, ac_int<8, false>(-1));
                  v240 = _bs_v240;	// L426
                  result_o = v240;	// L427
                } else {
                  ac_int<9, false> v241 = exp_result;	// L429
                  int32_t v242 = v241;	// L430
                  bool v243 = v242 <= 1;	// L431
                  ac_int<18, false> v244 = magnitude;	// L432
                  bool v245;
                  ac_int<18, true> _bs_v245 = v244;
                  v245 = _bs_v245[16];	// L433
                  bool v246 = v245 == 0;	// L434
                  bool v247 = v243 & v246;	// L435
                  if (v247) {	// L436
                    bool v248 = sign_large;	// L437
                    uint16_t v249 = result_o;	// L438
                    int16_t v250;
                    ac_int<16, true> _bs_v250 = v249;
                    _bs_v250[15] = v248;
                    v250 = _bs_v250;	// L439
                    result_o = v250;	// L440
                    uint8_t v251 = rounded;	// L441
                    ac_int<7, false> v252;
                    ac_int<8, true> _bs_v252 = v251;
                    v252 = _bs_v252.slc<7>(0);	// L442
                    uint16_t v253 = result_o;	// L443
                    int16_t v254;
                    ac_int<16, true> _bs_v254 = v253;
                    _bs_v254.set_slc(0, ac_int<7, false>(v252));
                    v254 = _bs_v254;	// L444
                    result_o = v254;	// L445
                  } else {
                    bool v255 = sign_large;	// L447
                    uint16_t v256 = result_o;	// L448
                    int16_t v257;
                    ac_int<16, true> _bs_v257 = v256;
                    _bs_v257[15] = v255;
                    v257 = _bs_v257;	// L449
                    result_o = v257;	// L450
                    ac_int<9, false> v258 = exp_result;	// L451
                    uint8_t v259;
                    ac_int<9, true> _bs_v259 = v258;
                    v259 = _bs_v259.slc<8>(0);	// L452
                    uint16_t v260 = result_o;	// L453
                    int16_t v261;
                    ac_int<16, true> _bs_v261 = v260;
                    _bs_v261.set_slc(7, ac_int<8, false>(v259));
                    v261 = _bs_v261;	// L454
                    result_o = v261;	// L455
                    uint8_t v262 = rounded;	// L456
                    ac_int<7, false> v263;
                    ac_int<8, true> _bs_v263 = v262;
                    v263 = _bs_v263.slc<7>(0);	// L457
                    uint16_t v264 = result_o;	// L458
                    int16_t v265;
                    ac_int<16, true> _bs_v265 = v264;
                    _bs_v265.set_slc(0, ac_int<7, false>(v263));
                    v265 = _bs_v265;	// L459
                    result_o = v265;	// L460
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  *v12 = result_o;	// L470
}


#pragma hls_design top
#pragma hls_design ccore
#pragma hls_ccore_type combinational
void bf16_add_comb(ac_int<16, false> a_i, ac_int<16, false> b_i, ac_int<16, false> &result_o) {
  uint16_t r;
  add_bits(a_i.to_uint(), b_i.to_uint(), &r);
  result_o = r;
}
