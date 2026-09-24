
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Catapult High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <ac_std_float.h>
#include <math.h>
#include <stdint.h>
using namespace std;
void sequencer_0(
  uint64_t v0[56],
  ac_channel< uint64_t >& v1,
  ac_channel< uint64_t >& v2,
  ac_channel< uint64_t >& v3,
  ac_channel< uint64_t >& v4,
  ac_channel< uint64_t >& v5
) {	// L4
  uint64_t program[56];	// L47
  l_S_group_0_group: for (int group = 0; group < 7; group++) {	// L48
    uint64_t v6 = v0[(group * 8)];	// L49
    program[(group * 8)] = v6;	// L50
    uint64_t v7 = v0[((group * 8) + 1)];	// L51
    program[((group * 8) + 1)] = v7;	// L52
    uint64_t v8 = v0[((group * 8) + 2)];	// L53
    program[((group * 8) + 2)] = v8;	// L54
    uint64_t v9 = v0[((group * 8) + 3)];	// L55
    program[((group * 8) + 3)] = v9;	// L56
    uint64_t v10 = v0[((group * 8) + 4)];	// L57
    program[((group * 8) + 4)] = v10;	// L58
    uint64_t v11 = v0[((group * 8) + 5)];	// L59
    program[((group * 8) + 5)] = v11;	// L60
    uint64_t v12 = v0[((group * 8) + 6)];	// L61
    program[((group * 8) + 6)] = v12;	// L62
    uint64_t v13 = v0[((group * 8) + 7)];	// L63
    program[((group * 8) + 7)] = v13;	// L64
  }
  uint64_t v14 = program[0];	// L66
  uint64_t header;	// L67
  header = v14;	// L68
  uint64_t v15 = header;	// L69
  uint16_t v16;
  ap_int<64> v16_tmp = v15;
  v16 = v16_tmp(15, 0);	// L70
  int32_t v17 = v16;	// L71
  int32_t n_instr;	// L72
  n_instr = v17;	// L73
  uint64_t v18 = program[1];	// L74
  v1.write(v18);	// L75
  uint64_t v19 = program[7];	// L76
  v1.write(v19);	// L77
  uint64_t v20 = program[2];	// L78
  v2.write(v20);	// L79
  uint64_t v21 = program[4];	// L80
  v2.write(v21);	// L81
  uint64_t v22 = program[3];	// L82
  v3.write(v22);	// L83
  uint64_t v23 = program[5];	// L84
  v4.write(v23);	// L85
  uint64_t v24 = program[6];	// L86
  v5.write(v24);	// L87
  int32_t loop_body[4];	// L88
  for (int v25 = 0; v25 < 4; v25++) {	// L89
    loop_body[v25] = 0;	// L89
  }
  int32_t loop_iter[4];	// L90
  for (int v26 = 0; v26 < 4; v26++) {	// L91
    loop_iter[v26] = 0;	// L91
  }
  int32_t loop_trip[4];	// L92
  for (int v27 = 0; v27 < 4; v27++) {	// L93
    loop_trip[v27] = 0;	// L93
  }
  int32_t live_iv[4];	// L94
  for (int v28 = 0; v28 < 4; v28++) {	// L95
    live_iv[v28] = 0;	// L95
  }
  int32_t loop_sp;	// L96
  loop_sp = 0;	// L97
  int32_t pc;	// L98
  pc = 0;	// L99
  int32_t running;	// L100
  running = 1;	// L101
  while (true) {	// L102
    int32_t v29 = running;	// L103
    bool v30 = v29 == 1;	// L104
    if (!(v30)) break;
    int32_t v31 = pc;	// L107
    int64_t v32 = v31;	// L108
    int64_t v33 = v32 * 2;	// L109
    ac_int<65, true> v34 = v33;	// L110
    ac_int<65, true> v35 = v34 + 8;	// L111
    int v36 = v35;	// L112
    uint64_t v37 = program[v36];	// L113
    uint64_t control_word;	// L114
    control_word = v37;	// L115
    int32_t v38 = pc;	// L116
    int64_t v39 = v38;	// L117
    int64_t v40 = v39 * 2;	// L118
    ac_int<65, true> v41 = v40;	// L119
    ac_int<65, true> v42 = v41 + 8;	// L120
    ac_int<66, true> v43 = v42;	// L121
    ac_int<66, true> v44 = v43 + 1;	// L122
    int v45 = v44;	// L123
    uint64_t v46 = program[v45];	// L124
    uint64_t agu_word;	// L125
    agu_word = v46;	// L126
    uint64_t v47 = control_word;	// L127
    ac_int<6, false> v48;
    ap_int<64> v48_tmp = v47;
    v48 = v48_tmp(5, 0);	// L128
    int32_t v49 = v48;	// L129
    int32_t op;	// L130
    op = v49;	// L131
    uint64_t v50 = control_word;	// L132
    uint8_t v51;
    ap_int<64> v51_tmp = v50;
    v51 = v51_tmp(61, 54);	// L133
    int32_t v52 = v51;	// L134
    int32_t nr;	// L135
    nr = v52;	// L136
    int32_t v53 = op;	// L137
    bool v54 = v53 == 8;	// L138
    if (v54) {	// L139
      int32_t v55 = pc;	// L140
      ac_int<33, true> v56 = v55;	// L141
      ac_int<33, true> v57 = v56 + 1;	// L142
      int32_t v58 = v57;	// L143
      int32_t v59 = loop_sp;	// L144
      int v60 = v59;	// L145
      loop_body[v60] = v58;	// L146
      int32_t v61 = loop_sp;	// L147
      int v62 = v61;	// L148
      loop_iter[v62] = 0;	// L149
      int32_t v63 = nr;	// L150
      int32_t v64 = loop_sp;	// L151
      int v65 = v64;	// L152
      loop_trip[v65] = v63;	// L153
      int32_t v66 = loop_sp;	// L154
      int v67 = v66;	// L155
      live_iv[v67] = 0;	// L156
      int32_t v68 = loop_sp;	// L157
      ac_int<33, true> v69 = v68;	// L158
      ac_int<33, true> v70 = v69 + 1;	// L159
      int32_t v71 = v70;	// L160
      loop_sp = v71;	// L161
      int32_t v72 = pc;	// L162
      ac_int<33, true> v73 = v72;	// L163
      ac_int<33, true> v74 = v73 + 1;	// L164
      int32_t v75 = v74;	// L165
      pc = v75;	// L166
    } else {
      int32_t v76 = op;	// L168
      bool v77 = v76 == 9;	// L169
      if (v77) {	// L170
        int32_t v78 = loop_sp;	// L171
        ac_int<33, true> v79 = v78;	// L172
        ac_int<33, true> v80 = v79 - 1;	// L173
        int v81 = v80;	// L174
        int32_t v82 = loop_iter[v81];	// L175
        ac_int<33, true> v83 = v82;	// L176
        ac_int<33, true> v84 = v83 + 1;	// L177
        int32_t v85 = v84;	// L178
        int32_t next_iter;	// L179
        next_iter = v85;	// L180
        int32_t v86 = next_iter;	// L181
        int32_t v87 = loop_sp;	// L182
        ac_int<33, true> v88 = v87;	// L183
        ac_int<33, true> v89 = v88 - 1;	// L184
        int v90 = v89;	// L185
        int32_t v91 = loop_trip[v90];	// L186
        bool v92 = v86 < v91;	// L187
        if (v92) {	// L188
          int32_t v93 = next_iter;	// L189
          int32_t v94 = loop_sp;	// L190
          ac_int<33, true> v95 = v94;	// L191
          ac_int<33, true> v96 = v95 - 1;	// L192
          int v97 = v96;	// L193
          loop_iter[v97] = v93;	// L194
          int32_t v98 = next_iter;	// L195
          int32_t v99 = loop_sp;	// L196
          ac_int<33, true> v100 = v99;	// L197
          ac_int<33, true> v101 = v100 - 1;	// L198
          int v102 = v101;	// L199
          live_iv[v102] = v98;	// L200
          int32_t v103 = loop_sp;	// L201
          ac_int<33, true> v104 = v103;	// L202
          ac_int<33, true> v105 = v104 - 1;	// L203
          int v106 = v105;	// L204
          int32_t v107 = loop_body[v106];	// L205
          pc = v107;	// L206
        } else {
          int32_t v108 = loop_sp;	// L208
          ac_int<33, true> v109 = v108;	// L209
          ac_int<33, true> v110 = v109 - 1;	// L210
          int32_t v111 = v110;	// L211
          loop_sp = v111;	// L212
          int32_t v112 = pc;	// L213
          ac_int<33, true> v113 = v112;	// L214
          ac_int<33, true> v114 = v113 + 1;	// L215
          int32_t v115 = v114;	// L216
          pc = v115;	// L217
        }
      } else {
        uint64_t v116 = control_word;	// L220
        ac_int<12, false> v117;
        ap_int<64> v117_tmp = v116;
        v117 = v117_tmp(17, 6);	// L221
        int32_t v118 = v117;	// L222
        int32_t f0;	// L223
        f0 = v118;	// L224
        uint64_t v119 = control_word;	// L225
        ac_int<12, false> v120;
        ap_int<64> v120_tmp = v119;
        v120 = v120_tmp(29, 18);	// L226
        int32_t v121 = v120;	// L227
        int32_t f1;	// L228
        f1 = v121;	// L229
        uint64_t v122 = control_word;	// L230
        ac_int<12, false> v123;
        ap_int<64> v123_tmp = v122;
        v123 = v123_tmp(41, 30);	// L231
        int32_t v124 = v123;	// L232
        int32_t f2;	// L233
        f2 = v124;	// L234
        uint64_t v125 = control_word;	// L235
        ac_int<12, false> v126;
        ap_int<64> v126_tmp = v125;
        v126 = v126_tmp(53, 42);	// L236
        int32_t v127 = v126;	// L237
        int32_t f3;	// L238
        f3 = v127;	// L239
        uint64_t v128 = agu_word;	// L240
        ac_int<4, false> v129;
        ap_int<64> v129_tmp = v128;
        v129 = v129_tmp(3, 0);	// L241
        int32_t v130 = v129;	// L242
        int32_t target;	// L243
        target = v130;	// L244
        uint64_t v131 = agu_word;	// L245
        ac_int<3, false> v132;
        ap_int<64> v132_tmp = v131;
        v132 = v132_tmp(6, 4);	// L246
        int32_t v133 = v132;	// L247
        int32_t level;	// L248
        level = v133;	// L249
        uint64_t v134 = agu_word;	// L250
        ac_int<12, false> v135;
        ap_int<64> v135_tmp = v134;
        v135 = v135_tmp(18, 7);	// L251
        int32_t v136 = v135;	// L252
        int32_t stride;	// L253
        stride = v136;	// L254
        int32_t v137 = level;	// L255
        int v138 = v137;	// L256
        int32_t v139 = live_iv[v138];	// L257
        int32_t v140 = stride;	// L258
        int64_t v141 = v139;	// L259
        int64_t v142 = v140;	// L260
        int64_t v143 = v141 * v142;	// L261
        int32_t v144 = v143;	// L262
        int32_t offset;	// L263
        offset = v144;	// L264
        int32_t v145 = target;	// L265
        bool v146 = v145 == 1;	// L266
        if (v146) {	// L267
          int32_t v147 = f0;	// L268
          int32_t v148 = offset;	// L269
          ac_int<33, true> v149 = v147;	// L270
          ac_int<33, true> v150 = v148;	// L271
          ac_int<33, true> v151 = v149 + v150;	// L272
          int32_t v152 = v151;	// L273
          f0 = v152;	// L274
        }
        int32_t v153 = target;	// L276
        bool v154 = v153 == 2;	// L277
        if (v154) {	// L278
          int32_t v155 = f1;	// L279
          int32_t v156 = offset;	// L280
          ac_int<33, true> v157 = v155;	// L281
          ac_int<33, true> v158 = v156;	// L282
          ac_int<33, true> v159 = v157 + v158;	// L283
          int32_t v160 = v159;	// L284
          f1 = v160;	// L285
        }
        int32_t v161 = target;	// L287
        bool v162 = v161 == 3;	// L288
        if (v162) {	// L289
          int32_t v163 = f2;	// L290
          int32_t v164 = offset;	// L291
          ac_int<33, true> v165 = v163;	// L292
          ac_int<33, true> v166 = v164;	// L293
          ac_int<33, true> v167 = v165 + v166;	// L294
          int32_t v168 = v167;	// L295
          f2 = v168;	// L296
        }
        int32_t v169 = target;	// L298
        bool v170 = v169 == 4;	// L299
        if (v170) {	// L300
          int32_t v171 = f3;	// L301
          int32_t v172 = offset;	// L302
          ac_int<33, true> v173 = v171;	// L303
          ac_int<33, true> v174 = v172;	// L304
          ac_int<33, true> v175 = v173 + v174;	// L305
          int32_t v176 = v175;	// L306
          f3 = v176;	// L307
        }
        uint64_t v177 = agu_word;	// L309
        ac_int<4, false> v178;
        ap_int<64> v178_tmp = v177;
        v178 = v178_tmp(22, 19);	// L310
        int32_t v179 = v178;	// L311
        int32_t target1;	// L312
        target1 = v179;	// L313
        uint64_t v180 = agu_word;	// L314
        ac_int<3, false> v181;
        ap_int<64> v181_tmp = v180;
        v181 = v181_tmp(25, 23);	// L315
        int32_t v182 = v181;	// L316
        int32_t level1;	// L317
        level1 = v182;	// L318
        uint64_t v183 = agu_word;	// L319
        ac_int<12, false> v184;
        ap_int<64> v184_tmp = v183;
        v184 = v184_tmp(37, 26);	// L320
        int32_t v185 = v184;	// L321
        int32_t stride1;	// L322
        stride1 = v185;	// L323
        int32_t v186 = level1;	// L324
        int v187 = v186;	// L325
        int32_t v188 = live_iv[v187];	// L326
        int32_t v189 = stride1;	// L327
        int64_t v190 = v188;	// L328
        int64_t v191 = v189;	// L329
        int64_t v192 = v190 * v191;	// L330
        int32_t v193 = v192;	// L331
        int32_t offset1;	// L332
        offset1 = v193;	// L333
        int32_t v194 = target1;	// L334
        bool v195 = v194 == 1;	// L335
        if (v195) {	// L336
          int32_t v196 = f0;	// L337
          int32_t v197 = offset1;	// L338
          ac_int<33, true> v198 = v196;	// L339
          ac_int<33, true> v199 = v197;	// L340
          ac_int<33, true> v200 = v198 + v199;	// L341
          int32_t v201 = v200;	// L342
          f0 = v201;	// L343
        }
        int32_t v202 = target1;	// L345
        bool v203 = v202 == 2;	// L346
        if (v203) {	// L347
          int32_t v204 = f1;	// L348
          int32_t v205 = offset1;	// L349
          ac_int<33, true> v206 = v204;	// L350
          ac_int<33, true> v207 = v205;	// L351
          ac_int<33, true> v208 = v206 + v207;	// L352
          int32_t v209 = v208;	// L353
          f1 = v209;	// L354
        }
        int32_t v210 = target1;	// L356
        bool v211 = v210 == 3;	// L357
        if (v211) {	// L358
          int32_t v212 = f2;	// L359
          int32_t v213 = offset1;	// L360
          ac_int<33, true> v214 = v212;	// L361
          ac_int<33, true> v215 = v213;	// L362
          ac_int<33, true> v216 = v214 + v215;	// L363
          int32_t v217 = v216;	// L364
          f2 = v217;	// L365
        }
        int32_t v218 = target1;	// L367
        bool v219 = v218 == 4;	// L368
        if (v219) {	// L369
          int32_t v220 = f3;	// L370
          int32_t v221 = offset1;	// L371
          ac_int<33, true> v222 = v220;	// L372
          ac_int<33, true> v223 = v221;	// L373
          ac_int<33, true> v224 = v222 + v223;	// L374
          int32_t v225 = v224;	// L375
          f3 = v225;	// L376
        }
        uint64_t v226 = agu_word;	// L378
        ac_int<4, false> v227;
        ap_int<64> v227_tmp = v226;
        v227 = v227_tmp(41, 38);	// L379
        int32_t v228 = v227;	// L380
        int32_t target2;	// L381
        target2 = v228;	// L382
        uint64_t v229 = agu_word;	// L383
        ac_int<3, false> v230;
        ap_int<64> v230_tmp = v229;
        v230 = v230_tmp(44, 42);	// L384
        int32_t v231 = v230;	// L385
        int32_t level2;	// L386
        level2 = v231;	// L387
        uint64_t v232 = agu_word;	// L388
        ac_int<12, false> v233;
        ap_int<64> v233_tmp = v232;
        v233 = v233_tmp(56, 45);	// L389
        int32_t v234 = v233;	// L390
        int32_t stride2;	// L391
        stride2 = v234;	// L392
        int32_t v235 = level2;	// L393
        int v236 = v235;	// L394
        int32_t v237 = live_iv[v236];	// L395
        int32_t v238 = stride2;	// L396
        int64_t v239 = v237;	// L397
        int64_t v240 = v238;	// L398
        int64_t v241 = v239 * v240;	// L399
        int32_t v242 = v241;	// L400
        int32_t offset2;	// L401
        offset2 = v242;	// L402
        int32_t v243 = target2;	// L403
        bool v244 = v243 == 1;	// L404
        if (v244) {	// L405
          int32_t v245 = f0;	// L406
          int32_t v246 = offset2;	// L407
          ac_int<33, true> v247 = v245;	// L408
          ac_int<33, true> v248 = v246;	// L409
          ac_int<33, true> v249 = v247 + v248;	// L410
          int32_t v250 = v249;	// L411
          f0 = v250;	// L412
        }
        int32_t v251 = target2;	// L414
        bool v252 = v251 == 2;	// L415
        if (v252) {	// L416
          int32_t v253 = f1;	// L417
          int32_t v254 = offset2;	// L418
          ac_int<33, true> v255 = v253;	// L419
          ac_int<33, true> v256 = v254;	// L420
          ac_int<33, true> v257 = v255 + v256;	// L421
          int32_t v258 = v257;	// L422
          f1 = v258;	// L423
        }
        int32_t v259 = target2;	// L425
        bool v260 = v259 == 3;	// L426
        if (v260) {	// L427
          int32_t v261 = f2;	// L428
          int32_t v262 = offset2;	// L429
          ac_int<33, true> v263 = v261;	// L430
          ac_int<33, true> v264 = v262;	// L431
          ac_int<33, true> v265 = v263 + v264;	// L432
          int32_t v266 = v265;	// L433
          f2 = v266;	// L434
        }
        int32_t v267 = target2;	// L436
        bool v268 = v267 == 4;	// L437
        if (v268) {	// L438
          int32_t v269 = f3;	// L439
          int32_t v270 = offset2;	// L440
          ac_int<33, true> v271 = v269;	// L441
          ac_int<33, true> v272 = v270;	// L442
          ac_int<33, true> v273 = v271 + v272;	// L443
          int32_t v274 = v273;	// L444
          f3 = v274;	// L445
        }
        uint64_t v275 = control_word;	// L447
        uint64_t resolved;	// L448
        resolved = v275;	// L449
        int32_t v276 = f0;	// L450
        ac_int<12, false> v277 = v276;	// L451
        uint64_t v278 = resolved;	// L452
        int64_t v279;
        ap_int<64> v279_tmp = v278;
        v279_tmp(17, 6) = v277;
        v279 = v279_tmp;	// L453
        resolved = v279;	// L454
        int32_t v280 = f1;	// L455
        ac_int<12, false> v281 = v280;	// L456
        uint64_t v282 = resolved;	// L457
        int64_t v283;
        ap_int<64> v283_tmp = v282;
        v283_tmp(29, 18) = v281;
        v283 = v283_tmp;	// L458
        resolved = v283;	// L459
        int32_t v284 = f2;	// L460
        ac_int<12, false> v285 = v284;	// L461
        uint64_t v286 = resolved;	// L462
        int64_t v287;
        ap_int<64> v287_tmp = v286;
        v287_tmp(41, 30) = v285;
        v287 = v287_tmp;	// L463
        resolved = v287;	// L464
        int32_t v288 = f3;	// L465
        ac_int<12, false> v289 = v288;	// L466
        uint64_t v290 = resolved;	// L467
        int64_t v291;
        ap_int<64> v291_tmp = v290;
        v291_tmp(53, 42) = v289;
        v291 = v291_tmp;	// L468
        resolved = v291;	// L469
        int32_t v292 = op;	// L470
        bool v293 = v292 == 1;	// L471
        if (v293) {	// L472
          uint64_t v294 = resolved;	// L473
          v1.write(v294);	// L474
          int32_t v295 = f0;	// L475
          bool v296 = v295 >= 2;	// L476
          if (v296) {	// L477
            uint64_t v297 = resolved;	// L478
            v3.write(v297);	// L479
          } else {
            uint64_t v298 = resolved;	// L481
            v2.write(v298);	// L482
          }
        }
        int32_t v299 = op;	// L485
        bool v300 = v299 == 3;	// L486
        if (v300) {	// L487
          uint64_t v301 = resolved;	// L488
          v2.write(v301);	// L489
          uint64_t v302 = resolved;	// L490
          v3.write(v302);	// L491
        }
        int32_t v303 = op;	// L493
        bool v304 = v303 == 4;	// L494
        if (v304) {	// L495
          uint64_t v305 = resolved;	// L496
          uint64_t spm_copy;	// L497
          spm_copy = v305;	// L498
          uint64_t v306 = spm_copy;	// L499
          int64_t v307;
          ap_int<64> v307_tmp = v306;
          v307_tmp(61, 54) = 5;
          v307 = v307_tmp;	// L500
          spm_copy = v307;	// L501
          int32_t v308 = nr;	// L502
          ac_int<12, false> v309 = v308;	// L503
          uint64_t v310 = spm_copy;	// L504
          int64_t v311;
          ap_int<64> v311_tmp = v310;
          v311_tmp(29, 18) = v309;
          v311 = v311_tmp;	// L505
          spm_copy = v311;	// L506
          uint64_t v312 = spm_copy;	// L507
          v2.write(v312);	// L508
          uint64_t v313 = resolved;	// L509
          v3.write(v313);	// L510
          uint64_t v314 = resolved;	// L511
          v4.write(v314);	// L512
        }
        int32_t v315 = op;	// L514
        bool v316 = v315 == 5;	// L515
        if (v316) {	// L516
          uint64_t v317 = resolved;	// L517
          uint64_t accu_copy;	// L518
          accu_copy = v317;	// L519
          int32_t v318 = nr;	// L520
          int64_t v319 = v318;	// L521
          int64_t v320 = v319 * 2;	// L522
          uint8_t v321 = v320;	// L523
          uint64_t v322 = accu_copy;	// L524
          int64_t v323;
          ap_int<64> v323_tmp = v322;
          v323_tmp(61, 54) = v321;
          v323 = v323_tmp;	// L525
          accu_copy = v323;	// L526
          uint64_t v324 = accu_copy;	// L527
          v4.write(v324);	// L528
        }
        int32_t v325 = op;	// L530
        bool v326 = v325 == 6;	// L531
        if (v326) {	// L532
          uint64_t v327 = resolved;	// L533
          v4.write(v327);	// L534
        }
        int32_t v328 = op;	// L536
        bool v329 = v328 == 10;	// L537
        if (v329) {	// L538
          uint64_t v330 = resolved;	// L539
          uint64_t fused_copy;	// L540
          fused_copy = v330;	// L541
          int32_t v331 = nr;	// L542
          int64_t v332 = v331;	// L543
          int64_t v333 = v332 * 2;	// L544
          uint8_t v334 = v333;	// L545
          uint64_t v335 = fused_copy;	// L546
          int64_t v336;
          ap_int<64> v336_tmp = v335;
          v336_tmp(61, 54) = v334;
          v336 = v336_tmp;	// L547
          fused_copy = v336;	// L548
          uint64_t v337 = fused_copy;	// L549
          v4.write(v337);	// L550
        }
        int32_t v338 = op;	// L552
        bool v339 = v338 == 7;	// L553
        if (v339) {	// L554
          uint64_t v340 = resolved;	// L555
          v4.write(v340);	// L556
          uint64_t v341 = resolved;	// L557
          v5.write(v341);	// L558
        }
        int32_t v342 = pc;	// L560
        ac_int<33, true> v343 = v342;	// L561
        ac_int<33, true> v344 = v343 + 1;	// L562
        int32_t v345 = v344;	// L563
        pc = v345;	// L564
      }
    }
    int32_t v346 = pc;	// L567
    int32_t v347 = n_instr;	// L568
    bool v348 = v346 >= v347;	// L569
    if (v348) {	// L570
      running = 0;	// L571
    }
  }
}

void dma_ld_0(
  int8_t v349[256],
  int8_t v350[256],
  ac_channel< uint64_t >& v351,
  ac_channel< uint32_t >& v352,
  ac_channel< uint32_t >& v353
) {	// L577
  uint64_t v354 = v351.read();	// L606
  uint64_t count_word;	// L607
  count_word = v354;	// L608
  uint64_t v355 = count_word;	// L609
  uint16_t v356;
  ap_int<64> v356_tmp = v355;
  v356 = v356_tmp(15, 0);	// L610
  int32_t v357 = v356;	// L611
  int32_t n_row;	// L612
  n_row = v357;	// L613
  uint64_t v358 = v351.read();	// L614
  uint64_t span_word;	// L615
  span_word = v358;	// L616
  uint64_t v359 = span_word;	// L617
  uint16_t v360;
  ap_int<64> v360_tmp = v359;
  v360 = v360_tmp(15, 0);	// L618
  int32_t v361 = v360;	// L619
  int32_t a_rows;	// L620
  a_rows = v361;	// L621
  uint64_t v362 = span_word;	// L622
  uint16_t v363;
  ap_int<64> v363_tmp = v362;
  v363 = v363_tmp(31, 16);	// L623
  int32_t v364 = v363;	// L624
  int32_t b_rows;	// L625
  b_rows = v364;	// L626
  uint32_t a_onchip[65];	// L627
  uint32_t b_onchip[65];	// L628
  int32_t v365 = a_rows;	// L629
  int64_t v366 = v365;	// L630
  int64_t v367 = v366 * 4;	// L631
  int32_t v368 = v367;	// L632
  int32_t a_groups;	// L633
  a_groups = v368;	// L634
  int32_t v369 = a_groups;	// L635
  int v370 = v369;	// L636
  for (int v371 = 0; v371 < v370; v371 += 1) {	// L637
    uint32_t packed_a;	// L638
    packed_a = 0;	// L639
    int64_t v372 = v371;	// L640
    ac_int<97, true> v373 = v372;	// L641
    ac_int<97, true> v374 = v373 * 4;	// L642
    int v375 = v374;	// L643
    int8_t v376 = v349[v375];	// L644
    int8_t a_value;	// L645
    a_value = v376;	// L646
    int8_t v377 = a_value;	// L647
    uint32_t v378 = packed_a;	// L648
    int32_t v379;
    ap_int<32> v379_tmp = v378;
    v379_tmp(7, 0) = v377;
    v379 = v379_tmp;	// L649
    packed_a = v379;	// L650
    ac_int<98, true> v380 = v374;	// L651
    ac_int<98, true> v381 = v380 + 1;	// L652
    int v382 = v381;	// L653
    int8_t v383 = v349[v382];	// L654
    int8_t a_value1;	// L655
    a_value1 = v383;	// L656
    int8_t v384 = a_value1;	// L657
    uint32_t v385 = packed_a;	// L658
    int32_t v386;
    ap_int<32> v386_tmp = v385;
    v386_tmp(15, 8) = v384;
    v386 = v386_tmp;	// L659
    packed_a = v386;	// L660
    ac_int<98, true> v387 = v380 + 2;	// L661
    int v388 = v387;	// L662
    int8_t v389 = v349[v388];	// L663
    int8_t a_value2;	// L664
    a_value2 = v389;	// L665
    int8_t v390 = a_value2;	// L666
    uint32_t v391 = packed_a;	// L667
    int32_t v392;
    ap_int<32> v392_tmp = v391;
    v392_tmp(23, 16) = v390;
    v392 = v392_tmp;	// L668
    packed_a = v392;	// L669
    ac_int<98, true> v393 = v380 + 3;	// L670
    int v394 = v393;	// L671
    int8_t v395 = v349[v394];	// L672
    int8_t a_value3;	// L673
    a_value3 = v395;	// L674
    int8_t v396 = a_value3;	// L675
    uint32_t v397 = packed_a;	// L676
    int32_t v398;
    ap_int<32> v398_tmp = v397;
    v398_tmp(31, 24) = v396;
    v398 = v398_tmp;	// L677
    packed_a = v398;	// L678
    uint32_t v399 = packed_a;	// L679
    a_onchip[v371] = v399;	// L680
  }
  int32_t v400 = b_rows;	// L682
  int64_t v401 = v400;	// L683
  int64_t v402 = v401 * 4;	// L684
  int32_t v403 = v402;	// L685
  int32_t b_groups;	// L686
  b_groups = v403;	// L687
  int32_t v404 = b_groups;	// L688
  int v405 = v404;	// L689
  for (int v406 = 0; v406 < v405; v406 += 1) {	// L690
    uint32_t packed_b;	// L691
    packed_b = 0;	// L692
    int64_t v407 = v406;	// L693
    ac_int<97, true> v408 = v407;	// L694
    ac_int<97, true> v409 = v408 * 4;	// L695
    int v410 = v409;	// L696
    int8_t v411 = v350[v410];	// L697
    int8_t b_value;	// L698
    b_value = v411;	// L699
    int8_t v412 = b_value;	// L700
    uint32_t v413 = packed_b;	// L701
    int32_t v414;
    ap_int<32> v414_tmp = v413;
    v414_tmp(7, 0) = v412;
    v414 = v414_tmp;	// L702
    packed_b = v414;	// L703
    ac_int<98, true> v415 = v409;	// L704
    ac_int<98, true> v416 = v415 + 1;	// L705
    int v417 = v416;	// L706
    int8_t v418 = v350[v417];	// L707
    int8_t b_value1;	// L708
    b_value1 = v418;	// L709
    int8_t v419 = b_value1;	// L710
    uint32_t v420 = packed_b;	// L711
    int32_t v421;
    ap_int<32> v421_tmp = v420;
    v421_tmp(15, 8) = v419;
    v421 = v421_tmp;	// L712
    packed_b = v421;	// L713
    ac_int<98, true> v422 = v415 + 2;	// L714
    int v423 = v422;	// L715
    int8_t v424 = v350[v423];	// L716
    int8_t b_value2;	// L717
    b_value2 = v424;	// L718
    int8_t v425 = b_value2;	// L719
    uint32_t v426 = packed_b;	// L720
    int32_t v427;
    ap_int<32> v427_tmp = v426;
    v427_tmp(23, 16) = v425;
    v427 = v427_tmp;	// L721
    packed_b = v427;	// L722
    ac_int<98, true> v428 = v415 + 3;	// L723
    int v429 = v428;	// L724
    int8_t v430 = v350[v429];	// L725
    int8_t b_value3;	// L726
    b_value3 = v430;	// L727
    int8_t v431 = b_value3;	// L728
    uint32_t v432 = packed_b;	// L729
    int32_t v433;
    ap_int<32> v433_tmp = v432;
    v433_tmp(31, 24) = v431;
    v433 = v433_tmp;	// L730
    packed_b = v433;	// L731
    uint32_t v434 = packed_b;	// L732
    b_onchip[v406] = v434;	// L733
  }
  int32_t route;	// L735
  route = 0;	// L736
  int32_t dram_row0;	// L737
  dram_row0 = 0;	// L738
  int32_t col_block;	// L739
  col_block = 0;	// L740
  int32_t instr_rows;	// L741
  instr_rows = 0;	// L742
  int32_t row;	// L743
  row = -1;	// L744
  int32_t v435 = n_row;	// L745
  int v436 = v435;	// L746
  for (int v437 = 0; v437 < v436; v437 += 1) {	// L747
    int32_t v438 = row;	// L748
    ac_int<33, true> v439 = v438;	// L749
    ac_int<33, true> v440 = v439 + 1;	// L750
    int32_t v441 = v440;	// L751
    row = v441;	// L752
    int32_t v442 = row;	// L753
    int32_t v443 = instr_rows;	// L754
    bool v444 = v442 >= v443;	// L755
    if (v444) {	// L756
      uint64_t v445 = v351.read();	// L757
      uint64_t word;	// L758
      word = v445;	// L759
      uint64_t v446 = word;	// L760
      ac_int<12, false> v447;
      ap_int<64> v447_tmp = v446;
      v447 = v447_tmp(17, 6);	// L761
      int32_t v448 = v447;	// L762
      route = v448;	// L763
      uint64_t v449 = word;	// L764
      ac_int<12, false> v450;
      ap_int<64> v450_tmp = v449;
      v450 = v450_tmp(29, 18);	// L765
      int32_t v451 = v450;	// L766
      dram_row0 = v451;	// L767
      uint64_t v452 = word;	// L768
      ac_int<12, false> v453;
      ap_int<64> v453_tmp = v452;
      v453 = v453_tmp(41, 30);	// L769
      int32_t v454 = v453;	// L770
      col_block = v454;	// L771
      uint64_t v455 = word;	// L772
      uint8_t v456;
      ap_int<64> v456_tmp = v455;
      v456 = v456_tmp(61, 54);	// L773
      int32_t v457 = v456;	// L774
      instr_rows = v457;	// L775
      row = 0;	// L776
    }
    uint32_t packed;	// L778
    packed = 0;	// L779
    int32_t v458 = route;	// L780
    int32_t v459 = v458 & 1;	// L781
    bool v460 = v459 == 0;	// L782
    if (v460) {	// L783
      int32_t v461 = dram_row0;	// L784
      int32_t v462 = row;	// L785
      ac_int<33, true> v463 = v461;	// L786
      ac_int<33, true> v464 = v462;	// L787
      ac_int<33, true> v465 = v463 + v464;	// L788
      ac_int<65, true> v466 = v465;	// L789
      ac_int<65, true> v467 = v466 * 4;	// L790
      int32_t v468 = col_block;	// L791
      ac_int<66, true> v469 = v467;	// L792
      ac_int<66, true> v470 = v468;	// L793
      ac_int<66, true> v471 = v469 + v470;	// L794
      int v472 = v471;	// L795
      uint32_t v473 = a_onchip[v472];	// L796
      packed = v473;	// L797
    } else {
      int32_t v474 = dram_row0;	// L799
      int32_t v475 = row;	// L800
      ac_int<33, true> v476 = v474;	// L801
      ac_int<33, true> v477 = v475;	// L802
      ac_int<33, true> v478 = v476 + v477;	// L803
      ac_int<65, true> v479 = v478;	// L804
      ac_int<65, true> v480 = v479 * 4;	// L805
      int32_t v481 = col_block;	// L806
      ac_int<66, true> v482 = v480;	// L807
      ac_int<66, true> v483 = v481;	// L808
      ac_int<66, true> v484 = v482 + v483;	// L809
      int v485 = v484;	// L810
      uint32_t v486 = b_onchip[v485];	// L811
      packed = v486;	// L812
    }
    int32_t v487 = route;	// L814
    bool v488 = v487 >= 2;	// L815
    if (v488) {	// L816
      uint32_t v489 = packed;	// L817
      v352.write(v489);	// L818
    } else {
      uint32_t v490 = packed;	// L820
      v353.write(v490);	// L821
    }
  }
}

void spm_0(
  ac_channel< uint64_t >& v491,
  ac_channel< uint32_t >& v492,
  ac_channel< uint32_t >& v493,
  ac_channel< uint32_t >& v494
) {	// L826
  uint32_t spad[64];	// L846
  uint64_t v495 = v491.read();	// L847
  uint64_t count_word1;	// L848
  count_word1 = v495;	// L849
  uint64_t v496 = count_word1;	// L850
  uint16_t v497;
  ap_int<64> v497_tmp = v496;
  v497 = v497_tmp(15, 0);	// L851
  int32_t v498 = v497;	// L852
  int32_t n_row1;	// L853
  n_row1 = v498;	// L854
  uint64_t v499 = v491.read();	// L855
  uint64_t array_counts_word;	// L856
  array_counts_word = v499;	// L857
  uint32_t array_counts;	// L858
  array_counts = 0;	// L859
  uint64_t v500 = array_counts_word;	// L860
  uint32_t v501;
  ap_int<64> v501_tmp = v500;
  v501 = v501_tmp(31, 0);	// L861
  uint32_t v502 = array_counts;	// L862
  int32_t v503;
  ap_int<32> v503_tmp = v502;
  v503_tmp(31, 0) = v501;
  v503 = v503_tmp;	// L863
  array_counts = v503;	// L864
  uint32_t v504 = array_counts;	// L865
  v492.write(v504);	// L866
  int32_t op1;	// L867
  op1 = 0;	// L868
  int32_t f11;	// L869
  f11 = 0;	// L870
  int32_t spad_base;	// L871
  spad_base = 0;	// L872
  int32_t instr_rows1;	// L873
  instr_rows1 = 0;	// L874
  int32_t row1;	// L875
  row1 = -1;	// L876
  int32_t v505 = n_row1;	// L877
  int v506 = v505;	// L878
  for (int v507 = 0; v507 < v506; v507 += 1) {	// L879
    int32_t v508 = row1;	// L880
    ac_int<33, true> v509 = v508;	// L881
    ac_int<33, true> v510 = v509 + 1;	// L882
    int32_t v511 = v510;	// L883
    row1 = v511;	// L884
    int32_t v512 = row1;	// L885
    int32_t v513 = instr_rows1;	// L886
    bool v514 = v512 >= v513;	// L887
    if (v514) {	// L888
      uint64_t v515 = v491.read();	// L889
      uint64_t word1;	// L890
      word1 = v515;	// L891
      uint64_t v516 = word1;	// L892
      ac_int<6, false> v517;
      ap_int<64> v517_tmp = v516;
      v517 = v517_tmp(5, 0);	// L893
      int32_t v518 = v517;	// L894
      op1 = v518;	// L895
      uint64_t v519 = word1;	// L896
      ac_int<12, false> v520;
      ap_int<64> v520_tmp = v519;
      v520 = v520_tmp(29, 18);	// L897
      int32_t v521 = v520;	// L898
      f11 = v521;	// L899
      uint64_t v522 = word1;	// L900
      ac_int<12, false> v523;
      ap_int<64> v523_tmp = v522;
      v523 = v523_tmp(53, 42);	// L901
      int32_t v524 = v523;	// L902
      spad_base = v524;	// L903
      uint64_t v525 = word1;	// L904
      uint8_t v526;
      ap_int<64> v526_tmp = v525;
      v526 = v526_tmp(61, 54);	// L905
      int32_t v527 = v526;	// L906
      instr_rows1 = v527;	// L907
      row1 = 0;	// L908
    }
    int32_t v528 = f11;	// L910
    int32_t v529 = row1;	// L911
    ac_int<33, true> v530 = v528;	// L912
    ac_int<33, true> v531 = v529;	// L913
    ac_int<33, true> v532 = v530 + v531;	// L914
    int32_t v533 = v532;	// L915
    int32_t read_row;	// L916
    read_row = v533;	// L917
    int32_t v534 = op1;	// L918
    bool v535 = v534 == 4;	// L919
    if (v535) {	// L920
      int32_t v536 = spad_base;	// L921
      int32_t v537 = row1;	// L922
      ac_int<33, true> v538 = v536;	// L923
      ac_int<33, true> v539 = v537;	// L924
      ac_int<33, true> v540 = v538 + v539;	// L925
      ac_int<34, true> v541 = v540;	// L926
      ac_int<34, true> v542 = v541 - 1;	// L927
      int32_t v543 = v542;	// L928
      read_row = v543;	// L929
      int32_t v544 = row1;	// L930
      bool v545 = v544 == 0;	// L931
      if (v545) {	// L932
        int32_t v546 = spad_base;	// L933
        read_row = v546;	// L934
      }
    }
    int32_t v547 = op1;	// L937
    bool v548 = v547 == 1;	// L938
    if (v548) {	// L939
      uint32_t v549 = v493.read();	// L940
      int32_t v550 = spad_base;	// L941
      int32_t v551 = row1;	// L942
      ac_int<33, true> v552 = v550;	// L943
      ac_int<33, true> v553 = v551;	// L944
      ac_int<33, true> v554 = v552 + v553;	// L945
      int v555 = v554;	// L946
      spad[v555] = v549;	// L947
    } else {
      int32_t v556 = read_row;	// L949
      int v557 = v556;	// L950
      uint32_t v558 = spad[v557];	// L951
      uint32_t loaded;	// L952
      loaded = v558;	// L953
      int32_t v559 = op1;	// L954
      bool v560 = v559 == 3;	// L955
      if (v560) {	// L956
        uint32_t v561 = loaded;	// L957
        v494.write(v561);	// L958
      } else {
        uint32_t v562 = loaded;	// L960
        uint32_t weight_word;	// L961
        weight_word = v562;	// L962
        int32_t v563 = row1;	// L963
        bool v564 = v563 == 0;	// L964
        if (v564) {	// L965
          uint32_t header1;	// L966
          header1 = 0;	// L967
          int32_t v565 = f11;	// L968
          ac_int<12, false> v566 = v565;	// L969
          uint32_t v567 = header1;	// L970
          int32_t v568;
          ap_int<32> v568_tmp = v567;
          v568_tmp(11, 0) = v566;
          v568 = v568_tmp;	// L971
          header1 = v568;	// L972
          uint32_t v569 = header1;	// L973
          weight_word = v569;	// L974
        }
        uint32_t v570 = weight_word;	// L976
        v492.write(v570);	// L977
      }
    }
  }
}

void vru_0(
  ac_channel< uint64_t >& v571,
  ac_channel< uint32_t >& v572,
  ac_channel< uint32_t >& v573,
  ac_channel< uint32_t >& v574
) {	// L983
  uint64_t v575 = v571.read();	// L1000
  uint64_t count_word2;	// L1001
  count_word2 = v575;	// L1002
  uint64_t v576 = count_word2;	// L1003
  uint16_t v577;
  ap_int<64> v577_tmp = v576;
  v577 = v577_tmp(15, 0);	// L1004
  int32_t v578 = v577;	// L1005
  int32_t n_word;	// L1006
  n_word = v578;	// L1007
  uint32_t vr[64];	// L1008
  int32_t op2;	// L1009
  op2 = 0;	// L1010
  int32_t vr_base;	// L1011
  vr_base = 0;	// L1012
  int32_t dma_base;	// L1013
  dma_base = 0;	// L1014
  int32_t instr_rows2;	// L1015
  instr_rows2 = 0;	// L1016
  int32_t row2;	// L1017
  row2 = -1;	// L1018
  int32_t v579 = n_word;	// L1019
  int v580 = v579;	// L1020
  for (int v581 = 0; v581 < v580; v581 += 1) {	// L1021
    int32_t v582 = row2;	// L1022
    ac_int<33, true> v583 = v582;	// L1023
    ac_int<33, true> v584 = v583 + 1;	// L1024
    int32_t v585 = v584;	// L1025
    row2 = v585;	// L1026
    int32_t v586 = row2;	// L1027
    int32_t v587 = instr_rows2;	// L1028
    bool v588 = v586 >= v587;	// L1029
    if (v588) {	// L1030
      uint64_t v589 = v571.read();	// L1031
      uint64_t word2;	// L1032
      word2 = v589;	// L1033
      uint64_t v590 = word2;	// L1034
      ac_int<6, false> v591;
      ap_int<64> v591_tmp = v590;
      v591 = v591_tmp(5, 0);	// L1035
      int32_t v592 = v591;	// L1036
      op2 = v592;	// L1037
      uint64_t v593 = word2;	// L1038
      ac_int<12, false> v594;
      ap_int<64> v594_tmp = v593;
      v594 = v594_tmp(17, 6);	// L1039
      int32_t v595 = v594;	// L1040
      vr_base = v595;	// L1041
      uint64_t v596 = word2;	// L1042
      ac_int<12, false> v597;
      ap_int<64> v597_tmp = v596;
      v597 = v597_tmp(53, 42);	// L1043
      int32_t v598 = v597;	// L1044
      dma_base = v598;	// L1045
      uint64_t v599 = word2;	// L1046
      uint8_t v600;
      ap_int<64> v600_tmp = v599;
      v600 = v600_tmp(61, 54);	// L1047
      int32_t v601 = v600;	// L1048
      instr_rows2 = v601;	// L1049
      row2 = 0;	// L1050
    }
    int32_t v602 = op2;	// L1052
    bool v603 = v602 == 4;	// L1053
    if (v603) {	// L1054
      int32_t v604 = vr_base;	// L1055
      int32_t v605 = row2;	// L1056
      ac_int<33, true> v606 = v604;	// L1057
      ac_int<33, true> v607 = v605;	// L1058
      ac_int<33, true> v608 = v606 + v607;	// L1059
      int v609 = v608;	// L1060
      uint32_t v610 = vr[v609];	// L1061
      uint32_t activation;	// L1062
      activation = v610;	// L1063
      uint32_t v611 = activation;	// L1064
      v572.write(v611);	// L1065
    } else {
      int32_t v612 = vr_base;	// L1067
      int32_t v613 = row2;	// L1068
      ac_int<33, true> v614 = v612;	// L1069
      ac_int<33, true> v615 = v613;	// L1070
      ac_int<33, true> v616 = v614 + v615;	// L1071
      int32_t v617 = v616;	// L1072
      int32_t write_row;	// L1073
      write_row = v617;	// L1074
      int32_t v618 = op2;	// L1075
      bool v619 = v618 == 1;	// L1076
      if (v619) {	// L1077
        int32_t v620 = dma_base;	// L1078
        int32_t v621 = row2;	// L1079
        ac_int<33, true> v622 = v620;	// L1080
        ac_int<33, true> v623 = v621;	// L1081
        ac_int<33, true> v624 = v622 + v623;	// L1082
        int32_t v625 = v624;	// L1083
        write_row = v625;	// L1084
      }
      uint32_t write_word;	// L1086
      write_word = 0;	// L1087
      int32_t v626 = op2;	// L1088
      bool v627 = v626 == 3;	// L1089
      if (v627) {	// L1090
        uint32_t v628 = v573.read();	// L1091
        write_word = v628;	// L1092
      } else {
        uint32_t v629 = v574.read();	// L1094
        write_word = v629;	// L1095
      }
      uint32_t v630 = write_word;	// L1097
      int32_t v631 = write_row;	// L1098
      int v632 = v631;	// L1099
      vr[v632] = v630;	// L1100
    }
  }
}

void wld_0_0(
  ac_channel< uint32_t >& v633,
  ac_channel< uint32_t >& v634,
  ac_channel< uint32_t >& v635,
  ac_channel< uint32_t >& v636
) {	// L1105
  uint32_t counts_word;	// L1116
  counts_word = 0;	// L1117
  uint32_t v637 = v633.read();	// L1118
  counts_word = v637;	// L1119
  uint32_t v638 = counts_word;	// L1120
  v634.write(v638);	// L1121
  uint32_t v639 = counts_word;	// L1122
  v635.write(v639);	// L1123
  uint32_t v640 = counts_word;	// L1124
  uint16_t v641;
  ap_int<32> v641_tmp = v640;
  v641 = v641_tmp(15, 0);	// L1125
  int32_t v642 = v641;	// L1126
  int32_t n_mm;	// L1127
  n_mm = v642;	// L1128
  uint32_t trip_word;	// L1129
  trip_word = 0;	// L1130
  uint32_t v643 = counts_word;	// L1131
  uint16_t v644;
  ap_int<32> v644_tmp = v643;
  v644 = v644_tmp(31, 16);	// L1132
  uint32_t v645 = trip_word;	// L1133
  int32_t v646;
  ap_int<32> v646_tmp = v645;
  v646_tmp(15, 0) = v644;
  v646 = v646_tmp;	// L1134
  trip_word = v646;	// L1135
  uint32_t v647 = trip_word;	// L1136
  v636.write(v647);	// L1137
  int32_t v648 = n_mm;	// L1138
  int v649 = v648;	// L1139
  for (int v650 = 0; v650 < v649; v650 += 1) {	// L1140
    uint32_t header2;	// L1141
    header2 = 0;	// L1142
    uint32_t v651 = v633.read();	// L1143
    header2 = v651;	// L1144
    uint32_t v652 = header2;	// L1145
    v634.write(v652);	// L1146
    uint32_t v653 = header2;	// L1147
    v635.write(v653);	// L1148
    uint32_t weight_word1;	// L1149
    weight_word1 = 0;	// L1150
    uint32_t v654 = v633.read();	// L1151
    weight_word1 = v654;	// L1152
    uint32_t v655 = v633.read();	// L1153
    v634.write(v655);	// L1154
    uint32_t v656 = v633.read();	// L1155
    v634.write(v656);	// L1156
    uint32_t v657 = v633.read();	// L1157
    v634.write(v657);	// L1158
    uint32_t v658 = weight_word1;	// L1159
    v635.write(v658);	// L1160
    uint32_t pe_word;	// L1161
    pe_word = 0;	// L1162
    uint32_t v659 = weight_word1;	// L1163
    uint8_t v660;
    ap_int<32> v660_tmp = v659;
    v660 = v660_tmp(7, 0);	// L1164
    uint32_t v661 = pe_word;	// L1165
    int32_t v662;
    ap_int<32> v662_tmp = v661;
    v662_tmp(7, 0) = v660;
    v662 = v662_tmp;	// L1166
    pe_word = v662;	// L1167
    uint32_t v663 = header2;	// L1168
    ac_int<12, false> v664;
    ap_int<32> v664_tmp = v663;
    v664 = v664_tmp(11, 0);	// L1169
    uint32_t v665 = pe_word;	// L1170
    int32_t v666;
    ap_int<32> v666_tmp = v665;
    v666_tmp(19, 8) = v664;
    v666 = v666_tmp;	// L1171
    pe_word = v666;	// L1172
    uint32_t v667 = pe_word;	// L1173
    v636.write(v667);	// L1174
  }
}

void wld_0_1(
  ac_channel< uint32_t >& v668,
  ac_channel< uint32_t >& v669,
  ac_channel< uint32_t >& v670
) {	// L1178
  uint32_t counts_word1;	// L1189
  counts_word1 = 0;	// L1190
  uint32_t v671 = v668.read();	// L1191
  counts_word1 = v671;	// L1192
  uint32_t v672 = counts_word1;	// L1193
  v669.write(v672);	// L1194
  uint32_t v673 = counts_word1;	// L1195
  uint16_t v674;
  ap_int<32> v674_tmp = v673;
  v674 = v674_tmp(15, 0);	// L1196
  int32_t v675 = v674;	// L1197
  int32_t n_mm1;	// L1198
  n_mm1 = v675;	// L1199
  uint32_t trip_word1;	// L1200
  trip_word1 = 0;	// L1201
  uint32_t v676 = counts_word1;	// L1202
  uint16_t v677;
  ap_int<32> v677_tmp = v676;
  v677 = v677_tmp(31, 16);	// L1203
  uint32_t v678 = trip_word1;	// L1204
  int32_t v679;
  ap_int<32> v679_tmp = v678;
  v679_tmp(15, 0) = v677;
  v679 = v679_tmp;	// L1205
  trip_word1 = v679;	// L1206
  uint32_t v680 = trip_word1;	// L1207
  v670.write(v680);	// L1208
  int32_t v681 = n_mm1;	// L1209
  int v682 = v681;	// L1210
  for (int v683 = 0; v683 < v682; v683 += 1) {	// L1211
    uint32_t header3;	// L1212
    header3 = 0;	// L1213
    uint32_t v684 = v668.read();	// L1214
    header3 = v684;	// L1215
    uint32_t v685 = header3;	// L1216
    v669.write(v685);	// L1217
    uint32_t weight_word2;	// L1218
    weight_word2 = 0;	// L1219
    uint32_t v686 = v668.read();	// L1220
    weight_word2 = v686;	// L1221
    uint32_t v687 = weight_word2;	// L1222
    v669.write(v687);	// L1223
    uint32_t pe_word1;	// L1224
    pe_word1 = 0;	// L1225
    uint32_t v688 = weight_word2;	// L1226
    uint8_t v689;
    ap_int<32> v689_tmp = v688;
    v689 = v689_tmp(15, 8);	// L1227
    uint32_t v690 = pe_word1;	// L1228
    int32_t v691;
    ap_int<32> v691_tmp = v690;
    v691_tmp(7, 0) = v689;
    v691 = v691_tmp;	// L1229
    pe_word1 = v691;	// L1230
    uint32_t v692 = header3;	// L1231
    ac_int<12, false> v693;
    ap_int<32> v693_tmp = v692;
    v693 = v693_tmp(11, 0);	// L1232
    uint32_t v694 = pe_word1;	// L1233
    int32_t v695;
    ap_int<32> v695_tmp = v694;
    v695_tmp(19, 8) = v693;
    v695 = v695_tmp;	// L1234
    pe_word1 = v695;	// L1235
    uint32_t v696 = pe_word1;	// L1236
    v670.write(v696);	// L1237
  }
}

void wld_0_2(
  ac_channel< uint32_t >& v697,
  ac_channel< uint32_t >& v698,
  ac_channel< uint32_t >& v699
) {	// L1241
  uint32_t counts_word2;	// L1253
  counts_word2 = 0;	// L1254
  uint32_t v700 = v697.read();	// L1255
  counts_word2 = v700;	// L1256
  uint32_t v701 = counts_word2;	// L1257
  v698.write(v701);	// L1258
  uint32_t v702 = counts_word2;	// L1259
  uint16_t v703;
  ap_int<32> v703_tmp = v702;
  v703 = v703_tmp(15, 0);	// L1260
  int32_t v704 = v703;	// L1261
  int32_t n_mm2;	// L1262
  n_mm2 = v704;	// L1263
  uint32_t trip_word2;	// L1264
  trip_word2 = 0;	// L1265
  uint32_t v705 = counts_word2;	// L1266
  uint16_t v706;
  ap_int<32> v706_tmp = v705;
  v706 = v706_tmp(31, 16);	// L1267
  uint32_t v707 = trip_word2;	// L1268
  int32_t v708;
  ap_int<32> v708_tmp = v707;
  v708_tmp(15, 0) = v706;
  v708 = v708_tmp;	// L1269
  trip_word2 = v708;	// L1270
  uint32_t v709 = trip_word2;	// L1271
  v699.write(v709);	// L1272
  int32_t v710 = n_mm2;	// L1273
  int v711 = v710;	// L1274
  for (int v712 = 0; v712 < v711; v712 += 1) {	// L1275
    uint32_t header4;	// L1276
    header4 = 0;	// L1277
    uint32_t v713 = v697.read();	// L1278
    header4 = v713;	// L1279
    uint32_t v714 = header4;	// L1280
    v698.write(v714);	// L1281
    uint32_t weight_word3;	// L1282
    weight_word3 = 0;	// L1283
    uint32_t v715 = v697.read();	// L1284
    weight_word3 = v715;	// L1285
    uint32_t v716 = weight_word3;	// L1286
    v698.write(v716);	// L1287
    uint32_t pe_word2;	// L1288
    pe_word2 = 0;	// L1289
    uint32_t v717 = weight_word3;	// L1290
    uint8_t v718;
    ap_int<32> v718_tmp = v717;
    v718 = v718_tmp(23, 16);	// L1291
    uint32_t v719 = pe_word2;	// L1292
    int32_t v720;
    ap_int<32> v720_tmp = v719;
    v720_tmp(7, 0) = v718;
    v720 = v720_tmp;	// L1293
    pe_word2 = v720;	// L1294
    uint32_t v721 = header4;	// L1295
    ac_int<12, false> v722;
    ap_int<32> v722_tmp = v721;
    v722 = v722_tmp(11, 0);	// L1296
    uint32_t v723 = pe_word2;	// L1297
    int32_t v724;
    ap_int<32> v724_tmp = v723;
    v724_tmp(19, 8) = v722;
    v724 = v724_tmp;	// L1298
    pe_word2 = v724;	// L1299
    uint32_t v725 = pe_word2;	// L1300
    v699.write(v725);	// L1301
  }
}

void wld_0_3(
  ac_channel< uint32_t >& v726,
  ac_channel< uint32_t >& v727
) {	// L1305
  uint32_t counts_word3;	// L1317
  counts_word3 = 0;	// L1318
  uint32_t v728 = v726.read();	// L1319
  counts_word3 = v728;	// L1320
  uint32_t v729 = counts_word3;	// L1321
  uint16_t v730;
  ap_int<32> v730_tmp = v729;
  v730 = v730_tmp(15, 0);	// L1322
  int32_t v731 = v730;	// L1323
  int32_t n_mm3;	// L1324
  n_mm3 = v731;	// L1325
  uint32_t trip_word3;	// L1326
  trip_word3 = 0;	// L1327
  uint32_t v732 = counts_word3;	// L1328
  uint16_t v733;
  ap_int<32> v733_tmp = v732;
  v733 = v733_tmp(31, 16);	// L1329
  uint32_t v734 = trip_word3;	// L1330
  int32_t v735;
  ap_int<32> v735_tmp = v734;
  v735_tmp(15, 0) = v733;
  v735 = v735_tmp;	// L1331
  trip_word3 = v735;	// L1332
  uint32_t v736 = trip_word3;	// L1333
  v727.write(v736);	// L1334
  int32_t v737 = n_mm3;	// L1335
  int v738 = v737;	// L1336
  for (int v739 = 0; v739 < v738; v739 += 1) {	// L1337
    uint32_t header5;	// L1338
    header5 = 0;	// L1339
    uint32_t v740 = v726.read();	// L1340
    header5 = v740;	// L1341
    uint32_t weight_word4;	// L1342
    weight_word4 = 0;	// L1343
    uint32_t v741 = v726.read();	// L1344
    weight_word4 = v741;	// L1345
    uint32_t pe_word3;	// L1346
    pe_word3 = 0;	// L1347
    uint32_t v742 = weight_word4;	// L1348
    uint8_t v743;
    ap_int<32> v743_tmp = v742;
    v743 = v743_tmp(31, 24);	// L1349
    uint32_t v744 = pe_word3;	// L1350
    int32_t v745;
    ap_int<32> v745_tmp = v744;
    v745_tmp(7, 0) = v743;
    v745 = v745_tmp;	// L1351
    pe_word3 = v745;	// L1352
    uint32_t v746 = header5;	// L1353
    ac_int<12, false> v747;
    ap_int<32> v747_tmp = v746;
    v747 = v747_tmp(11, 0);	// L1354
    uint32_t v748 = pe_word3;	// L1355
    int32_t v749;
    ap_int<32> v749_tmp = v748;
    v749_tmp(19, 8) = v747;
    v749 = v749_tmp;	// L1356
    pe_word3 = v749;	// L1357
    uint32_t v750 = pe_word3;	// L1358
    v727.write(v750);	// L1359
  }
}

void wld_1_0(
  ac_channel< uint32_t >& v751,
  ac_channel< uint32_t >& v752,
  ac_channel< uint32_t >& v753,
  ac_channel< uint32_t >& v754
) {	// L1363
  uint32_t counts_word4;	// L1374
  counts_word4 = 0;	// L1375
  uint32_t v755 = v751.read();	// L1376
  counts_word4 = v755;	// L1377
  uint32_t v756 = counts_word4;	// L1378
  v752.write(v756);	// L1379
  uint32_t v757 = counts_word4;	// L1380
  v753.write(v757);	// L1381
  uint32_t v758 = counts_word4;	// L1382
  uint16_t v759;
  ap_int<32> v759_tmp = v758;
  v759 = v759_tmp(15, 0);	// L1383
  int32_t v760 = v759;	// L1384
  int32_t n_mm4;	// L1385
  n_mm4 = v760;	// L1386
  uint32_t trip_word4;	// L1387
  trip_word4 = 0;	// L1388
  uint32_t v761 = counts_word4;	// L1389
  uint16_t v762;
  ap_int<32> v762_tmp = v761;
  v762 = v762_tmp(31, 16);	// L1390
  uint32_t v763 = trip_word4;	// L1391
  int32_t v764;
  ap_int<32> v764_tmp = v763;
  v764_tmp(15, 0) = v762;
  v764 = v764_tmp;	// L1392
  trip_word4 = v764;	// L1393
  uint32_t v765 = trip_word4;	// L1394
  v754.write(v765);	// L1395
  int32_t v766 = n_mm4;	// L1396
  int v767 = v766;	// L1397
  for (int v768 = 0; v768 < v767; v768 += 1) {	// L1398
    uint32_t header6;	// L1399
    header6 = 0;	// L1400
    uint32_t v769 = v751.read();	// L1401
    header6 = v769;	// L1402
    uint32_t v770 = header6;	// L1403
    v752.write(v770);	// L1404
    uint32_t v771 = header6;	// L1405
    v753.write(v771);	// L1406
    uint32_t weight_word5;	// L1407
    weight_word5 = 0;	// L1408
    uint32_t v772 = v751.read();	// L1409
    weight_word5 = v772;	// L1410
    uint32_t v773 = v751.read();	// L1411
    v752.write(v773);	// L1412
    uint32_t v774 = v751.read();	// L1413
    v752.write(v774);	// L1414
    uint32_t v775 = weight_word5;	// L1415
    v753.write(v775);	// L1416
    uint32_t pe_word4;	// L1417
    pe_word4 = 0;	// L1418
    uint32_t v776 = weight_word5;	// L1419
    uint8_t v777;
    ap_int<32> v777_tmp = v776;
    v777 = v777_tmp(7, 0);	// L1420
    uint32_t v778 = pe_word4;	// L1421
    int32_t v779;
    ap_int<32> v779_tmp = v778;
    v779_tmp(7, 0) = v777;
    v779 = v779_tmp;	// L1422
    pe_word4 = v779;	// L1423
    uint32_t v780 = header6;	// L1424
    ac_int<12, false> v781;
    ap_int<32> v781_tmp = v780;
    v781 = v781_tmp(11, 0);	// L1425
    uint32_t v782 = pe_word4;	// L1426
    int32_t v783;
    ap_int<32> v783_tmp = v782;
    v783_tmp(19, 8) = v781;
    v783 = v783_tmp;	// L1427
    pe_word4 = v783;	// L1428
    uint32_t v784 = pe_word4;	// L1429
    v754.write(v784);	// L1430
  }
}

void wld_1_1(
  ac_channel< uint32_t >& v785,
  ac_channel< uint32_t >& v786,
  ac_channel< uint32_t >& v787
) {	// L1434
  uint32_t counts_word5;	// L1445
  counts_word5 = 0;	// L1446
  uint32_t v788 = v785.read();	// L1447
  counts_word5 = v788;	// L1448
  uint32_t v789 = counts_word5;	// L1449
  v786.write(v789);	// L1450
  uint32_t v790 = counts_word5;	// L1451
  uint16_t v791;
  ap_int<32> v791_tmp = v790;
  v791 = v791_tmp(15, 0);	// L1452
  int32_t v792 = v791;	// L1453
  int32_t n_mm5;	// L1454
  n_mm5 = v792;	// L1455
  uint32_t trip_word5;	// L1456
  trip_word5 = 0;	// L1457
  uint32_t v793 = counts_word5;	// L1458
  uint16_t v794;
  ap_int<32> v794_tmp = v793;
  v794 = v794_tmp(31, 16);	// L1459
  uint32_t v795 = trip_word5;	// L1460
  int32_t v796;
  ap_int<32> v796_tmp = v795;
  v796_tmp(15, 0) = v794;
  v796 = v796_tmp;	// L1461
  trip_word5 = v796;	// L1462
  uint32_t v797 = trip_word5;	// L1463
  v787.write(v797);	// L1464
  int32_t v798 = n_mm5;	// L1465
  int v799 = v798;	// L1466
  for (int v800 = 0; v800 < v799; v800 += 1) {	// L1467
    uint32_t header7;	// L1468
    header7 = 0;	// L1469
    uint32_t v801 = v785.read();	// L1470
    header7 = v801;	// L1471
    uint32_t v802 = header7;	// L1472
    v786.write(v802);	// L1473
    uint32_t weight_word6;	// L1474
    weight_word6 = 0;	// L1475
    uint32_t v803 = v785.read();	// L1476
    weight_word6 = v803;	// L1477
    uint32_t v804 = weight_word6;	// L1478
    v786.write(v804);	// L1479
    uint32_t pe_word5;	// L1480
    pe_word5 = 0;	// L1481
    uint32_t v805 = weight_word6;	// L1482
    uint8_t v806;
    ap_int<32> v806_tmp = v805;
    v806 = v806_tmp(15, 8);	// L1483
    uint32_t v807 = pe_word5;	// L1484
    int32_t v808;
    ap_int<32> v808_tmp = v807;
    v808_tmp(7, 0) = v806;
    v808 = v808_tmp;	// L1485
    pe_word5 = v808;	// L1486
    uint32_t v809 = header7;	// L1487
    ac_int<12, false> v810;
    ap_int<32> v810_tmp = v809;
    v810 = v810_tmp(11, 0);	// L1488
    uint32_t v811 = pe_word5;	// L1489
    int32_t v812;
    ap_int<32> v812_tmp = v811;
    v812_tmp(19, 8) = v810;
    v812 = v812_tmp;	// L1490
    pe_word5 = v812;	// L1491
    uint32_t v813 = pe_word5;	// L1492
    v787.write(v813);	// L1493
  }
}

void wld_1_2(
  ac_channel< uint32_t >& v814,
  ac_channel< uint32_t >& v815,
  ac_channel< uint32_t >& v816
) {	// L1497
  uint32_t counts_word6;	// L1509
  counts_word6 = 0;	// L1510
  uint32_t v817 = v814.read();	// L1511
  counts_word6 = v817;	// L1512
  uint32_t v818 = counts_word6;	// L1513
  v815.write(v818);	// L1514
  uint32_t v819 = counts_word6;	// L1515
  uint16_t v820;
  ap_int<32> v820_tmp = v819;
  v820 = v820_tmp(15, 0);	// L1516
  int32_t v821 = v820;	// L1517
  int32_t n_mm6;	// L1518
  n_mm6 = v821;	// L1519
  uint32_t trip_word6;	// L1520
  trip_word6 = 0;	// L1521
  uint32_t v822 = counts_word6;	// L1522
  uint16_t v823;
  ap_int<32> v823_tmp = v822;
  v823 = v823_tmp(31, 16);	// L1523
  uint32_t v824 = trip_word6;	// L1524
  int32_t v825;
  ap_int<32> v825_tmp = v824;
  v825_tmp(15, 0) = v823;
  v825 = v825_tmp;	// L1525
  trip_word6 = v825;	// L1526
  uint32_t v826 = trip_word6;	// L1527
  v816.write(v826);	// L1528
  int32_t v827 = n_mm6;	// L1529
  int v828 = v827;	// L1530
  for (int v829 = 0; v829 < v828; v829 += 1) {	// L1531
    uint32_t header8;	// L1532
    header8 = 0;	// L1533
    uint32_t v830 = v814.read();	// L1534
    header8 = v830;	// L1535
    uint32_t v831 = header8;	// L1536
    v815.write(v831);	// L1537
    uint32_t weight_word7;	// L1538
    weight_word7 = 0;	// L1539
    uint32_t v832 = v814.read();	// L1540
    weight_word7 = v832;	// L1541
    uint32_t v833 = weight_word7;	// L1542
    v815.write(v833);	// L1543
    uint32_t pe_word6;	// L1544
    pe_word6 = 0;	// L1545
    uint32_t v834 = weight_word7;	// L1546
    uint8_t v835;
    ap_int<32> v835_tmp = v834;
    v835 = v835_tmp(23, 16);	// L1547
    uint32_t v836 = pe_word6;	// L1548
    int32_t v837;
    ap_int<32> v837_tmp = v836;
    v837_tmp(7, 0) = v835;
    v837 = v837_tmp;	// L1549
    pe_word6 = v837;	// L1550
    uint32_t v838 = header8;	// L1551
    ac_int<12, false> v839;
    ap_int<32> v839_tmp = v838;
    v839 = v839_tmp(11, 0);	// L1552
    uint32_t v840 = pe_word6;	// L1553
    int32_t v841;
    ap_int<32> v841_tmp = v840;
    v841_tmp(19, 8) = v839;
    v841 = v841_tmp;	// L1554
    pe_word6 = v841;	// L1555
    uint32_t v842 = pe_word6;	// L1556
    v816.write(v842);	// L1557
  }
}

void wld_1_3(
  ac_channel< uint32_t >& v843,
  ac_channel< uint32_t >& v844
) {	// L1561
  uint32_t counts_word7;	// L1573
  counts_word7 = 0;	// L1574
  uint32_t v845 = v843.read();	// L1575
  counts_word7 = v845;	// L1576
  uint32_t v846 = counts_word7;	// L1577
  uint16_t v847;
  ap_int<32> v847_tmp = v846;
  v847 = v847_tmp(15, 0);	// L1578
  int32_t v848 = v847;	// L1579
  int32_t n_mm7;	// L1580
  n_mm7 = v848;	// L1581
  uint32_t trip_word7;	// L1582
  trip_word7 = 0;	// L1583
  uint32_t v849 = counts_word7;	// L1584
  uint16_t v850;
  ap_int<32> v850_tmp = v849;
  v850 = v850_tmp(31, 16);	// L1585
  uint32_t v851 = trip_word7;	// L1586
  int32_t v852;
  ap_int<32> v852_tmp = v851;
  v852_tmp(15, 0) = v850;
  v852 = v852_tmp;	// L1587
  trip_word7 = v852;	// L1588
  uint32_t v853 = trip_word7;	// L1589
  v844.write(v853);	// L1590
  int32_t v854 = n_mm7;	// L1591
  int v855 = v854;	// L1592
  for (int v856 = 0; v856 < v855; v856 += 1) {	// L1593
    uint32_t header9;	// L1594
    header9 = 0;	// L1595
    uint32_t v857 = v843.read();	// L1596
    header9 = v857;	// L1597
    uint32_t weight_word8;	// L1598
    weight_word8 = 0;	// L1599
    uint32_t v858 = v843.read();	// L1600
    weight_word8 = v858;	// L1601
    uint32_t pe_word7;	// L1602
    pe_word7 = 0;	// L1603
    uint32_t v859 = weight_word8;	// L1604
    uint8_t v860;
    ap_int<32> v860_tmp = v859;
    v860 = v860_tmp(31, 24);	// L1605
    uint32_t v861 = pe_word7;	// L1606
    int32_t v862;
    ap_int<32> v862_tmp = v861;
    v862_tmp(7, 0) = v860;
    v862 = v862_tmp;	// L1607
    pe_word7 = v862;	// L1608
    uint32_t v863 = header9;	// L1609
    ac_int<12, false> v864;
    ap_int<32> v864_tmp = v863;
    v864 = v864_tmp(11, 0);	// L1610
    uint32_t v865 = pe_word7;	// L1611
    int32_t v866;
    ap_int<32> v866_tmp = v865;
    v866_tmp(19, 8) = v864;
    v866 = v866_tmp;	// L1612
    pe_word7 = v866;	// L1613
    uint32_t v867 = pe_word7;	// L1614
    v844.write(v867);	// L1615
  }
}

void wld_2_0(
  ac_channel< uint32_t >& v868,
  ac_channel< uint32_t >& v869,
  ac_channel< uint32_t >& v870,
  ac_channel< uint32_t >& v871
) {	// L1619
  uint32_t counts_word8;	// L1630
  counts_word8 = 0;	// L1631
  uint32_t v872 = v868.read();	// L1632
  counts_word8 = v872;	// L1633
  uint32_t v873 = counts_word8;	// L1634
  v869.write(v873);	// L1635
  uint32_t v874 = counts_word8;	// L1636
  v870.write(v874);	// L1637
  uint32_t v875 = counts_word8;	// L1638
  uint16_t v876;
  ap_int<32> v876_tmp = v875;
  v876 = v876_tmp(15, 0);	// L1639
  int32_t v877 = v876;	// L1640
  int32_t n_mm8;	// L1641
  n_mm8 = v877;	// L1642
  uint32_t trip_word8;	// L1643
  trip_word8 = 0;	// L1644
  uint32_t v878 = counts_word8;	// L1645
  uint16_t v879;
  ap_int<32> v879_tmp = v878;
  v879 = v879_tmp(31, 16);	// L1646
  uint32_t v880 = trip_word8;	// L1647
  int32_t v881;
  ap_int<32> v881_tmp = v880;
  v881_tmp(15, 0) = v879;
  v881 = v881_tmp;	// L1648
  trip_word8 = v881;	// L1649
  uint32_t v882 = trip_word8;	// L1650
  v871.write(v882);	// L1651
  int32_t v883 = n_mm8;	// L1652
  int v884 = v883;	// L1653
  for (int v885 = 0; v885 < v884; v885 += 1) {	// L1654
    uint32_t header10;	// L1655
    header10 = 0;	// L1656
    uint32_t v886 = v868.read();	// L1657
    header10 = v886;	// L1658
    uint32_t v887 = header10;	// L1659
    v869.write(v887);	// L1660
    uint32_t v888 = header10;	// L1661
    v870.write(v888);	// L1662
    uint32_t weight_word9;	// L1663
    weight_word9 = 0;	// L1664
    uint32_t v889 = v868.read();	// L1665
    weight_word9 = v889;	// L1666
    uint32_t v890 = v868.read();	// L1667
    v869.write(v890);	// L1668
    uint32_t v891 = weight_word9;	// L1669
    v870.write(v891);	// L1670
    uint32_t pe_word8;	// L1671
    pe_word8 = 0;	// L1672
    uint32_t v892 = weight_word9;	// L1673
    uint8_t v893;
    ap_int<32> v893_tmp = v892;
    v893 = v893_tmp(7, 0);	// L1674
    uint32_t v894 = pe_word8;	// L1675
    int32_t v895;
    ap_int<32> v895_tmp = v894;
    v895_tmp(7, 0) = v893;
    v895 = v895_tmp;	// L1676
    pe_word8 = v895;	// L1677
    uint32_t v896 = header10;	// L1678
    ac_int<12, false> v897;
    ap_int<32> v897_tmp = v896;
    v897 = v897_tmp(11, 0);	// L1679
    uint32_t v898 = pe_word8;	// L1680
    int32_t v899;
    ap_int<32> v899_tmp = v898;
    v899_tmp(19, 8) = v897;
    v899 = v899_tmp;	// L1681
    pe_word8 = v899;	// L1682
    uint32_t v900 = pe_word8;	// L1683
    v871.write(v900);	// L1684
  }
}

void wld_2_1(
  ac_channel< uint32_t >& v901,
  ac_channel< uint32_t >& v902,
  ac_channel< uint32_t >& v903
) {	// L1688
  uint32_t counts_word9;	// L1699
  counts_word9 = 0;	// L1700
  uint32_t v904 = v901.read();	// L1701
  counts_word9 = v904;	// L1702
  uint32_t v905 = counts_word9;	// L1703
  v902.write(v905);	// L1704
  uint32_t v906 = counts_word9;	// L1705
  uint16_t v907;
  ap_int<32> v907_tmp = v906;
  v907 = v907_tmp(15, 0);	// L1706
  int32_t v908 = v907;	// L1707
  int32_t n_mm9;	// L1708
  n_mm9 = v908;	// L1709
  uint32_t trip_word9;	// L1710
  trip_word9 = 0;	// L1711
  uint32_t v909 = counts_word9;	// L1712
  uint16_t v910;
  ap_int<32> v910_tmp = v909;
  v910 = v910_tmp(31, 16);	// L1713
  uint32_t v911 = trip_word9;	// L1714
  int32_t v912;
  ap_int<32> v912_tmp = v911;
  v912_tmp(15, 0) = v910;
  v912 = v912_tmp;	// L1715
  trip_word9 = v912;	// L1716
  uint32_t v913 = trip_word9;	// L1717
  v903.write(v913);	// L1718
  int32_t v914 = n_mm9;	// L1719
  int v915 = v914;	// L1720
  for (int v916 = 0; v916 < v915; v916 += 1) {	// L1721
    uint32_t header11;	// L1722
    header11 = 0;	// L1723
    uint32_t v917 = v901.read();	// L1724
    header11 = v917;	// L1725
    uint32_t v918 = header11;	// L1726
    v902.write(v918);	// L1727
    uint32_t weight_word10;	// L1728
    weight_word10 = 0;	// L1729
    uint32_t v919 = v901.read();	// L1730
    weight_word10 = v919;	// L1731
    uint32_t v920 = weight_word10;	// L1732
    v902.write(v920);	// L1733
    uint32_t pe_word9;	// L1734
    pe_word9 = 0;	// L1735
    uint32_t v921 = weight_word10;	// L1736
    uint8_t v922;
    ap_int<32> v922_tmp = v921;
    v922 = v922_tmp(15, 8);	// L1737
    uint32_t v923 = pe_word9;	// L1738
    int32_t v924;
    ap_int<32> v924_tmp = v923;
    v924_tmp(7, 0) = v922;
    v924 = v924_tmp;	// L1739
    pe_word9 = v924;	// L1740
    uint32_t v925 = header11;	// L1741
    ac_int<12, false> v926;
    ap_int<32> v926_tmp = v925;
    v926 = v926_tmp(11, 0);	// L1742
    uint32_t v927 = pe_word9;	// L1743
    int32_t v928;
    ap_int<32> v928_tmp = v927;
    v928_tmp(19, 8) = v926;
    v928 = v928_tmp;	// L1744
    pe_word9 = v928;	// L1745
    uint32_t v929 = pe_word9;	// L1746
    v903.write(v929);	// L1747
  }
}

void wld_2_2(
  ac_channel< uint32_t >& v930,
  ac_channel< uint32_t >& v931,
  ac_channel< uint32_t >& v932
) {	// L1751
  uint32_t counts_word10;	// L1763
  counts_word10 = 0;	// L1764
  uint32_t v933 = v930.read();	// L1765
  counts_word10 = v933;	// L1766
  uint32_t v934 = counts_word10;	// L1767
  v931.write(v934);	// L1768
  uint32_t v935 = counts_word10;	// L1769
  uint16_t v936;
  ap_int<32> v936_tmp = v935;
  v936 = v936_tmp(15, 0);	// L1770
  int32_t v937 = v936;	// L1771
  int32_t n_mm10;	// L1772
  n_mm10 = v937;	// L1773
  uint32_t trip_word10;	// L1774
  trip_word10 = 0;	// L1775
  uint32_t v938 = counts_word10;	// L1776
  uint16_t v939;
  ap_int<32> v939_tmp = v938;
  v939 = v939_tmp(31, 16);	// L1777
  uint32_t v940 = trip_word10;	// L1778
  int32_t v941;
  ap_int<32> v941_tmp = v940;
  v941_tmp(15, 0) = v939;
  v941 = v941_tmp;	// L1779
  trip_word10 = v941;	// L1780
  uint32_t v942 = trip_word10;	// L1781
  v932.write(v942);	// L1782
  int32_t v943 = n_mm10;	// L1783
  int v944 = v943;	// L1784
  for (int v945 = 0; v945 < v944; v945 += 1) {	// L1785
    uint32_t header12;	// L1786
    header12 = 0;	// L1787
    uint32_t v946 = v930.read();	// L1788
    header12 = v946;	// L1789
    uint32_t v947 = header12;	// L1790
    v931.write(v947);	// L1791
    uint32_t weight_word11;	// L1792
    weight_word11 = 0;	// L1793
    uint32_t v948 = v930.read();	// L1794
    weight_word11 = v948;	// L1795
    uint32_t v949 = weight_word11;	// L1796
    v931.write(v949);	// L1797
    uint32_t pe_word10;	// L1798
    pe_word10 = 0;	// L1799
    uint32_t v950 = weight_word11;	// L1800
    uint8_t v951;
    ap_int<32> v951_tmp = v950;
    v951 = v951_tmp(23, 16);	// L1801
    uint32_t v952 = pe_word10;	// L1802
    int32_t v953;
    ap_int<32> v953_tmp = v952;
    v953_tmp(7, 0) = v951;
    v953 = v953_tmp;	// L1803
    pe_word10 = v953;	// L1804
    uint32_t v954 = header12;	// L1805
    ac_int<12, false> v955;
    ap_int<32> v955_tmp = v954;
    v955 = v955_tmp(11, 0);	// L1806
    uint32_t v956 = pe_word10;	// L1807
    int32_t v957;
    ap_int<32> v957_tmp = v956;
    v957_tmp(19, 8) = v955;
    v957 = v957_tmp;	// L1808
    pe_word10 = v957;	// L1809
    uint32_t v958 = pe_word10;	// L1810
    v932.write(v958);	// L1811
  }
}

void wld_2_3(
  ac_channel< uint32_t >& v959,
  ac_channel< uint32_t >& v960
) {	// L1815
  uint32_t counts_word11;	// L1827
  counts_word11 = 0;	// L1828
  uint32_t v961 = v959.read();	// L1829
  counts_word11 = v961;	// L1830
  uint32_t v962 = counts_word11;	// L1831
  uint16_t v963;
  ap_int<32> v963_tmp = v962;
  v963 = v963_tmp(15, 0);	// L1832
  int32_t v964 = v963;	// L1833
  int32_t n_mm11;	// L1834
  n_mm11 = v964;	// L1835
  uint32_t trip_word11;	// L1836
  trip_word11 = 0;	// L1837
  uint32_t v965 = counts_word11;	// L1838
  uint16_t v966;
  ap_int<32> v966_tmp = v965;
  v966 = v966_tmp(31, 16);	// L1839
  uint32_t v967 = trip_word11;	// L1840
  int32_t v968;
  ap_int<32> v968_tmp = v967;
  v968_tmp(15, 0) = v966;
  v968 = v968_tmp;	// L1841
  trip_word11 = v968;	// L1842
  uint32_t v969 = trip_word11;	// L1843
  v960.write(v969);	// L1844
  int32_t v970 = n_mm11;	// L1845
  int v971 = v970;	// L1846
  for (int v972 = 0; v972 < v971; v972 += 1) {	// L1847
    uint32_t header13;	// L1848
    header13 = 0;	// L1849
    uint32_t v973 = v959.read();	// L1850
    header13 = v973;	// L1851
    uint32_t weight_word12;	// L1852
    weight_word12 = 0;	// L1853
    uint32_t v974 = v959.read();	// L1854
    weight_word12 = v974;	// L1855
    uint32_t pe_word11;	// L1856
    pe_word11 = 0;	// L1857
    uint32_t v975 = weight_word12;	// L1858
    uint8_t v976;
    ap_int<32> v976_tmp = v975;
    v976 = v976_tmp(31, 24);	// L1859
    uint32_t v977 = pe_word11;	// L1860
    int32_t v978;
    ap_int<32> v978_tmp = v977;
    v978_tmp(7, 0) = v976;
    v978 = v978_tmp;	// L1861
    pe_word11 = v978;	// L1862
    uint32_t v979 = header13;	// L1863
    ac_int<12, false> v980;
    ap_int<32> v980_tmp = v979;
    v980 = v980_tmp(11, 0);	// L1864
    uint32_t v981 = pe_word11;	// L1865
    int32_t v982;
    ap_int<32> v982_tmp = v981;
    v982_tmp(19, 8) = v980;
    v982 = v982_tmp;	// L1866
    pe_word11 = v982;	// L1867
    uint32_t v983 = pe_word11;	// L1868
    v960.write(v983);	// L1869
  }
}

void wld_3_0(
  ac_channel< uint32_t >& v984,
  ac_channel< uint32_t >& v985,
  ac_channel< uint32_t >& v986
) {	// L1873
  uint32_t counts_word12;	// L1884
  counts_word12 = 0;	// L1885
  uint32_t v987 = v984.read();	// L1886
  counts_word12 = v987;	// L1887
  uint32_t v988 = counts_word12;	// L1888
  v985.write(v988);	// L1889
  uint32_t v989 = counts_word12;	// L1890
  uint16_t v990;
  ap_int<32> v990_tmp = v989;
  v990 = v990_tmp(15, 0);	// L1891
  int32_t v991 = v990;	// L1892
  int32_t n_mm12;	// L1893
  n_mm12 = v991;	// L1894
  uint32_t trip_word12;	// L1895
  trip_word12 = 0;	// L1896
  uint32_t v992 = counts_word12;	// L1897
  uint16_t v993;
  ap_int<32> v993_tmp = v992;
  v993 = v993_tmp(31, 16);	// L1898
  uint32_t v994 = trip_word12;	// L1899
  int32_t v995;
  ap_int<32> v995_tmp = v994;
  v995_tmp(15, 0) = v993;
  v995 = v995_tmp;	// L1900
  trip_word12 = v995;	// L1901
  uint32_t v996 = trip_word12;	// L1902
  v986.write(v996);	// L1903
  int32_t v997 = n_mm12;	// L1904
  int v998 = v997;	// L1905
  for (int v999 = 0; v999 < v998; v999 += 1) {	// L1906
    uint32_t header14;	// L1907
    header14 = 0;	// L1908
    uint32_t v1000 = v984.read();	// L1909
    header14 = v1000;	// L1910
    uint32_t v1001 = header14;	// L1911
    v985.write(v1001);	// L1912
    uint32_t weight_word13;	// L1913
    weight_word13 = 0;	// L1914
    uint32_t v1002 = v984.read();	// L1915
    weight_word13 = v1002;	// L1916
    uint32_t v1003 = weight_word13;	// L1917
    v985.write(v1003);	// L1918
    uint32_t pe_word12;	// L1919
    pe_word12 = 0;	// L1920
    uint32_t v1004 = weight_word13;	// L1921
    uint8_t v1005;
    ap_int<32> v1005_tmp = v1004;
    v1005 = v1005_tmp(7, 0);	// L1922
    uint32_t v1006 = pe_word12;	// L1923
    int32_t v1007;
    ap_int<32> v1007_tmp = v1006;
    v1007_tmp(7, 0) = v1005;
    v1007 = v1007_tmp;	// L1924
    pe_word12 = v1007;	// L1925
    uint32_t v1008 = header14;	// L1926
    ac_int<12, false> v1009;
    ap_int<32> v1009_tmp = v1008;
    v1009 = v1009_tmp(11, 0);	// L1927
    uint32_t v1010 = pe_word12;	// L1928
    int32_t v1011;
    ap_int<32> v1011_tmp = v1010;
    v1011_tmp(19, 8) = v1009;
    v1011 = v1011_tmp;	// L1929
    pe_word12 = v1011;	// L1930
    uint32_t v1012 = pe_word12;	// L1931
    v986.write(v1012);	// L1932
  }
}

void wld_3_1(
  ac_channel< uint32_t >& v1013,
  ac_channel< uint32_t >& v1014,
  ac_channel< uint32_t >& v1015
) {	// L1936
  uint32_t counts_word13;	// L1947
  counts_word13 = 0;	// L1948
  uint32_t v1016 = v1013.read();	// L1949
  counts_word13 = v1016;	// L1950
  uint32_t v1017 = counts_word13;	// L1951
  v1014.write(v1017);	// L1952
  uint32_t v1018 = counts_word13;	// L1953
  uint16_t v1019;
  ap_int<32> v1019_tmp = v1018;
  v1019 = v1019_tmp(15, 0);	// L1954
  int32_t v1020 = v1019;	// L1955
  int32_t n_mm13;	// L1956
  n_mm13 = v1020;	// L1957
  uint32_t trip_word13;	// L1958
  trip_word13 = 0;	// L1959
  uint32_t v1021 = counts_word13;	// L1960
  uint16_t v1022;
  ap_int<32> v1022_tmp = v1021;
  v1022 = v1022_tmp(31, 16);	// L1961
  uint32_t v1023 = trip_word13;	// L1962
  int32_t v1024;
  ap_int<32> v1024_tmp = v1023;
  v1024_tmp(15, 0) = v1022;
  v1024 = v1024_tmp;	// L1963
  trip_word13 = v1024;	// L1964
  uint32_t v1025 = trip_word13;	// L1965
  v1015.write(v1025);	// L1966
  int32_t v1026 = n_mm13;	// L1967
  int v1027 = v1026;	// L1968
  for (int v1028 = 0; v1028 < v1027; v1028 += 1) {	// L1969
    uint32_t header15;	// L1970
    header15 = 0;	// L1971
    uint32_t v1029 = v1013.read();	// L1972
    header15 = v1029;	// L1973
    uint32_t v1030 = header15;	// L1974
    v1014.write(v1030);	// L1975
    uint32_t weight_word14;	// L1976
    weight_word14 = 0;	// L1977
    uint32_t v1031 = v1013.read();	// L1978
    weight_word14 = v1031;	// L1979
    uint32_t v1032 = weight_word14;	// L1980
    v1014.write(v1032);	// L1981
    uint32_t pe_word13;	// L1982
    pe_word13 = 0;	// L1983
    uint32_t v1033 = weight_word14;	// L1984
    uint8_t v1034;
    ap_int<32> v1034_tmp = v1033;
    v1034 = v1034_tmp(15, 8);	// L1985
    uint32_t v1035 = pe_word13;	// L1986
    int32_t v1036;
    ap_int<32> v1036_tmp = v1035;
    v1036_tmp(7, 0) = v1034;
    v1036 = v1036_tmp;	// L1987
    pe_word13 = v1036;	// L1988
    uint32_t v1037 = header15;	// L1989
    ac_int<12, false> v1038;
    ap_int<32> v1038_tmp = v1037;
    v1038 = v1038_tmp(11, 0);	// L1990
    uint32_t v1039 = pe_word13;	// L1991
    int32_t v1040;
    ap_int<32> v1040_tmp = v1039;
    v1040_tmp(19, 8) = v1038;
    v1040 = v1040_tmp;	// L1992
    pe_word13 = v1040;	// L1993
    uint32_t v1041 = pe_word13;	// L1994
    v1015.write(v1041);	// L1995
  }
}

void wld_3_2(
  ac_channel< uint32_t >& v1042,
  ac_channel< uint32_t >& v1043,
  ac_channel< uint32_t >& v1044
) {	// L1999
  uint32_t counts_word14;	// L2011
  counts_word14 = 0;	// L2012
  uint32_t v1045 = v1042.read();	// L2013
  counts_word14 = v1045;	// L2014
  uint32_t v1046 = counts_word14;	// L2015
  v1043.write(v1046);	// L2016
  uint32_t v1047 = counts_word14;	// L2017
  uint16_t v1048;
  ap_int<32> v1048_tmp = v1047;
  v1048 = v1048_tmp(15, 0);	// L2018
  int32_t v1049 = v1048;	// L2019
  int32_t n_mm14;	// L2020
  n_mm14 = v1049;	// L2021
  uint32_t trip_word14;	// L2022
  trip_word14 = 0;	// L2023
  uint32_t v1050 = counts_word14;	// L2024
  uint16_t v1051;
  ap_int<32> v1051_tmp = v1050;
  v1051 = v1051_tmp(31, 16);	// L2025
  uint32_t v1052 = trip_word14;	// L2026
  int32_t v1053;
  ap_int<32> v1053_tmp = v1052;
  v1053_tmp(15, 0) = v1051;
  v1053 = v1053_tmp;	// L2027
  trip_word14 = v1053;	// L2028
  uint32_t v1054 = trip_word14;	// L2029
  v1044.write(v1054);	// L2030
  int32_t v1055 = n_mm14;	// L2031
  int v1056 = v1055;	// L2032
  for (int v1057 = 0; v1057 < v1056; v1057 += 1) {	// L2033
    uint32_t header16;	// L2034
    header16 = 0;	// L2035
    uint32_t v1058 = v1042.read();	// L2036
    header16 = v1058;	// L2037
    uint32_t v1059 = header16;	// L2038
    v1043.write(v1059);	// L2039
    uint32_t weight_word15;	// L2040
    weight_word15 = 0;	// L2041
    uint32_t v1060 = v1042.read();	// L2042
    weight_word15 = v1060;	// L2043
    uint32_t v1061 = weight_word15;	// L2044
    v1043.write(v1061);	// L2045
    uint32_t pe_word14;	// L2046
    pe_word14 = 0;	// L2047
    uint32_t v1062 = weight_word15;	// L2048
    uint8_t v1063;
    ap_int<32> v1063_tmp = v1062;
    v1063 = v1063_tmp(23, 16);	// L2049
    uint32_t v1064 = pe_word14;	// L2050
    int32_t v1065;
    ap_int<32> v1065_tmp = v1064;
    v1065_tmp(7, 0) = v1063;
    v1065 = v1065_tmp;	// L2051
    pe_word14 = v1065;	// L2052
    uint32_t v1066 = header16;	// L2053
    ac_int<12, false> v1067;
    ap_int<32> v1067_tmp = v1066;
    v1067 = v1067_tmp(11, 0);	// L2054
    uint32_t v1068 = pe_word14;	// L2055
    int32_t v1069;
    ap_int<32> v1069_tmp = v1068;
    v1069_tmp(19, 8) = v1067;
    v1069 = v1069_tmp;	// L2056
    pe_word14 = v1069;	// L2057
    uint32_t v1070 = pe_word14;	// L2058
    v1044.write(v1070);	// L2059
  }
}

void wld_3_3(
  ac_channel< uint32_t >& v1071,
  ac_channel< uint32_t >& v1072
) {	// L2063
  uint32_t counts_word15;	// L2075
  counts_word15 = 0;	// L2076
  uint32_t v1073 = v1071.read();	// L2077
  counts_word15 = v1073;	// L2078
  uint32_t v1074 = counts_word15;	// L2079
  uint16_t v1075;
  ap_int<32> v1075_tmp = v1074;
  v1075 = v1075_tmp(15, 0);	// L2080
  int32_t v1076 = v1075;	// L2081
  int32_t n_mm15;	// L2082
  n_mm15 = v1076;	// L2083
  uint32_t trip_word15;	// L2084
  trip_word15 = 0;	// L2085
  uint32_t v1077 = counts_word15;	// L2086
  uint16_t v1078;
  ap_int<32> v1078_tmp = v1077;
  v1078 = v1078_tmp(31, 16);	// L2087
  uint32_t v1079 = trip_word15;	// L2088
  int32_t v1080;
  ap_int<32> v1080_tmp = v1079;
  v1080_tmp(15, 0) = v1078;
  v1080 = v1080_tmp;	// L2089
  trip_word15 = v1080;	// L2090
  uint32_t v1081 = trip_word15;	// L2091
  v1072.write(v1081);	// L2092
  int32_t v1082 = n_mm15;	// L2093
  int v1083 = v1082;	// L2094
  for (int v1084 = 0; v1084 < v1083; v1084 += 1) {	// L2095
    uint32_t header17;	// L2096
    header17 = 0;	// L2097
    uint32_t v1085 = v1071.read();	// L2098
    header17 = v1085;	// L2099
    uint32_t weight_word16;	// L2100
    weight_word16 = 0;	// L2101
    uint32_t v1086 = v1071.read();	// L2102
    weight_word16 = v1086;	// L2103
    uint32_t pe_word15;	// L2104
    pe_word15 = 0;	// L2105
    uint32_t v1087 = weight_word16;	// L2106
    uint8_t v1088;
    ap_int<32> v1088_tmp = v1087;
    v1088 = v1088_tmp(31, 24);	// L2107
    uint32_t v1089 = pe_word15;	// L2108
    int32_t v1090;
    ap_int<32> v1090_tmp = v1089;
    v1090_tmp(7, 0) = v1088;
    v1090 = v1090_tmp;	// L2109
    pe_word15 = v1090;	// L2110
    uint32_t v1091 = header17;	// L2111
    ac_int<12, false> v1092;
    ap_int<32> v1092_tmp = v1091;
    v1092 = v1092_tmp(11, 0);	// L2112
    uint32_t v1093 = pe_word15;	// L2113
    int32_t v1094;
    ap_int<32> v1094_tmp = v1093;
    v1094_tmp(19, 8) = v1092;
    v1094 = v1094_tmp;	// L2114
    pe_word15 = v1094;	// L2115
    uint32_t v1095 = pe_word15;	// L2116
    v1072.write(v1095);	// L2117
  }
}

void pe_0_0(
  ac_channel< uint32_t >& v1096,
  ac_channel< uint32_t >& v1097,
  ac_channel< uint32_t >& v1098,
  ac_channel< int32_t >& v1099,
  ac_channel< int8_t >& v1100
) {	// L2121
  uint32_t v1101 = v1096.read();	// L2132
  uint32_t trip_word16;	// L2133
  trip_word16 = v1101;	// L2134
  uint32_t v1102 = trip_word16;	// L2135
  uint16_t v1103;
  ap_int<32> v1103_tmp = v1102;
  v1103 = v1103_tmp(15, 0);	// L2136
  int32_t v1104 = v1103;	// L2137
  int32_t n_wavefront_row;	// L2138
  n_wavefront_row = v1104;	// L2139
  int8_t weight;	// L2140
  weight = 0;	// L2141
  int32_t mm_rows;	// L2142
  mm_rows = 0;	// L2143
  int32_t row3;	// L2144
  row3 = -1;	// L2145
  int32_t v1105 = n_wavefront_row;	// L2146
  int v1106 = v1105;	// L2147
  for (int v1107 = 0; v1107 < v1106; v1107 += 1) {	// L2148
    int32_t v1108 = row3;	// L2149
    ac_int<33, true> v1109 = v1108;	// L2150
    ac_int<33, true> v1110 = v1109 + 1;	// L2151
    int32_t v1111 = v1110;	// L2152
    row3 = v1111;	// L2153
    int32_t v1112 = row3;	// L2154
    int32_t v1113 = mm_rows;	// L2155
    bool v1114 = v1112 >= v1113;	// L2156
    if (v1114) {	// L2157
      uint32_t v1115 = v1096.read();	// L2158
      uint32_t pe_word16;	// L2159
      pe_word16 = v1115;	// L2160
      uint32_t v1116 = pe_word16;	// L2161
      uint8_t v1117;
      ap_int<32> v1117_tmp = v1116;
      v1117 = v1117_tmp(7, 0);	// L2162
      weight = v1117;	// L2163
      uint32_t v1118 = pe_word16;	// L2164
      ac_int<12, false> v1119;
      ap_int<32> v1119_tmp = v1118;
      v1119 = v1119_tmp(19, 8);	// L2165
      int32_t v1120 = v1119;	// L2166
      mm_rows = v1120;	// L2167
      row3 = 0;	// L2168
    }
    int8_t activation1;	// L2170
    activation1 = 0;	// L2171
    uint32_t v1121 = v1097.read();	// L2172
    uint32_t activation_word;	// L2173
    activation_word = v1121;	// L2174
    uint32_t v1122 = activation_word;	// L2175
    v1098.write(v1122);	// L2176
    uint32_t v1123 = activation_word;	// L2177
    uint8_t v1124;
    ap_int<32> v1124_tmp = v1123;
    v1124 = v1124_tmp(7, 0);	// L2178
    activation1 = v1124;	// L2179
    int32_t psum_north;	// L2180
    psum_north = 0;	// L2181
    int8_t v1125 = activation1;	// L2182
    int16_t v1126 = v1125;	// L2183
    int16_t activation16;	// L2184
    activation16 = v1126;	// L2185
    int8_t v1127 = weight;	// L2186
    int16_t v1128 = v1127;	// L2187
    int16_t weight16;	// L2188
    weight16 = v1128;	// L2189
    int32_t v1129 = psum_north;	// L2190
    int16_t v1130 = activation16;	// L2191
    int16_t v1131 = weight16;	// L2192
    int32_t v1132 = v1130;	// L2193
    int32_t v1133 = v1131;	// L2194
    int32_t v1134 = v1132 * v1133;	// L2195
    ac_int<33, true> v1135 = v1129;	// L2196
    ac_int<33, true> v1136 = v1134;	// L2197
    ac_int<33, true> v1137 = v1135 + v1136;	// L2198
    int32_t v1138 = v1137;	// L2199
    int32_t psum;	// L2200
    psum = v1138;	// L2201
    int32_t v1139 = psum;	// L2202
    v1099.write(v1139);	// L2203
    int8_t v1140 = activation1;	// L2204
    v1100.write(v1140);	// L2205
  }
}

void pe_0_1(
  ac_channel< uint32_t >& v1141,
  ac_channel< int8_t >& v1142,
  ac_channel< int32_t >& v1143,
  ac_channel< int8_t >& v1144
) {	// L2209
  uint32_t v1145 = v1141.read();	// L2220
  uint32_t trip_word17;	// L2221
  trip_word17 = v1145;	// L2222
  uint32_t v1146 = trip_word17;	// L2223
  uint16_t v1147;
  ap_int<32> v1147_tmp = v1146;
  v1147 = v1147_tmp(15, 0);	// L2224
  int32_t v1148 = v1147;	// L2225
  int32_t n_wavefront_row1;	// L2226
  n_wavefront_row1 = v1148;	// L2227
  int8_t weight1;	// L2228
  weight1 = 0;	// L2229
  int32_t mm_rows1;	// L2230
  mm_rows1 = 0;	// L2231
  int32_t row4;	// L2232
  row4 = -1;	// L2233
  int32_t v1149 = n_wavefront_row1;	// L2234
  int v1150 = v1149;	// L2235
  for (int v1151 = 0; v1151 < v1150; v1151 += 1) {	// L2236
    int32_t v1152 = row4;	// L2237
    ac_int<33, true> v1153 = v1152;	// L2238
    ac_int<33, true> v1154 = v1153 + 1;	// L2239
    int32_t v1155 = v1154;	// L2240
    row4 = v1155;	// L2241
    int32_t v1156 = row4;	// L2242
    int32_t v1157 = mm_rows1;	// L2243
    bool v1158 = v1156 >= v1157;	// L2244
    if (v1158) {	// L2245
      uint32_t v1159 = v1141.read();	// L2246
      uint32_t pe_word17;	// L2247
      pe_word17 = v1159;	// L2248
      uint32_t v1160 = pe_word17;	// L2249
      uint8_t v1161;
      ap_int<32> v1161_tmp = v1160;
      v1161 = v1161_tmp(7, 0);	// L2250
      weight1 = v1161;	// L2251
      uint32_t v1162 = pe_word17;	// L2252
      ac_int<12, false> v1163;
      ap_int<32> v1163_tmp = v1162;
      v1163 = v1163_tmp(19, 8);	// L2253
      int32_t v1164 = v1163;	// L2254
      mm_rows1 = v1164;	// L2255
      row4 = 0;	// L2256
    }
    int8_t activation2;	// L2258
    activation2 = 0;	// L2259
    int8_t v1165 = v1142.read();	// L2260
    activation2 = v1165;	// L2261
    int32_t psum_north1;	// L2262
    psum_north1 = 0;	// L2263
    int8_t v1166 = activation2;	// L2264
    int16_t v1167 = v1166;	// L2265
    int16_t activation161;	// L2266
    activation161 = v1167;	// L2267
    int8_t v1168 = weight1;	// L2268
    int16_t v1169 = v1168;	// L2269
    int16_t weight161;	// L2270
    weight161 = v1169;	// L2271
    int32_t v1170 = psum_north1;	// L2272
    int16_t v1171 = activation161;	// L2273
    int16_t v1172 = weight161;	// L2274
    int32_t v1173 = v1171;	// L2275
    int32_t v1174 = v1172;	// L2276
    int32_t v1175 = v1173 * v1174;	// L2277
    ac_int<33, true> v1176 = v1170;	// L2278
    ac_int<33, true> v1177 = v1175;	// L2279
    ac_int<33, true> v1178 = v1176 + v1177;	// L2280
    int32_t v1179 = v1178;	// L2281
    int32_t psum1;	// L2282
    psum1 = v1179;	// L2283
    int32_t v1180 = psum1;	// L2284
    v1143.write(v1180);	// L2285
    int8_t v1181 = activation2;	// L2286
    v1144.write(v1181);	// L2287
  }
}

void pe_0_2(
  ac_channel< uint32_t >& v1182,
  ac_channel< int8_t >& v1183,
  ac_channel< int32_t >& v1184,
  ac_channel< int8_t >& v1185
) {	// L2291
  uint32_t v1186 = v1182.read();	// L2302
  uint32_t trip_word18;	// L2303
  trip_word18 = v1186;	// L2304
  uint32_t v1187 = trip_word18;	// L2305
  uint16_t v1188;
  ap_int<32> v1188_tmp = v1187;
  v1188 = v1188_tmp(15, 0);	// L2306
  int32_t v1189 = v1188;	// L2307
  int32_t n_wavefront_row2;	// L2308
  n_wavefront_row2 = v1189;	// L2309
  int8_t weight2;	// L2310
  weight2 = 0;	// L2311
  int32_t mm_rows2;	// L2312
  mm_rows2 = 0;	// L2313
  int32_t row5;	// L2314
  row5 = -1;	// L2315
  int32_t v1190 = n_wavefront_row2;	// L2316
  int v1191 = v1190;	// L2317
  for (int v1192 = 0; v1192 < v1191; v1192 += 1) {	// L2318
    int32_t v1193 = row5;	// L2319
    ac_int<33, true> v1194 = v1193;	// L2320
    ac_int<33, true> v1195 = v1194 + 1;	// L2321
    int32_t v1196 = v1195;	// L2322
    row5 = v1196;	// L2323
    int32_t v1197 = row5;	// L2324
    int32_t v1198 = mm_rows2;	// L2325
    bool v1199 = v1197 >= v1198;	// L2326
    if (v1199) {	// L2327
      uint32_t v1200 = v1182.read();	// L2328
      uint32_t pe_word18;	// L2329
      pe_word18 = v1200;	// L2330
      uint32_t v1201 = pe_word18;	// L2331
      uint8_t v1202;
      ap_int<32> v1202_tmp = v1201;
      v1202 = v1202_tmp(7, 0);	// L2332
      weight2 = v1202;	// L2333
      uint32_t v1203 = pe_word18;	// L2334
      ac_int<12, false> v1204;
      ap_int<32> v1204_tmp = v1203;
      v1204 = v1204_tmp(19, 8);	// L2335
      int32_t v1205 = v1204;	// L2336
      mm_rows2 = v1205;	// L2337
      row5 = 0;	// L2338
    }
    int8_t activation3;	// L2340
    activation3 = 0;	// L2341
    int8_t v1206 = v1183.read();	// L2342
    activation3 = v1206;	// L2343
    int32_t psum_north2;	// L2344
    psum_north2 = 0;	// L2345
    int8_t v1207 = activation3;	// L2346
    int16_t v1208 = v1207;	// L2347
    int16_t activation162;	// L2348
    activation162 = v1208;	// L2349
    int8_t v1209 = weight2;	// L2350
    int16_t v1210 = v1209;	// L2351
    int16_t weight162;	// L2352
    weight162 = v1210;	// L2353
    int32_t v1211 = psum_north2;	// L2354
    int16_t v1212 = activation162;	// L2355
    int16_t v1213 = weight162;	// L2356
    int32_t v1214 = v1212;	// L2357
    int32_t v1215 = v1213;	// L2358
    int32_t v1216 = v1214 * v1215;	// L2359
    ac_int<33, true> v1217 = v1211;	// L2360
    ac_int<33, true> v1218 = v1216;	// L2361
    ac_int<33, true> v1219 = v1217 + v1218;	// L2362
    int32_t v1220 = v1219;	// L2363
    int32_t psum2;	// L2364
    psum2 = v1220;	// L2365
    int32_t v1221 = psum2;	// L2366
    v1184.write(v1221);	// L2367
    int8_t v1222 = activation3;	// L2368
    v1185.write(v1222);	// L2369
  }
}

void pe_0_3(
  ac_channel< uint32_t >& v1223,
  ac_channel< int8_t >& v1224,
  ac_channel< int32_t >& v1225
) {	// L2373
  uint32_t v1226 = v1223.read();	// L2384
  uint32_t trip_word19;	// L2385
  trip_word19 = v1226;	// L2386
  uint32_t v1227 = trip_word19;	// L2387
  uint16_t v1228;
  ap_int<32> v1228_tmp = v1227;
  v1228 = v1228_tmp(15, 0);	// L2388
  int32_t v1229 = v1228;	// L2389
  int32_t n_wavefront_row3;	// L2390
  n_wavefront_row3 = v1229;	// L2391
  int8_t weight3;	// L2392
  weight3 = 0;	// L2393
  int32_t mm_rows3;	// L2394
  mm_rows3 = 0;	// L2395
  int32_t row6;	// L2396
  row6 = -1;	// L2397
  int32_t v1230 = n_wavefront_row3;	// L2398
  int v1231 = v1230;	// L2399
  for (int v1232 = 0; v1232 < v1231; v1232 += 1) {	// L2400
    int32_t v1233 = row6;	// L2401
    ac_int<33, true> v1234 = v1233;	// L2402
    ac_int<33, true> v1235 = v1234 + 1;	// L2403
    int32_t v1236 = v1235;	// L2404
    row6 = v1236;	// L2405
    int32_t v1237 = row6;	// L2406
    int32_t v1238 = mm_rows3;	// L2407
    bool v1239 = v1237 >= v1238;	// L2408
    if (v1239) {	// L2409
      uint32_t v1240 = v1223.read();	// L2410
      uint32_t pe_word19;	// L2411
      pe_word19 = v1240;	// L2412
      uint32_t v1241 = pe_word19;	// L2413
      uint8_t v1242;
      ap_int<32> v1242_tmp = v1241;
      v1242 = v1242_tmp(7, 0);	// L2414
      weight3 = v1242;	// L2415
      uint32_t v1243 = pe_word19;	// L2416
      ac_int<12, false> v1244;
      ap_int<32> v1244_tmp = v1243;
      v1244 = v1244_tmp(19, 8);	// L2417
      int32_t v1245 = v1244;	// L2418
      mm_rows3 = v1245;	// L2419
      row6 = 0;	// L2420
    }
    int8_t activation4;	// L2422
    activation4 = 0;	// L2423
    int8_t v1246 = v1224.read();	// L2424
    activation4 = v1246;	// L2425
    int32_t psum_north3;	// L2426
    psum_north3 = 0;	// L2427
    int8_t v1247 = activation4;	// L2428
    int16_t v1248 = v1247;	// L2429
    int16_t activation163;	// L2430
    activation163 = v1248;	// L2431
    int8_t v1249 = weight3;	// L2432
    int16_t v1250 = v1249;	// L2433
    int16_t weight163;	// L2434
    weight163 = v1250;	// L2435
    int32_t v1251 = psum_north3;	// L2436
    int16_t v1252 = activation163;	// L2437
    int16_t v1253 = weight163;	// L2438
    int32_t v1254 = v1252;	// L2439
    int32_t v1255 = v1253;	// L2440
    int32_t v1256 = v1254 * v1255;	// L2441
    ac_int<33, true> v1257 = v1251;	// L2442
    ac_int<33, true> v1258 = v1256;	// L2443
    ac_int<33, true> v1259 = v1257 + v1258;	// L2444
    int32_t v1260 = v1259;	// L2445
    int32_t psum3;	// L2446
    psum3 = v1260;	// L2447
    int32_t v1261 = psum3;	// L2448
    v1225.write(v1261);	// L2449
  }
}

void pe_1_0(
  ac_channel< uint32_t >& v1262,
  ac_channel< uint32_t >& v1263,
  ac_channel< uint32_t >& v1264,
  ac_channel< int32_t >& v1265,
  ac_channel< int32_t >& v1266,
  ac_channel< int8_t >& v1267
) {	// L2453
  uint32_t v1268 = v1262.read();	// L2464
  uint32_t trip_word20;	// L2465
  trip_word20 = v1268;	// L2466
  uint32_t v1269 = trip_word20;	// L2467
  uint16_t v1270;
  ap_int<32> v1270_tmp = v1269;
  v1270 = v1270_tmp(15, 0);	// L2468
  int32_t v1271 = v1270;	// L2469
  int32_t n_wavefront_row4;	// L2470
  n_wavefront_row4 = v1271;	// L2471
  int8_t weight4;	// L2472
  weight4 = 0;	// L2473
  int32_t mm_rows4;	// L2474
  mm_rows4 = 0;	// L2475
  int32_t row7;	// L2476
  row7 = -1;	// L2477
  int32_t v1272 = n_wavefront_row4;	// L2478
  int v1273 = v1272;	// L2479
  for (int v1274 = 0; v1274 < v1273; v1274 += 1) {	// L2480
    int32_t v1275 = row7;	// L2481
    ac_int<33, true> v1276 = v1275;	// L2482
    ac_int<33, true> v1277 = v1276 + 1;	// L2483
    int32_t v1278 = v1277;	// L2484
    row7 = v1278;	// L2485
    int32_t v1279 = row7;	// L2486
    int32_t v1280 = mm_rows4;	// L2487
    bool v1281 = v1279 >= v1280;	// L2488
    if (v1281) {	// L2489
      uint32_t v1282 = v1262.read();	// L2490
      uint32_t pe_word20;	// L2491
      pe_word20 = v1282;	// L2492
      uint32_t v1283 = pe_word20;	// L2493
      uint8_t v1284;
      ap_int<32> v1284_tmp = v1283;
      v1284 = v1284_tmp(7, 0);	// L2494
      weight4 = v1284;	// L2495
      uint32_t v1285 = pe_word20;	// L2496
      ac_int<12, false> v1286;
      ap_int<32> v1286_tmp = v1285;
      v1286 = v1286_tmp(19, 8);	// L2497
      int32_t v1287 = v1286;	// L2498
      mm_rows4 = v1287;	// L2499
      row7 = 0;	// L2500
    }
    int8_t activation5;	// L2502
    activation5 = 0;	// L2503
    uint32_t v1288 = v1263.read();	// L2504
    uint32_t activation_word1;	// L2505
    activation_word1 = v1288;	// L2506
    uint32_t v1289 = activation_word1;	// L2507
    v1264.write(v1289);	// L2508
    uint32_t v1290 = activation_word1;	// L2509
    uint8_t v1291;
    ap_int<32> v1291_tmp = v1290;
    v1291 = v1291_tmp(15, 8);	// L2510
    activation5 = v1291;	// L2511
    int32_t psum_north4;	// L2512
    psum_north4 = 0;	// L2513
    int32_t v1292 = v1265.read();	// L2514
    psum_north4 = v1292;	// L2515
    int8_t v1293 = activation5;	// L2516
    int16_t v1294 = v1293;	// L2517
    int16_t activation164;	// L2518
    activation164 = v1294;	// L2519
    int8_t v1295 = weight4;	// L2520
    int16_t v1296 = v1295;	// L2521
    int16_t weight164;	// L2522
    weight164 = v1296;	// L2523
    int32_t v1297 = psum_north4;	// L2524
    int16_t v1298 = activation164;	// L2525
    int16_t v1299 = weight164;	// L2526
    int32_t v1300 = v1298;	// L2527
    int32_t v1301 = v1299;	// L2528
    int32_t v1302 = v1300 * v1301;	// L2529
    ac_int<33, true> v1303 = v1297;	// L2530
    ac_int<33, true> v1304 = v1302;	// L2531
    ac_int<33, true> v1305 = v1303 + v1304;	// L2532
    int32_t v1306 = v1305;	// L2533
    int32_t psum4;	// L2534
    psum4 = v1306;	// L2535
    int32_t v1307 = psum4;	// L2536
    v1266.write(v1307);	// L2537
    int8_t v1308 = activation5;	// L2538
    v1267.write(v1308);	// L2539
  }
}

void pe_1_1(
  ac_channel< uint32_t >& v1309,
  ac_channel< int8_t >& v1310,
  ac_channel< int32_t >& v1311,
  ac_channel< int32_t >& v1312,
  ac_channel< int8_t >& v1313
) {	// L2543
  uint32_t v1314 = v1309.read();	// L2554
  uint32_t trip_word21;	// L2555
  trip_word21 = v1314;	// L2556
  uint32_t v1315 = trip_word21;	// L2557
  uint16_t v1316;
  ap_int<32> v1316_tmp = v1315;
  v1316 = v1316_tmp(15, 0);	// L2558
  int32_t v1317 = v1316;	// L2559
  int32_t n_wavefront_row5;	// L2560
  n_wavefront_row5 = v1317;	// L2561
  int8_t weight5;	// L2562
  weight5 = 0;	// L2563
  int32_t mm_rows5;	// L2564
  mm_rows5 = 0;	// L2565
  int32_t row8;	// L2566
  row8 = -1;	// L2567
  int32_t v1318 = n_wavefront_row5;	// L2568
  int v1319 = v1318;	// L2569
  for (int v1320 = 0; v1320 < v1319; v1320 += 1) {	// L2570
    int32_t v1321 = row8;	// L2571
    ac_int<33, true> v1322 = v1321;	// L2572
    ac_int<33, true> v1323 = v1322 + 1;	// L2573
    int32_t v1324 = v1323;	// L2574
    row8 = v1324;	// L2575
    int32_t v1325 = row8;	// L2576
    int32_t v1326 = mm_rows5;	// L2577
    bool v1327 = v1325 >= v1326;	// L2578
    if (v1327) {	// L2579
      uint32_t v1328 = v1309.read();	// L2580
      uint32_t pe_word21;	// L2581
      pe_word21 = v1328;	// L2582
      uint32_t v1329 = pe_word21;	// L2583
      uint8_t v1330;
      ap_int<32> v1330_tmp = v1329;
      v1330 = v1330_tmp(7, 0);	// L2584
      weight5 = v1330;	// L2585
      uint32_t v1331 = pe_word21;	// L2586
      ac_int<12, false> v1332;
      ap_int<32> v1332_tmp = v1331;
      v1332 = v1332_tmp(19, 8);	// L2587
      int32_t v1333 = v1332;	// L2588
      mm_rows5 = v1333;	// L2589
      row8 = 0;	// L2590
    }
    int8_t activation6;	// L2592
    activation6 = 0;	// L2593
    int8_t v1334 = v1310.read();	// L2594
    activation6 = v1334;	// L2595
    int32_t psum_north5;	// L2596
    psum_north5 = 0;	// L2597
    int32_t v1335 = v1311.read();	// L2598
    psum_north5 = v1335;	// L2599
    int8_t v1336 = activation6;	// L2600
    int16_t v1337 = v1336;	// L2601
    int16_t activation165;	// L2602
    activation165 = v1337;	// L2603
    int8_t v1338 = weight5;	// L2604
    int16_t v1339 = v1338;	// L2605
    int16_t weight165;	// L2606
    weight165 = v1339;	// L2607
    int32_t v1340 = psum_north5;	// L2608
    int16_t v1341 = activation165;	// L2609
    int16_t v1342 = weight165;	// L2610
    int32_t v1343 = v1341;	// L2611
    int32_t v1344 = v1342;	// L2612
    int32_t v1345 = v1343 * v1344;	// L2613
    ac_int<33, true> v1346 = v1340;	// L2614
    ac_int<33, true> v1347 = v1345;	// L2615
    ac_int<33, true> v1348 = v1346 + v1347;	// L2616
    int32_t v1349 = v1348;	// L2617
    int32_t psum5;	// L2618
    psum5 = v1349;	// L2619
    int32_t v1350 = psum5;	// L2620
    v1312.write(v1350);	// L2621
    int8_t v1351 = activation6;	// L2622
    v1313.write(v1351);	// L2623
  }
}

void pe_1_2(
  ac_channel< uint32_t >& v1352,
  ac_channel< int8_t >& v1353,
  ac_channel< int32_t >& v1354,
  ac_channel< int32_t >& v1355,
  ac_channel< int8_t >& v1356
) {	// L2627
  uint32_t v1357 = v1352.read();	// L2638
  uint32_t trip_word22;	// L2639
  trip_word22 = v1357;	// L2640
  uint32_t v1358 = trip_word22;	// L2641
  uint16_t v1359;
  ap_int<32> v1359_tmp = v1358;
  v1359 = v1359_tmp(15, 0);	// L2642
  int32_t v1360 = v1359;	// L2643
  int32_t n_wavefront_row6;	// L2644
  n_wavefront_row6 = v1360;	// L2645
  int8_t weight6;	// L2646
  weight6 = 0;	// L2647
  int32_t mm_rows6;	// L2648
  mm_rows6 = 0;	// L2649
  int32_t row9;	// L2650
  row9 = -1;	// L2651
  int32_t v1361 = n_wavefront_row6;	// L2652
  int v1362 = v1361;	// L2653
  for (int v1363 = 0; v1363 < v1362; v1363 += 1) {	// L2654
    int32_t v1364 = row9;	// L2655
    ac_int<33, true> v1365 = v1364;	// L2656
    ac_int<33, true> v1366 = v1365 + 1;	// L2657
    int32_t v1367 = v1366;	// L2658
    row9 = v1367;	// L2659
    int32_t v1368 = row9;	// L2660
    int32_t v1369 = mm_rows6;	// L2661
    bool v1370 = v1368 >= v1369;	// L2662
    if (v1370) {	// L2663
      uint32_t v1371 = v1352.read();	// L2664
      uint32_t pe_word22;	// L2665
      pe_word22 = v1371;	// L2666
      uint32_t v1372 = pe_word22;	// L2667
      uint8_t v1373;
      ap_int<32> v1373_tmp = v1372;
      v1373 = v1373_tmp(7, 0);	// L2668
      weight6 = v1373;	// L2669
      uint32_t v1374 = pe_word22;	// L2670
      ac_int<12, false> v1375;
      ap_int<32> v1375_tmp = v1374;
      v1375 = v1375_tmp(19, 8);	// L2671
      int32_t v1376 = v1375;	// L2672
      mm_rows6 = v1376;	// L2673
      row9 = 0;	// L2674
    }
    int8_t activation7;	// L2676
    activation7 = 0;	// L2677
    int8_t v1377 = v1353.read();	// L2678
    activation7 = v1377;	// L2679
    int32_t psum_north6;	// L2680
    psum_north6 = 0;	// L2681
    int32_t v1378 = v1354.read();	// L2682
    psum_north6 = v1378;	// L2683
    int8_t v1379 = activation7;	// L2684
    int16_t v1380 = v1379;	// L2685
    int16_t activation166;	// L2686
    activation166 = v1380;	// L2687
    int8_t v1381 = weight6;	// L2688
    int16_t v1382 = v1381;	// L2689
    int16_t weight166;	// L2690
    weight166 = v1382;	// L2691
    int32_t v1383 = psum_north6;	// L2692
    int16_t v1384 = activation166;	// L2693
    int16_t v1385 = weight166;	// L2694
    int32_t v1386 = v1384;	// L2695
    int32_t v1387 = v1385;	// L2696
    int32_t v1388 = v1386 * v1387;	// L2697
    ac_int<33, true> v1389 = v1383;	// L2698
    ac_int<33, true> v1390 = v1388;	// L2699
    ac_int<33, true> v1391 = v1389 + v1390;	// L2700
    int32_t v1392 = v1391;	// L2701
    int32_t psum6;	// L2702
    psum6 = v1392;	// L2703
    int32_t v1393 = psum6;	// L2704
    v1355.write(v1393);	// L2705
    int8_t v1394 = activation7;	// L2706
    v1356.write(v1394);	// L2707
  }
}

void pe_1_3(
  ac_channel< uint32_t >& v1395,
  ac_channel< int8_t >& v1396,
  ac_channel< int32_t >& v1397,
  ac_channel< int32_t >& v1398
) {	// L2711
  uint32_t v1399 = v1395.read();	// L2722
  uint32_t trip_word23;	// L2723
  trip_word23 = v1399;	// L2724
  uint32_t v1400 = trip_word23;	// L2725
  uint16_t v1401;
  ap_int<32> v1401_tmp = v1400;
  v1401 = v1401_tmp(15, 0);	// L2726
  int32_t v1402 = v1401;	// L2727
  int32_t n_wavefront_row7;	// L2728
  n_wavefront_row7 = v1402;	// L2729
  int8_t weight7;	// L2730
  weight7 = 0;	// L2731
  int32_t mm_rows7;	// L2732
  mm_rows7 = 0;	// L2733
  int32_t row10;	// L2734
  row10 = -1;	// L2735
  int32_t v1403 = n_wavefront_row7;	// L2736
  int v1404 = v1403;	// L2737
  for (int v1405 = 0; v1405 < v1404; v1405 += 1) {	// L2738
    int32_t v1406 = row10;	// L2739
    ac_int<33, true> v1407 = v1406;	// L2740
    ac_int<33, true> v1408 = v1407 + 1;	// L2741
    int32_t v1409 = v1408;	// L2742
    row10 = v1409;	// L2743
    int32_t v1410 = row10;	// L2744
    int32_t v1411 = mm_rows7;	// L2745
    bool v1412 = v1410 >= v1411;	// L2746
    if (v1412) {	// L2747
      uint32_t v1413 = v1395.read();	// L2748
      uint32_t pe_word23;	// L2749
      pe_word23 = v1413;	// L2750
      uint32_t v1414 = pe_word23;	// L2751
      uint8_t v1415;
      ap_int<32> v1415_tmp = v1414;
      v1415 = v1415_tmp(7, 0);	// L2752
      weight7 = v1415;	// L2753
      uint32_t v1416 = pe_word23;	// L2754
      ac_int<12, false> v1417;
      ap_int<32> v1417_tmp = v1416;
      v1417 = v1417_tmp(19, 8);	// L2755
      int32_t v1418 = v1417;	// L2756
      mm_rows7 = v1418;	// L2757
      row10 = 0;	// L2758
    }
    int8_t activation8;	// L2760
    activation8 = 0;	// L2761
    int8_t v1419 = v1396.read();	// L2762
    activation8 = v1419;	// L2763
    int32_t psum_north7;	// L2764
    psum_north7 = 0;	// L2765
    int32_t v1420 = v1397.read();	// L2766
    psum_north7 = v1420;	// L2767
    int8_t v1421 = activation8;	// L2768
    int16_t v1422 = v1421;	// L2769
    int16_t activation167;	// L2770
    activation167 = v1422;	// L2771
    int8_t v1423 = weight7;	// L2772
    int16_t v1424 = v1423;	// L2773
    int16_t weight167;	// L2774
    weight167 = v1424;	// L2775
    int32_t v1425 = psum_north7;	// L2776
    int16_t v1426 = activation167;	// L2777
    int16_t v1427 = weight167;	// L2778
    int32_t v1428 = v1426;	// L2779
    int32_t v1429 = v1427;	// L2780
    int32_t v1430 = v1428 * v1429;	// L2781
    ac_int<33, true> v1431 = v1425;	// L2782
    ac_int<33, true> v1432 = v1430;	// L2783
    ac_int<33, true> v1433 = v1431 + v1432;	// L2784
    int32_t v1434 = v1433;	// L2785
    int32_t psum7;	// L2786
    psum7 = v1434;	// L2787
    int32_t v1435 = psum7;	// L2788
    v1398.write(v1435);	// L2789
  }
}

void pe_2_0(
  ac_channel< uint32_t >& v1436,
  ac_channel< uint32_t >& v1437,
  ac_channel< uint32_t >& v1438,
  ac_channel< int32_t >& v1439,
  ac_channel< int32_t >& v1440,
  ac_channel< int8_t >& v1441
) {	// L2793
  uint32_t v1442 = v1436.read();	// L2806
  uint32_t trip_word24;	// L2807
  trip_word24 = v1442;	// L2808
  uint32_t v1443 = trip_word24;	// L2809
  uint16_t v1444;
  ap_int<32> v1444_tmp = v1443;
  v1444 = v1444_tmp(15, 0);	// L2810
  int32_t v1445 = v1444;	// L2811
  int32_t n_wavefront_row8;	// L2812
  n_wavefront_row8 = v1445;	// L2813
  int8_t weight8;	// L2814
  weight8 = 0;	// L2815
  int32_t mm_rows8;	// L2816
  mm_rows8 = 0;	// L2817
  int32_t row11;	// L2818
  row11 = -1;	// L2819
  int32_t v1446 = n_wavefront_row8;	// L2820
  int v1447 = v1446;	// L2821
  for (int v1448 = 0; v1448 < v1447; v1448 += 1) {	// L2822
    int32_t v1449 = row11;	// L2823
    ac_int<33, true> v1450 = v1449;	// L2824
    ac_int<33, true> v1451 = v1450 + 1;	// L2825
    int32_t v1452 = v1451;	// L2826
    row11 = v1452;	// L2827
    int32_t v1453 = row11;	// L2828
    int32_t v1454 = mm_rows8;	// L2829
    bool v1455 = v1453 >= v1454;	// L2830
    if (v1455) {	// L2831
      uint32_t v1456 = v1436.read();	// L2832
      uint32_t pe_word24;	// L2833
      pe_word24 = v1456;	// L2834
      uint32_t v1457 = pe_word24;	// L2835
      uint8_t v1458;
      ap_int<32> v1458_tmp = v1457;
      v1458 = v1458_tmp(7, 0);	// L2836
      weight8 = v1458;	// L2837
      uint32_t v1459 = pe_word24;	// L2838
      ac_int<12, false> v1460;
      ap_int<32> v1460_tmp = v1459;
      v1460 = v1460_tmp(19, 8);	// L2839
      int32_t v1461 = v1460;	// L2840
      mm_rows8 = v1461;	// L2841
      row11 = 0;	// L2842
    }
    int8_t activation9;	// L2844
    activation9 = 0;	// L2845
    uint32_t v1462 = v1437.read();	// L2846
    uint32_t activation_word2;	// L2847
    activation_word2 = v1462;	// L2848
    uint32_t v1463 = activation_word2;	// L2849
    v1438.write(v1463);	// L2850
    uint32_t v1464 = activation_word2;	// L2851
    uint8_t v1465;
    ap_int<32> v1465_tmp = v1464;
    v1465 = v1465_tmp(23, 16);	// L2852
    activation9 = v1465;	// L2853
    int32_t psum_north8;	// L2854
    psum_north8 = 0;	// L2855
    int32_t v1466 = v1439.read();	// L2856
    psum_north8 = v1466;	// L2857
    int8_t v1467 = activation9;	// L2858
    int16_t v1468 = v1467;	// L2859
    int16_t activation168;	// L2860
    activation168 = v1468;	// L2861
    int8_t v1469 = weight8;	// L2862
    int16_t v1470 = v1469;	// L2863
    int16_t weight168;	// L2864
    weight168 = v1470;	// L2865
    int32_t v1471 = psum_north8;	// L2866
    int16_t v1472 = activation168;	// L2867
    int16_t v1473 = weight168;	// L2868
    int32_t v1474 = v1472;	// L2869
    int32_t v1475 = v1473;	// L2870
    int32_t v1476 = v1474 * v1475;	// L2871
    ac_int<33, true> v1477 = v1471;	// L2872
    ac_int<33, true> v1478 = v1476;	// L2873
    ac_int<33, true> v1479 = v1477 + v1478;	// L2874
    int32_t v1480 = v1479;	// L2875
    int32_t psum8;	// L2876
    psum8 = v1480;	// L2877
    int32_t v1481 = psum8;	// L2878
    v1440.write(v1481);	// L2879
    int8_t v1482 = activation9;	// L2880
    v1441.write(v1482);	// L2881
  }
}

void pe_2_1(
  ac_channel< uint32_t >& v1483,
  ac_channel< int8_t >& v1484,
  ac_channel< int32_t >& v1485,
  ac_channel< int32_t >& v1486,
  ac_channel< int8_t >& v1487
) {	// L2885
  uint32_t v1488 = v1483.read();	// L2896
  uint32_t trip_word25;	// L2897
  trip_word25 = v1488;	// L2898
  uint32_t v1489 = trip_word25;	// L2899
  uint16_t v1490;
  ap_int<32> v1490_tmp = v1489;
  v1490 = v1490_tmp(15, 0);	// L2900
  int32_t v1491 = v1490;	// L2901
  int32_t n_wavefront_row9;	// L2902
  n_wavefront_row9 = v1491;	// L2903
  int8_t weight9;	// L2904
  weight9 = 0;	// L2905
  int32_t mm_rows9;	// L2906
  mm_rows9 = 0;	// L2907
  int32_t row12;	// L2908
  row12 = -1;	// L2909
  int32_t v1492 = n_wavefront_row9;	// L2910
  int v1493 = v1492;	// L2911
  for (int v1494 = 0; v1494 < v1493; v1494 += 1) {	// L2912
    int32_t v1495 = row12;	// L2913
    ac_int<33, true> v1496 = v1495;	// L2914
    ac_int<33, true> v1497 = v1496 + 1;	// L2915
    int32_t v1498 = v1497;	// L2916
    row12 = v1498;	// L2917
    int32_t v1499 = row12;	// L2918
    int32_t v1500 = mm_rows9;	// L2919
    bool v1501 = v1499 >= v1500;	// L2920
    if (v1501) {	// L2921
      uint32_t v1502 = v1483.read();	// L2922
      uint32_t pe_word25;	// L2923
      pe_word25 = v1502;	// L2924
      uint32_t v1503 = pe_word25;	// L2925
      uint8_t v1504;
      ap_int<32> v1504_tmp = v1503;
      v1504 = v1504_tmp(7, 0);	// L2926
      weight9 = v1504;	// L2927
      uint32_t v1505 = pe_word25;	// L2928
      ac_int<12, false> v1506;
      ap_int<32> v1506_tmp = v1505;
      v1506 = v1506_tmp(19, 8);	// L2929
      int32_t v1507 = v1506;	// L2930
      mm_rows9 = v1507;	// L2931
      row12 = 0;	// L2932
    }
    int8_t activation10;	// L2934
    activation10 = 0;	// L2935
    int8_t v1508 = v1484.read();	// L2936
    activation10 = v1508;	// L2937
    int32_t psum_north9;	// L2938
    psum_north9 = 0;	// L2939
    int32_t v1509 = v1485.read();	// L2940
    psum_north9 = v1509;	// L2941
    int8_t v1510 = activation10;	// L2942
    int16_t v1511 = v1510;	// L2943
    int16_t activation169;	// L2944
    activation169 = v1511;	// L2945
    int8_t v1512 = weight9;	// L2946
    int16_t v1513 = v1512;	// L2947
    int16_t weight169;	// L2948
    weight169 = v1513;	// L2949
    int32_t v1514 = psum_north9;	// L2950
    int16_t v1515 = activation169;	// L2951
    int16_t v1516 = weight169;	// L2952
    int32_t v1517 = v1515;	// L2953
    int32_t v1518 = v1516;	// L2954
    int32_t v1519 = v1517 * v1518;	// L2955
    ac_int<33, true> v1520 = v1514;	// L2956
    ac_int<33, true> v1521 = v1519;	// L2957
    ac_int<33, true> v1522 = v1520 + v1521;	// L2958
    int32_t v1523 = v1522;	// L2959
    int32_t psum9;	// L2960
    psum9 = v1523;	// L2961
    int32_t v1524 = psum9;	// L2962
    v1486.write(v1524);	// L2963
    int8_t v1525 = activation10;	// L2964
    v1487.write(v1525);	// L2965
  }
}

void pe_2_2(
  ac_channel< uint32_t >& v1526,
  ac_channel< int8_t >& v1527,
  ac_channel< int32_t >& v1528,
  ac_channel< int32_t >& v1529,
  ac_channel< int8_t >& v1530
) {	// L2969
  uint32_t v1531 = v1526.read();	// L2980
  uint32_t trip_word26;	// L2981
  trip_word26 = v1531;	// L2982
  uint32_t v1532 = trip_word26;	// L2983
  uint16_t v1533;
  ap_int<32> v1533_tmp = v1532;
  v1533 = v1533_tmp(15, 0);	// L2984
  int32_t v1534 = v1533;	// L2985
  int32_t n_wavefront_row10;	// L2986
  n_wavefront_row10 = v1534;	// L2987
  int8_t weight10;	// L2988
  weight10 = 0;	// L2989
  int32_t mm_rows10;	// L2990
  mm_rows10 = 0;	// L2991
  int32_t row13;	// L2992
  row13 = -1;	// L2993
  int32_t v1535 = n_wavefront_row10;	// L2994
  int v1536 = v1535;	// L2995
  for (int v1537 = 0; v1537 < v1536; v1537 += 1) {	// L2996
    int32_t v1538 = row13;	// L2997
    ac_int<33, true> v1539 = v1538;	// L2998
    ac_int<33, true> v1540 = v1539 + 1;	// L2999
    int32_t v1541 = v1540;	// L3000
    row13 = v1541;	// L3001
    int32_t v1542 = row13;	// L3002
    int32_t v1543 = mm_rows10;	// L3003
    bool v1544 = v1542 >= v1543;	// L3004
    if (v1544) {	// L3005
      uint32_t v1545 = v1526.read();	// L3006
      uint32_t pe_word26;	// L3007
      pe_word26 = v1545;	// L3008
      uint32_t v1546 = pe_word26;	// L3009
      uint8_t v1547;
      ap_int<32> v1547_tmp = v1546;
      v1547 = v1547_tmp(7, 0);	// L3010
      weight10 = v1547;	// L3011
      uint32_t v1548 = pe_word26;	// L3012
      ac_int<12, false> v1549;
      ap_int<32> v1549_tmp = v1548;
      v1549 = v1549_tmp(19, 8);	// L3013
      int32_t v1550 = v1549;	// L3014
      mm_rows10 = v1550;	// L3015
      row13 = 0;	// L3016
    }
    int8_t activation11;	// L3018
    activation11 = 0;	// L3019
    int8_t v1551 = v1527.read();	// L3020
    activation11 = v1551;	// L3021
    int32_t psum_north10;	// L3022
    psum_north10 = 0;	// L3023
    int32_t v1552 = v1528.read();	// L3024
    psum_north10 = v1552;	// L3025
    int8_t v1553 = activation11;	// L3026
    int16_t v1554 = v1553;	// L3027
    int16_t activation1610;	// L3028
    activation1610 = v1554;	// L3029
    int8_t v1555 = weight10;	// L3030
    int16_t v1556 = v1555;	// L3031
    int16_t weight1610;	// L3032
    weight1610 = v1556;	// L3033
    int32_t v1557 = psum_north10;	// L3034
    int16_t v1558 = activation1610;	// L3035
    int16_t v1559 = weight1610;	// L3036
    int32_t v1560 = v1558;	// L3037
    int32_t v1561 = v1559;	// L3038
    int32_t v1562 = v1560 * v1561;	// L3039
    ac_int<33, true> v1563 = v1557;	// L3040
    ac_int<33, true> v1564 = v1562;	// L3041
    ac_int<33, true> v1565 = v1563 + v1564;	// L3042
    int32_t v1566 = v1565;	// L3043
    int32_t psum10;	// L3044
    psum10 = v1566;	// L3045
    int32_t v1567 = psum10;	// L3046
    v1529.write(v1567);	// L3047
    int8_t v1568 = activation11;	// L3048
    v1530.write(v1568);	// L3049
  }
}

void pe_2_3(
  ac_channel< uint32_t >& v1569,
  ac_channel< int8_t >& v1570,
  ac_channel< int32_t >& v1571,
  ac_channel< int32_t >& v1572
) {	// L3053
  uint32_t v1573 = v1569.read();	// L3064
  uint32_t trip_word27;	// L3065
  trip_word27 = v1573;	// L3066
  uint32_t v1574 = trip_word27;	// L3067
  uint16_t v1575;
  ap_int<32> v1575_tmp = v1574;
  v1575 = v1575_tmp(15, 0);	// L3068
  int32_t v1576 = v1575;	// L3069
  int32_t n_wavefront_row11;	// L3070
  n_wavefront_row11 = v1576;	// L3071
  int8_t weight11;	// L3072
  weight11 = 0;	// L3073
  int32_t mm_rows11;	// L3074
  mm_rows11 = 0;	// L3075
  int32_t row14;	// L3076
  row14 = -1;	// L3077
  int32_t v1577 = n_wavefront_row11;	// L3078
  int v1578 = v1577;	// L3079
  for (int v1579 = 0; v1579 < v1578; v1579 += 1) {	// L3080
    int32_t v1580 = row14;	// L3081
    ac_int<33, true> v1581 = v1580;	// L3082
    ac_int<33, true> v1582 = v1581 + 1;	// L3083
    int32_t v1583 = v1582;	// L3084
    row14 = v1583;	// L3085
    int32_t v1584 = row14;	// L3086
    int32_t v1585 = mm_rows11;	// L3087
    bool v1586 = v1584 >= v1585;	// L3088
    if (v1586) {	// L3089
      uint32_t v1587 = v1569.read();	// L3090
      uint32_t pe_word27;	// L3091
      pe_word27 = v1587;	// L3092
      uint32_t v1588 = pe_word27;	// L3093
      uint8_t v1589;
      ap_int<32> v1589_tmp = v1588;
      v1589 = v1589_tmp(7, 0);	// L3094
      weight11 = v1589;	// L3095
      uint32_t v1590 = pe_word27;	// L3096
      ac_int<12, false> v1591;
      ap_int<32> v1591_tmp = v1590;
      v1591 = v1591_tmp(19, 8);	// L3097
      int32_t v1592 = v1591;	// L3098
      mm_rows11 = v1592;	// L3099
      row14 = 0;	// L3100
    }
    int8_t activation12;	// L3102
    activation12 = 0;	// L3103
    int8_t v1593 = v1570.read();	// L3104
    activation12 = v1593;	// L3105
    int32_t psum_north11;	// L3106
    psum_north11 = 0;	// L3107
    int32_t v1594 = v1571.read();	// L3108
    psum_north11 = v1594;	// L3109
    int8_t v1595 = activation12;	// L3110
    int16_t v1596 = v1595;	// L3111
    int16_t activation1611;	// L3112
    activation1611 = v1596;	// L3113
    int8_t v1597 = weight11;	// L3114
    int16_t v1598 = v1597;	// L3115
    int16_t weight1611;	// L3116
    weight1611 = v1598;	// L3117
    int32_t v1599 = psum_north11;	// L3118
    int16_t v1600 = activation1611;	// L3119
    int16_t v1601 = weight1611;	// L3120
    int32_t v1602 = v1600;	// L3121
    int32_t v1603 = v1601;	// L3122
    int32_t v1604 = v1602 * v1603;	// L3123
    ac_int<33, true> v1605 = v1599;	// L3124
    ac_int<33, true> v1606 = v1604;	// L3125
    ac_int<33, true> v1607 = v1605 + v1606;	// L3126
    int32_t v1608 = v1607;	// L3127
    int32_t psum11;	// L3128
    psum11 = v1608;	// L3129
    int32_t v1609 = psum11;	// L3130
    v1572.write(v1609);	// L3131
  }
}

void pe_3_0(
  ac_channel< uint32_t >& v1610,
  ac_channel< uint32_t >& v1611,
  ac_channel< int32_t >& v1612,
  ac_channel< ac_int<128, false> >& v1613,
  ac_channel< int8_t >& v1614
) {	// L3135
  uint32_t v1615 = v1610.read();	// L3149
  uint32_t trip_word28;	// L3150
  trip_word28 = v1615;	// L3151
  uint32_t v1616 = trip_word28;	// L3152
  uint16_t v1617;
  ap_int<32> v1617_tmp = v1616;
  v1617 = v1617_tmp(15, 0);	// L3153
  int32_t v1618 = v1617;	// L3154
  int32_t n_wavefront_row12;	// L3155
  n_wavefront_row12 = v1618;	// L3156
  int8_t weight12;	// L3157
  weight12 = 0;	// L3158
  int32_t mm_rows12;	// L3159
  mm_rows12 = 0;	// L3160
  int32_t row15;	// L3161
  row15 = -1;	// L3162
  int32_t v1619 = n_wavefront_row12;	// L3163
  int v1620 = v1619;	// L3164
  for (int v1621 = 0; v1621 < v1620; v1621 += 1) {	// L3165
    int32_t v1622 = row15;	// L3166
    ac_int<33, true> v1623 = v1622;	// L3167
    ac_int<33, true> v1624 = v1623 + 1;	// L3168
    int32_t v1625 = v1624;	// L3169
    row15 = v1625;	// L3170
    int32_t v1626 = row15;	// L3171
    int32_t v1627 = mm_rows12;	// L3172
    bool v1628 = v1626 >= v1627;	// L3173
    if (v1628) {	// L3174
      uint32_t v1629 = v1610.read();	// L3175
      uint32_t pe_word28;	// L3176
      pe_word28 = v1629;	// L3177
      uint32_t v1630 = pe_word28;	// L3178
      uint8_t v1631;
      ap_int<32> v1631_tmp = v1630;
      v1631 = v1631_tmp(7, 0);	// L3179
      weight12 = v1631;	// L3180
      uint32_t v1632 = pe_word28;	// L3181
      ac_int<12, false> v1633;
      ap_int<32> v1633_tmp = v1632;
      v1633 = v1633_tmp(19, 8);	// L3182
      int32_t v1634 = v1633;	// L3183
      mm_rows12 = v1634;	// L3184
      row15 = 0;	// L3185
    }
    int8_t activation13;	// L3187
    activation13 = 0;	// L3188
    uint32_t v1635 = v1611.read();	// L3189
    uint32_t activation_word3;	// L3190
    activation_word3 = v1635;	// L3191
    uint32_t v1636 = activation_word3;	// L3192
    uint8_t v1637;
    ap_int<32> v1637_tmp = v1636;
    v1637 = v1637_tmp(31, 24);	// L3193
    activation13 = v1637;	// L3194
    int32_t psum_north12;	// L3195
    psum_north12 = 0;	// L3196
    int32_t v1638 = v1612.read();	// L3197
    psum_north12 = v1638;	// L3198
    int8_t v1639 = activation13;	// L3199
    int16_t v1640 = v1639;	// L3200
    int16_t activation1612;	// L3201
    activation1612 = v1640;	// L3202
    int8_t v1641 = weight12;	// L3203
    int16_t v1642 = v1641;	// L3204
    int16_t weight1612;	// L3205
    weight1612 = v1642;	// L3206
    int32_t v1643 = psum_north12;	// L3207
    int16_t v1644 = activation1612;	// L3208
    int16_t v1645 = weight1612;	// L3209
    int32_t v1646 = v1644;	// L3210
    int32_t v1647 = v1645;	// L3211
    int32_t v1648 = v1646 * v1647;	// L3212
    ac_int<33, true> v1649 = v1643;	// L3213
    ac_int<33, true> v1650 = v1648;	// L3214
    ac_int<33, true> v1651 = v1649 + v1650;	// L3215
    int32_t v1652 = v1651;	// L3216
    int32_t psum12;	// L3217
    psum12 = v1652;	// L3218
    ac_int<128, false> result_word;	// L3219
    result_word = 0;	// L3220
    int32_t v1653 = psum12;	// L3221
    ac_int<128, false> v1654 = result_word;	// L3222
    ac_int<128, true> v1655;
    ap_int<128> v1655_tmp = v1654;
    v1655_tmp(31, 0) = v1653;
    v1655 = v1655_tmp;	// L3223
    result_word = v1655;	// L3224
    ac_int<128, false> v1656 = result_word;	// L3225
    v1613.write(v1656);	// L3226
    int8_t v1657 = activation13;	// L3227
    v1614.write(v1657);	// L3228
  }
}

void pe_3_1(
  ac_channel< uint32_t >& v1658,
  ac_channel< int8_t >& v1659,
  ac_channel< int32_t >& v1660,
  ac_channel< ac_int<128, false> >& v1661,
  ac_channel< ac_int<128, false> >& v1662,
  ac_channel< int8_t >& v1663
) {	// L3232
  uint32_t v1664 = v1658.read();	// L3246
  uint32_t trip_word29;	// L3247
  trip_word29 = v1664;	// L3248
  uint32_t v1665 = trip_word29;	// L3249
  uint16_t v1666;
  ap_int<32> v1666_tmp = v1665;
  v1666 = v1666_tmp(15, 0);	// L3250
  int32_t v1667 = v1666;	// L3251
  int32_t n_wavefront_row13;	// L3252
  n_wavefront_row13 = v1667;	// L3253
  int8_t weight13;	// L3254
  weight13 = 0;	// L3255
  int32_t mm_rows13;	// L3256
  mm_rows13 = 0;	// L3257
  int32_t row16;	// L3258
  row16 = -1;	// L3259
  int32_t v1668 = n_wavefront_row13;	// L3260
  int v1669 = v1668;	// L3261
  for (int v1670 = 0; v1670 < v1669; v1670 += 1) {	// L3262
    int32_t v1671 = row16;	// L3263
    ac_int<33, true> v1672 = v1671;	// L3264
    ac_int<33, true> v1673 = v1672 + 1;	// L3265
    int32_t v1674 = v1673;	// L3266
    row16 = v1674;	// L3267
    int32_t v1675 = row16;	// L3268
    int32_t v1676 = mm_rows13;	// L3269
    bool v1677 = v1675 >= v1676;	// L3270
    if (v1677) {	// L3271
      uint32_t v1678 = v1658.read();	// L3272
      uint32_t pe_word29;	// L3273
      pe_word29 = v1678;	// L3274
      uint32_t v1679 = pe_word29;	// L3275
      uint8_t v1680;
      ap_int<32> v1680_tmp = v1679;
      v1680 = v1680_tmp(7, 0);	// L3276
      weight13 = v1680;	// L3277
      uint32_t v1681 = pe_word29;	// L3278
      ac_int<12, false> v1682;
      ap_int<32> v1682_tmp = v1681;
      v1682 = v1682_tmp(19, 8);	// L3279
      int32_t v1683 = v1682;	// L3280
      mm_rows13 = v1683;	// L3281
      row16 = 0;	// L3282
    }
    int8_t activation14;	// L3284
    activation14 = 0;	// L3285
    int8_t v1684 = v1659.read();	// L3286
    activation14 = v1684;	// L3287
    int32_t psum_north13;	// L3288
    psum_north13 = 0;	// L3289
    int32_t v1685 = v1660.read();	// L3290
    psum_north13 = v1685;	// L3291
    int8_t v1686 = activation14;	// L3292
    int16_t v1687 = v1686;	// L3293
    int16_t activation1613;	// L3294
    activation1613 = v1687;	// L3295
    int8_t v1688 = weight13;	// L3296
    int16_t v1689 = v1688;	// L3297
    int16_t weight1613;	// L3298
    weight1613 = v1689;	// L3299
    int32_t v1690 = psum_north13;	// L3300
    int16_t v1691 = activation1613;	// L3301
    int16_t v1692 = weight1613;	// L3302
    int32_t v1693 = v1691;	// L3303
    int32_t v1694 = v1692;	// L3304
    int32_t v1695 = v1693 * v1694;	// L3305
    ac_int<33, true> v1696 = v1690;	// L3306
    ac_int<33, true> v1697 = v1695;	// L3307
    ac_int<33, true> v1698 = v1696 + v1697;	// L3308
    int32_t v1699 = v1698;	// L3309
    int32_t psum13;	// L3310
    psum13 = v1699;	// L3311
    ac_int<128, false> result_word1;	// L3312
    result_word1 = 0;	// L3313
    ac_int<128, false> v1700 = v1661.read();	// L3314
    result_word1 = v1700;	// L3315
    int32_t v1701 = psum13;	// L3316
    ac_int<128, false> v1702 = result_word1;	// L3317
    ac_int<128, true> v1703;
    ap_int<128> v1703_tmp = v1702;
    v1703_tmp(63, 32) = v1701;
    v1703 = v1703_tmp;	// L3318
    result_word1 = v1703;	// L3319
    ac_int<128, false> v1704 = result_word1;	// L3320
    v1662.write(v1704);	// L3321
    int8_t v1705 = activation14;	// L3322
    v1663.write(v1705);	// L3323
  }
}

void pe_3_2(
  ac_channel< uint32_t >& v1706,
  ac_channel< int8_t >& v1707,
  ac_channel< int32_t >& v1708,
  ac_channel< ac_int<128, false> >& v1709,
  ac_channel< ac_int<128, false> >& v1710,
  ac_channel< int8_t >& v1711
) {	// L3327
  uint32_t v1712 = v1706.read();	// L3341
  uint32_t trip_word30;	// L3342
  trip_word30 = v1712;	// L3343
  uint32_t v1713 = trip_word30;	// L3344
  uint16_t v1714;
  ap_int<32> v1714_tmp = v1713;
  v1714 = v1714_tmp(15, 0);	// L3345
  int32_t v1715 = v1714;	// L3346
  int32_t n_wavefront_row14;	// L3347
  n_wavefront_row14 = v1715;	// L3348
  int8_t weight14;	// L3349
  weight14 = 0;	// L3350
  int32_t mm_rows14;	// L3351
  mm_rows14 = 0;	// L3352
  int32_t row17;	// L3353
  row17 = -1;	// L3354
  int32_t v1716 = n_wavefront_row14;	// L3355
  int v1717 = v1716;	// L3356
  for (int v1718 = 0; v1718 < v1717; v1718 += 1) {	// L3357
    int32_t v1719 = row17;	// L3358
    ac_int<33, true> v1720 = v1719;	// L3359
    ac_int<33, true> v1721 = v1720 + 1;	// L3360
    int32_t v1722 = v1721;	// L3361
    row17 = v1722;	// L3362
    int32_t v1723 = row17;	// L3363
    int32_t v1724 = mm_rows14;	// L3364
    bool v1725 = v1723 >= v1724;	// L3365
    if (v1725) {	// L3366
      uint32_t v1726 = v1706.read();	// L3367
      uint32_t pe_word30;	// L3368
      pe_word30 = v1726;	// L3369
      uint32_t v1727 = pe_word30;	// L3370
      uint8_t v1728;
      ap_int<32> v1728_tmp = v1727;
      v1728 = v1728_tmp(7, 0);	// L3371
      weight14 = v1728;	// L3372
      uint32_t v1729 = pe_word30;	// L3373
      ac_int<12, false> v1730;
      ap_int<32> v1730_tmp = v1729;
      v1730 = v1730_tmp(19, 8);	// L3374
      int32_t v1731 = v1730;	// L3375
      mm_rows14 = v1731;	// L3376
      row17 = 0;	// L3377
    }
    int8_t activation15;	// L3379
    activation15 = 0;	// L3380
    int8_t v1732 = v1707.read();	// L3381
    activation15 = v1732;	// L3382
    int32_t psum_north14;	// L3383
    psum_north14 = 0;	// L3384
    int32_t v1733 = v1708.read();	// L3385
    psum_north14 = v1733;	// L3386
    int8_t v1734 = activation15;	// L3387
    int16_t v1735 = v1734;	// L3388
    int16_t activation1614;	// L3389
    activation1614 = v1735;	// L3390
    int8_t v1736 = weight14;	// L3391
    int16_t v1737 = v1736;	// L3392
    int16_t weight1614;	// L3393
    weight1614 = v1737;	// L3394
    int32_t v1738 = psum_north14;	// L3395
    int16_t v1739 = activation1614;	// L3396
    int16_t v1740 = weight1614;	// L3397
    int32_t v1741 = v1739;	// L3398
    int32_t v1742 = v1740;	// L3399
    int32_t v1743 = v1741 * v1742;	// L3400
    ac_int<33, true> v1744 = v1738;	// L3401
    ac_int<33, true> v1745 = v1743;	// L3402
    ac_int<33, true> v1746 = v1744 + v1745;	// L3403
    int32_t v1747 = v1746;	// L3404
    int32_t psum14;	// L3405
    psum14 = v1747;	// L3406
    ac_int<128, false> result_word2;	// L3407
    result_word2 = 0;	// L3408
    ac_int<128, false> v1748 = v1709.read();	// L3409
    result_word2 = v1748;	// L3410
    int32_t v1749 = psum14;	// L3411
    ac_int<128, false> v1750 = result_word2;	// L3412
    ac_int<128, true> v1751;
    ap_int<128> v1751_tmp = v1750;
    v1751_tmp(95, 64) = v1749;
    v1751 = v1751_tmp;	// L3413
    result_word2 = v1751;	// L3414
    ac_int<128, false> v1752 = result_word2;	// L3415
    v1710.write(v1752);	// L3416
    int8_t v1753 = activation15;	// L3417
    v1711.write(v1753);	// L3418
  }
}

void pe_3_3(
  ac_channel< uint32_t >& v1754,
  ac_channel< int8_t >& v1755,
  ac_channel< int32_t >& v1756,
  ac_channel< ac_int<128, false> >& v1757,
  ac_channel< ac_int<128, false> >& v1758
) {	// L3422
  uint32_t v1759 = v1754.read();	// L3436
  uint32_t trip_word31;	// L3437
  trip_word31 = v1759;	// L3438
  uint32_t v1760 = trip_word31;	// L3439
  uint16_t v1761;
  ap_int<32> v1761_tmp = v1760;
  v1761 = v1761_tmp(15, 0);	// L3440
  int32_t v1762 = v1761;	// L3441
  int32_t n_wavefront_row15;	// L3442
  n_wavefront_row15 = v1762;	// L3443
  int8_t weight15;	// L3444
  weight15 = 0;	// L3445
  int32_t mm_rows15;	// L3446
  mm_rows15 = 0;	// L3447
  int32_t row18;	// L3448
  row18 = -1;	// L3449
  int32_t v1763 = n_wavefront_row15;	// L3450
  int v1764 = v1763;	// L3451
  for (int v1765 = 0; v1765 < v1764; v1765 += 1) {	// L3452
    int32_t v1766 = row18;	// L3453
    ac_int<33, true> v1767 = v1766;	// L3454
    ac_int<33, true> v1768 = v1767 + 1;	// L3455
    int32_t v1769 = v1768;	// L3456
    row18 = v1769;	// L3457
    int32_t v1770 = row18;	// L3458
    int32_t v1771 = mm_rows15;	// L3459
    bool v1772 = v1770 >= v1771;	// L3460
    if (v1772) {	// L3461
      uint32_t v1773 = v1754.read();	// L3462
      uint32_t pe_word31;	// L3463
      pe_word31 = v1773;	// L3464
      uint32_t v1774 = pe_word31;	// L3465
      uint8_t v1775;
      ap_int<32> v1775_tmp = v1774;
      v1775 = v1775_tmp(7, 0);	// L3466
      weight15 = v1775;	// L3467
      uint32_t v1776 = pe_word31;	// L3468
      ac_int<12, false> v1777;
      ap_int<32> v1777_tmp = v1776;
      v1777 = v1777_tmp(19, 8);	// L3469
      int32_t v1778 = v1777;	// L3470
      mm_rows15 = v1778;	// L3471
      row18 = 0;	// L3472
    }
    int8_t activation17;	// L3474
    activation17 = 0;	// L3475
    int8_t v1779 = v1755.read();	// L3476
    activation17 = v1779;	// L3477
    int32_t psum_north15;	// L3478
    psum_north15 = 0;	// L3479
    int32_t v1780 = v1756.read();	// L3480
    psum_north15 = v1780;	// L3481
    int8_t v1781 = activation17;	// L3482
    int16_t v1782 = v1781;	// L3483
    int16_t activation1615;	// L3484
    activation1615 = v1782;	// L3485
    int8_t v1783 = weight15;	// L3486
    int16_t v1784 = v1783;	// L3487
    int16_t weight1615;	// L3488
    weight1615 = v1784;	// L3489
    int32_t v1785 = psum_north15;	// L3490
    int16_t v1786 = activation1615;	// L3491
    int16_t v1787 = weight1615;	// L3492
    int32_t v1788 = v1786;	// L3493
    int32_t v1789 = v1787;	// L3494
    int32_t v1790 = v1788 * v1789;	// L3495
    ac_int<33, true> v1791 = v1785;	// L3496
    ac_int<33, true> v1792 = v1790;	// L3497
    ac_int<33, true> v1793 = v1791 + v1792;	// L3498
    int32_t v1794 = v1793;	// L3499
    int32_t psum15;	// L3500
    psum15 = v1794;	// L3501
    ac_int<128, false> result_word3;	// L3502
    result_word3 = 0;	// L3503
    ac_int<128, false> v1795 = v1757.read();	// L3504
    result_word3 = v1795;	// L3505
    int32_t v1796 = psum15;	// L3506
    ac_int<128, false> v1797 = result_word3;	// L3507
    ac_int<128, true> v1798;
    ap_int<128> v1798_tmp = v1797;
    v1798_tmp(127, 96) = v1796;
    v1798 = v1798_tmp;	// L3508
    result_word3 = v1798;	// L3509
    ac_int<128, false> v1799 = result_word3;	// L3510
    v1758.write(v1799);	// L3511
  }
}

void accu_0(
  ac_channel< uint64_t >& v1800,
  ac_channel< ac_int<128, false> >& v1801,
  ac_channel< uint32_t >& v1802
) {	// L3515
  ac_int<128, false> ar[128];	// L3552
  uint64_t v1803 = v1800.read();	// L3553
  uint64_t count_word3;	// L3554
  count_word3 = v1803;	// L3555
  uint64_t v1804 = count_word3;	// L3556
  uint16_t v1805;
  ap_int<64> v1805_tmp = v1804;
  v1805 = v1805_tmp(15, 0);	// L3557
  int32_t v1806 = v1805;	// L3558
  int32_t n_step;	// L3559
  n_step = v1806;	// L3560
  int32_t op3;	// L3561
  op3 = 0;	// L3562
  int32_t f01;	// L3563
  f01 = 0;	// L3564
  int32_t f12;	// L3565
  f12 = 0;	// L3566
  int32_t f21;	// L3567
  f21 = 0;	// L3568
  int32_t instr_steps;	// L3569
  instr_steps = 0;	// L3570
  int32_t step;	// L3571
  step = -1;	// L3572
  ac_int<128, false> vadd_first;	// L3573
  vadd_first = 0;	// L3574
  int32_t v1807 = n_step;	// L3575
  int v1808 = v1807;	// L3576
  for (int v1809 = 0; v1809 < v1808; v1809 += 1) {	// L3577
    int32_t v1810 = step;	// L3578
    ac_int<33, true> v1811 = v1810;	// L3579
    ac_int<33, true> v1812 = v1811 + 1;	// L3580
    int32_t v1813 = v1812;	// L3581
    step = v1813;	// L3582
    int32_t v1814 = step;	// L3583
    int32_t v1815 = instr_steps;	// L3584
    bool v1816 = v1814 >= v1815;	// L3585
    if (v1816) {	// L3586
      uint64_t v1817 = v1800.read();	// L3587
      uint64_t word3;	// L3588
      word3 = v1817;	// L3589
      uint64_t v1818 = word3;	// L3590
      ac_int<6, false> v1819;
      ap_int<64> v1819_tmp = v1818;
      v1819 = v1819_tmp(5, 0);	// L3591
      int32_t v1820 = v1819;	// L3592
      op3 = v1820;	// L3593
      uint64_t v1821 = word3;	// L3594
      ac_int<12, false> v1822;
      ap_int<64> v1822_tmp = v1821;
      v1822 = v1822_tmp(17, 6);	// L3595
      int32_t v1823 = v1822;	// L3596
      f01 = v1823;	// L3597
      uint64_t v1824 = word3;	// L3598
      ac_int<12, false> v1825;
      ap_int<64> v1825_tmp = v1824;
      v1825 = v1825_tmp(29, 18);	// L3599
      int32_t v1826 = v1825;	// L3600
      f12 = v1826;	// L3601
      uint64_t v1827 = word3;	// L3602
      ac_int<12, false> v1828;
      ap_int<64> v1828_tmp = v1827;
      v1828 = v1828_tmp(41, 30);	// L3603
      int32_t v1829 = v1828;	// L3604
      f21 = v1829;	// L3605
      uint64_t v1830 = word3;	// L3606
      uint8_t v1831;
      ap_int<64> v1831_tmp = v1830;
      v1831 = v1831_tmp(61, 54);	// L3607
      int32_t v1832 = v1831;	// L3608
      instr_steps = v1832;	// L3609
      step = 0;	// L3610
    }
    int32_t v1833 = step;	// L3612
    int32_t row19;	// L3613
    row19 = v1833;	// L3614
    int32_t phase;	// L3615
    phase = 0;	// L3616
    int32_t two_source;	// L3617
    two_source = 0;	// L3618
    int32_t v1834 = op3;	// L3619
    bool v1835 = v1834 == 5;	// L3620
    if (v1835) {	// L3621
      two_source = 1;	// L3622
    }
    int32_t v1836 = op3;	// L3624
    bool v1837 = v1836 == 10;	// L3625
    if (v1837) {	// L3626
      two_source = 1;	// L3627
    }
    int32_t v1838 = two_source;	// L3629
    bool v1839 = v1838 == 1;	// L3630
    if (v1839) {	// L3631
      int32_t v1840 = step;	// L3632
      int32_t v1841 = v1840 >> 1;	// L3633
      row19 = v1841;	// L3634
      int32_t v1842 = step;	// L3635
      int32_t v1843 = row19;	// L3636
      int32_t v1844 = v1843 << 1;	// L3637
      ac_int<33, true> v1845 = v1842;	// L3638
      ac_int<33, true> v1846 = v1844;	// L3639
      ac_int<33, true> v1847 = v1845 - v1846;	// L3640
      int32_t v1848 = v1847;	// L3641
      phase = v1848;	// L3642
    }
    int32_t v1849 = f12;	// L3644
    int32_t v1850 = row19;	// L3645
    ac_int<33, true> v1851 = v1849;	// L3646
    ac_int<33, true> v1852 = v1850;	// L3647
    ac_int<33, true> v1853 = v1851 + v1852;	// L3648
    int32_t v1854 = v1853;	// L3649
    int32_t read_row1;	// L3650
    read_row1 = v1854;	// L3651
    int32_t v1855 = f01;	// L3652
    int32_t v1856 = row19;	// L3653
    ac_int<33, true> v1857 = v1855;	// L3654
    ac_int<33, true> v1858 = v1856;	// L3655
    ac_int<33, true> v1859 = v1857 + v1858;	// L3656
    int32_t v1860 = v1859;	// L3657
    int32_t write_row1;	// L3658
    write_row1 = v1860;	// L3659
    int32_t v1861 = op3;	// L3660
    bool v1862 = v1861 == 4;	// L3661
    if (v1862) {	// L3662
      int32_t v1863 = f12;	// L3663
      int32_t v1864 = row19;	// L3664
      ac_int<33, true> v1865 = v1863;	// L3665
      ac_int<33, true> v1866 = v1864;	// L3666
      ac_int<33, true> v1867 = v1865 + v1866;	// L3667
      int32_t v1868 = v1867;	// L3668
      write_row1 = v1868;	// L3669
    }
    int32_t v1869 = op3;	// L3671
    bool v1870 = v1869 == 7;	// L3672
    if (v1870) {	// L3673
      int32_t v1871 = f01;	// L3674
      int32_t v1872 = row19;	// L3675
      ac_int<33, true> v1873 = v1871;	// L3676
      ac_int<33, true> v1874 = v1872;	// L3677
      ac_int<33, true> v1875 = v1873 + v1874;	// L3678
      int32_t v1876 = v1875;	// L3679
      read_row1 = v1876;	// L3680
    }
    int32_t v1877 = two_source;	// L3682
    bool v1878 = v1877 == 1;	// L3683
    if (v1878) {	// L3684
      int32_t v1879 = phase;	// L3685
      bool v1880 = v1879 == 1;	// L3686
      if (v1880) {	// L3687
        int32_t v1881 = f21;	// L3688
        int32_t v1882 = row19;	// L3689
        ac_int<33, true> v1883 = v1881;	// L3690
        ac_int<33, true> v1884 = v1882;	// L3691
        ac_int<33, true> v1885 = v1883 + v1884;	// L3692
        int32_t v1886 = v1885;	// L3693
        read_row1 = v1886;	// L3694
      }
    }
    int32_t v1887 = read_row1;	// L3697
    int v1888 = v1887;	// L3698
    ac_int<128, false> v1889 = ar[v1888];	// L3699
    ac_int<128, false> read_word;	// L3700
    read_word = v1889;	// L3701
    ac_int<128, false> write_word1;	// L3702
    write_word1 = 0;	// L3703
    int32_t do_write;	// L3704
    do_write = 1;	// L3705
    int32_t v1890 = op3;	// L3706
    bool v1891 = v1890 == 4;	// L3707
    if (v1891) {	// L3708
      ac_int<128, false> v1892 = v1801.read();	// L3709
      ac_int<128, false> array_word;	// L3710
      array_word = v1892;	// L3711
      ac_int<128, false> base;	// L3712
      base = 0;	// L3713
      int32_t v1893 = f21;	// L3714
      bool v1894 = v1893 == 1;	// L3715
      if (v1894) {	// L3716
        ac_int<128, false> v1895 = read_word;	// L3717
        base = v1895;	// L3718
      }
      ac_int<128, false> v1896 = base;	// L3720
      uint32_t v1897;
      ap_int<128> v1897_tmp = v1896;
      v1897 = v1897_tmp(31, 0);	// L3721
      int32_t base_lane;	// L3722
      base_lane = v1897;	// L3723
      ac_int<128, false> v1898 = array_word;	// L3724
      uint32_t v1899;
      ap_int<128> v1899_tmp = v1898;
      v1899 = v1899_tmp(31, 0);	// L3725
      int32_t array_lane;	// L3726
      array_lane = v1899;	// L3727
      int32_t v1900 = base_lane;	// L3728
      int32_t v1901 = array_lane;	// L3729
      ac_int<33, true> v1902 = v1900;	// L3730
      ac_int<33, true> v1903 = v1901;	// L3731
      ac_int<33, true> v1904 = v1902 + v1903;	// L3732
      int32_t v1905 = v1904;	// L3733
      int32_t summed;	// L3734
      summed = v1905;	// L3735
      int32_t v1906 = summed;	// L3736
      ac_int<128, false> v1907 = write_word1;	// L3737
      ac_int<128, true> v1908;
      ap_int<128> v1908_tmp = v1907;
      v1908_tmp(31, 0) = v1906;
      v1908 = v1908_tmp;	// L3738
      write_word1 = v1908;	// L3739
      ac_int<128, false> v1909 = base;	// L3740
      uint32_t v1910;
      ap_int<128> v1910_tmp = v1909;
      v1910 = v1910_tmp(63, 32);	// L3741
      int32_t base_lane1;	// L3742
      base_lane1 = v1910;	// L3743
      ac_int<128, false> v1911 = array_word;	// L3744
      uint32_t v1912;
      ap_int<128> v1912_tmp = v1911;
      v1912 = v1912_tmp(63, 32);	// L3745
      int32_t array_lane1;	// L3746
      array_lane1 = v1912;	// L3747
      int32_t v1913 = base_lane1;	// L3748
      int32_t v1914 = array_lane1;	// L3749
      ac_int<33, true> v1915 = v1913;	// L3750
      ac_int<33, true> v1916 = v1914;	// L3751
      ac_int<33, true> v1917 = v1915 + v1916;	// L3752
      int32_t v1918 = v1917;	// L3753
      int32_t summed1;	// L3754
      summed1 = v1918;	// L3755
      int32_t v1919 = summed1;	// L3756
      ac_int<128, false> v1920 = write_word1;	// L3757
      ac_int<128, true> v1921;
      ap_int<128> v1921_tmp = v1920;
      v1921_tmp(63, 32) = v1919;
      v1921 = v1921_tmp;	// L3758
      write_word1 = v1921;	// L3759
      ac_int<128, false> v1922 = base;	// L3760
      uint32_t v1923;
      ap_int<128> v1923_tmp = v1922;
      v1923 = v1923_tmp(95, 64);	// L3761
      int32_t base_lane2;	// L3762
      base_lane2 = v1923;	// L3763
      ac_int<128, false> v1924 = array_word;	// L3764
      uint32_t v1925;
      ap_int<128> v1925_tmp = v1924;
      v1925 = v1925_tmp(95, 64);	// L3765
      int32_t array_lane2;	// L3766
      array_lane2 = v1925;	// L3767
      int32_t v1926 = base_lane2;	// L3768
      int32_t v1927 = array_lane2;	// L3769
      ac_int<33, true> v1928 = v1926;	// L3770
      ac_int<33, true> v1929 = v1927;	// L3771
      ac_int<33, true> v1930 = v1928 + v1929;	// L3772
      int32_t v1931 = v1930;	// L3773
      int32_t summed2;	// L3774
      summed2 = v1931;	// L3775
      int32_t v1932 = summed2;	// L3776
      ac_int<128, false> v1933 = write_word1;	// L3777
      ac_int<128, true> v1934;
      ap_int<128> v1934_tmp = v1933;
      v1934_tmp(95, 64) = v1932;
      v1934 = v1934_tmp;	// L3778
      write_word1 = v1934;	// L3779
      ac_int<128, false> v1935 = base;	// L3780
      uint32_t v1936;
      ap_int<128> v1936_tmp = v1935;
      v1936 = v1936_tmp(127, 96);	// L3781
      int32_t base_lane3;	// L3782
      base_lane3 = v1936;	// L3783
      ac_int<128, false> v1937 = array_word;	// L3784
      uint32_t v1938;
      ap_int<128> v1938_tmp = v1937;
      v1938 = v1938_tmp(127, 96);	// L3785
      int32_t array_lane3;	// L3786
      array_lane3 = v1938;	// L3787
      int32_t v1939 = base_lane3;	// L3788
      int32_t v1940 = array_lane3;	// L3789
      ac_int<33, true> v1941 = v1939;	// L3790
      ac_int<33, true> v1942 = v1940;	// L3791
      ac_int<33, true> v1943 = v1941 + v1942;	// L3792
      int32_t v1944 = v1943;	// L3793
      int32_t summed3;	// L3794
      summed3 = v1944;	// L3795
      int32_t v1945 = summed3;	// L3796
      ac_int<128, false> v1946 = write_word1;	// L3797
      ac_int<128, true> v1947;
      ap_int<128> v1947_tmp = v1946;
      v1947_tmp(127, 96) = v1945;
      v1947 = v1947_tmp;	// L3798
      write_word1 = v1947;	// L3799
    } else {
      int32_t v1948 = op3;	// L3801
      bool v1949 = v1948 == 5;	// L3802
      if (v1949) {	// L3803
        int32_t v1950 = phase;	// L3804
        bool v1951 = v1950 == 0;	// L3805
        if (v1951) {	// L3806
          ac_int<128, false> v1952 = read_word;	// L3807
          vadd_first = v1952;	// L3808
          do_write = 0;	// L3809
        } else {
          ac_int<128, false> v1953 = vadd_first;	// L3811
          uint32_t v1954;
          ap_int<128> v1954_tmp = v1953;
          v1954 = v1954_tmp(31, 0);	// L3812
          int32_t first;	// L3813
          first = v1954;	// L3814
          ac_int<128, false> v1955 = read_word;	// L3815
          uint32_t v1956;
          ap_int<128> v1956_tmp = v1955;
          v1956 = v1956_tmp(31, 0);	// L3816
          int32_t second;	// L3817
          second = v1956;	// L3818
          int32_t v1957 = first;	// L3819
          int32_t v1958 = second;	// L3820
          ac_int<33, true> v1959 = v1957;	// L3821
          ac_int<33, true> v1960 = v1958;	// L3822
          ac_int<33, true> v1961 = v1959 + v1960;	// L3823
          int32_t v1962 = v1961;	// L3824
          int32_t added;	// L3825
          added = v1962;	// L3826
          int32_t v1963 = added;	// L3827
          ac_int<128, false> v1964 = write_word1;	// L3828
          ac_int<128, true> v1965;
          ap_int<128> v1965_tmp = v1964;
          v1965_tmp(31, 0) = v1963;
          v1965 = v1965_tmp;	// L3829
          write_word1 = v1965;	// L3830
          ac_int<128, false> v1966 = vadd_first;	// L3831
          uint32_t v1967;
          ap_int<128> v1967_tmp = v1966;
          v1967 = v1967_tmp(63, 32);	// L3832
          int32_t first1;	// L3833
          first1 = v1967;	// L3834
          ac_int<128, false> v1968 = read_word;	// L3835
          uint32_t v1969;
          ap_int<128> v1969_tmp = v1968;
          v1969 = v1969_tmp(63, 32);	// L3836
          int32_t second1;	// L3837
          second1 = v1969;	// L3838
          int32_t v1970 = first1;	// L3839
          int32_t v1971 = second1;	// L3840
          ac_int<33, true> v1972 = v1970;	// L3841
          ac_int<33, true> v1973 = v1971;	// L3842
          ac_int<33, true> v1974 = v1972 + v1973;	// L3843
          int32_t v1975 = v1974;	// L3844
          int32_t added1;	// L3845
          added1 = v1975;	// L3846
          int32_t v1976 = added1;	// L3847
          ac_int<128, false> v1977 = write_word1;	// L3848
          ac_int<128, true> v1978;
          ap_int<128> v1978_tmp = v1977;
          v1978_tmp(63, 32) = v1976;
          v1978 = v1978_tmp;	// L3849
          write_word1 = v1978;	// L3850
          ac_int<128, false> v1979 = vadd_first;	// L3851
          uint32_t v1980;
          ap_int<128> v1980_tmp = v1979;
          v1980 = v1980_tmp(95, 64);	// L3852
          int32_t first2;	// L3853
          first2 = v1980;	// L3854
          ac_int<128, false> v1981 = read_word;	// L3855
          uint32_t v1982;
          ap_int<128> v1982_tmp = v1981;
          v1982 = v1982_tmp(95, 64);	// L3856
          int32_t second2;	// L3857
          second2 = v1982;	// L3858
          int32_t v1983 = first2;	// L3859
          int32_t v1984 = second2;	// L3860
          ac_int<33, true> v1985 = v1983;	// L3861
          ac_int<33, true> v1986 = v1984;	// L3862
          ac_int<33, true> v1987 = v1985 + v1986;	// L3863
          int32_t v1988 = v1987;	// L3864
          int32_t added2;	// L3865
          added2 = v1988;	// L3866
          int32_t v1989 = added2;	// L3867
          ac_int<128, false> v1990 = write_word1;	// L3868
          ac_int<128, true> v1991;
          ap_int<128> v1991_tmp = v1990;
          v1991_tmp(95, 64) = v1989;
          v1991 = v1991_tmp;	// L3869
          write_word1 = v1991;	// L3870
          ac_int<128, false> v1992 = vadd_first;	// L3871
          uint32_t v1993;
          ap_int<128> v1993_tmp = v1992;
          v1993 = v1993_tmp(127, 96);	// L3872
          int32_t first3;	// L3873
          first3 = v1993;	// L3874
          ac_int<128, false> v1994 = read_word;	// L3875
          uint32_t v1995;
          ap_int<128> v1995_tmp = v1994;
          v1995 = v1995_tmp(127, 96);	// L3876
          int32_t second3;	// L3877
          second3 = v1995;	// L3878
          int32_t v1996 = first3;	// L3879
          int32_t v1997 = second3;	// L3880
          ac_int<33, true> v1998 = v1996;	// L3881
          ac_int<33, true> v1999 = v1997;	// L3882
          ac_int<33, true> v2000 = v1998 + v1999;	// L3883
          int32_t v2001 = v2000;	// L3884
          int32_t added3;	// L3885
          added3 = v2001;	// L3886
          int32_t v2002 = added3;	// L3887
          ac_int<128, false> v2003 = write_word1;	// L3888
          ac_int<128, true> v2004;
          ap_int<128> v2004_tmp = v2003;
          v2004_tmp(127, 96) = v2002;
          v2004 = v2004_tmp;	// L3889
          write_word1 = v2004;	// L3890
        }
      } else {
        int32_t v2005 = op3;	// L3893
        bool v2006 = v2005 == 10;	// L3894
        if (v2006) {	// L3895
          int32_t v2007 = phase;	// L3896
          bool v2008 = v2007 == 0;	// L3897
          if (v2008) {	// L3898
            ac_int<128, false> v2009 = read_word;	// L3899
            vadd_first = v2009;	// L3900
            do_write = 0;	// L3901
          } else {
            ac_int<128, false> v2010 = vadd_first;	// L3903
            uint32_t v2011;
            ap_int<128> v2011_tmp = v2010;
            v2011 = v2011_tmp(31, 0);	// L3904
            int32_t held;	// L3905
            held = v2011;	// L3906
            ac_int<128, false> v2012 = read_word;	// L3907
            uint32_t v2013;
            ap_int<128> v2013_tmp = v2012;
            v2013 = v2013_tmp(31, 0);	// L3908
            int32_t arriving;	// L3909
            arriving = v2013;	// L3910
            int32_t v2014 = held;	// L3911
            int32_t v2015 = arriving;	// L3912
            ac_int<33, true> v2016 = v2014;	// L3913
            ac_int<33, true> v2017 = v2015;	// L3914
            ac_int<33, true> v2018 = v2016 + v2017;	// L3915
            int32_t v2019 = v2018;	// L3916
            int32_t fused;	// L3917
            fused = v2019;	// L3918
            int32_t v2020 = fused;	// L3919
            bool v2021 = v2020 < 0;	// L3920
            if (v2021) {	// L3921
              fused = 0;	// L3922
            }
            int32_t v2022 = fused;	// L3924
            ac_int<128, false> v2023 = write_word1;	// L3925
            ac_int<128, true> v2024;
            ap_int<128> v2024_tmp = v2023;
            v2024_tmp(31, 0) = v2022;
            v2024 = v2024_tmp;	// L3926
            write_word1 = v2024;	// L3927
            ac_int<128, false> v2025 = vadd_first;	// L3928
            uint32_t v2026;
            ap_int<128> v2026_tmp = v2025;
            v2026 = v2026_tmp(63, 32);	// L3929
            int32_t held1;	// L3930
            held1 = v2026;	// L3931
            ac_int<128, false> v2027 = read_word;	// L3932
            uint32_t v2028;
            ap_int<128> v2028_tmp = v2027;
            v2028 = v2028_tmp(63, 32);	// L3933
            int32_t arriving1;	// L3934
            arriving1 = v2028;	// L3935
            int32_t v2029 = held1;	// L3936
            int32_t v2030 = arriving1;	// L3937
            ac_int<33, true> v2031 = v2029;	// L3938
            ac_int<33, true> v2032 = v2030;	// L3939
            ac_int<33, true> v2033 = v2031 + v2032;	// L3940
            int32_t v2034 = v2033;	// L3941
            int32_t fused1;	// L3942
            fused1 = v2034;	// L3943
            int32_t v2035 = fused1;	// L3944
            bool v2036 = v2035 < 0;	// L3945
            if (v2036) {	// L3946
              fused1 = 0;	// L3947
            }
            int32_t v2037 = fused1;	// L3949
            ac_int<128, false> v2038 = write_word1;	// L3950
            ac_int<128, true> v2039;
            ap_int<128> v2039_tmp = v2038;
            v2039_tmp(63, 32) = v2037;
            v2039 = v2039_tmp;	// L3951
            write_word1 = v2039;	// L3952
            ac_int<128, false> v2040 = vadd_first;	// L3953
            uint32_t v2041;
            ap_int<128> v2041_tmp = v2040;
            v2041 = v2041_tmp(95, 64);	// L3954
            int32_t held2;	// L3955
            held2 = v2041;	// L3956
            ac_int<128, false> v2042 = read_word;	// L3957
            uint32_t v2043;
            ap_int<128> v2043_tmp = v2042;
            v2043 = v2043_tmp(95, 64);	// L3958
            int32_t arriving2;	// L3959
            arriving2 = v2043;	// L3960
            int32_t v2044 = held2;	// L3961
            int32_t v2045 = arriving2;	// L3962
            ac_int<33, true> v2046 = v2044;	// L3963
            ac_int<33, true> v2047 = v2045;	// L3964
            ac_int<33, true> v2048 = v2046 + v2047;	// L3965
            int32_t v2049 = v2048;	// L3966
            int32_t fused2;	// L3967
            fused2 = v2049;	// L3968
            int32_t v2050 = fused2;	// L3969
            bool v2051 = v2050 < 0;	// L3970
            if (v2051) {	// L3971
              fused2 = 0;	// L3972
            }
            int32_t v2052 = fused2;	// L3974
            ac_int<128, false> v2053 = write_word1;	// L3975
            ac_int<128, true> v2054;
            ap_int<128> v2054_tmp = v2053;
            v2054_tmp(95, 64) = v2052;
            v2054 = v2054_tmp;	// L3976
            write_word1 = v2054;	// L3977
            ac_int<128, false> v2055 = vadd_first;	// L3978
            uint32_t v2056;
            ap_int<128> v2056_tmp = v2055;
            v2056 = v2056_tmp(127, 96);	// L3979
            int32_t held3;	// L3980
            held3 = v2056;	// L3981
            ac_int<128, false> v2057 = read_word;	// L3982
            uint32_t v2058;
            ap_int<128> v2058_tmp = v2057;
            v2058 = v2058_tmp(127, 96);	// L3983
            int32_t arriving3;	// L3984
            arriving3 = v2058;	// L3985
            int32_t v2059 = held3;	// L3986
            int32_t v2060 = arriving3;	// L3987
            ac_int<33, true> v2061 = v2059;	// L3988
            ac_int<33, true> v2062 = v2060;	// L3989
            ac_int<33, true> v2063 = v2061 + v2062;	// L3990
            int32_t v2064 = v2063;	// L3991
            int32_t fused3;	// L3992
            fused3 = v2064;	// L3993
            int32_t v2065 = fused3;	// L3994
            bool v2066 = v2065 < 0;	// L3995
            if (v2066) {	// L3996
              fused3 = 0;	// L3997
            }
            int32_t v2067 = fused3;	// L3999
            ac_int<128, false> v2068 = write_word1;	// L4000
            ac_int<128, true> v2069;
            ap_int<128> v2069_tmp = v2068;
            v2069_tmp(127, 96) = v2067;
            v2069 = v2069_tmp;	// L4001
            write_word1 = v2069;	// L4002
          }
        } else {
          int32_t v2070 = op3;	// L4005
          bool v2071 = v2070 == 6;	// L4006
          if (v2071) {	// L4007
            ac_int<128, false> v2072 = read_word;	// L4008
            uint32_t v2073;
            ap_int<128> v2073_tmp = v2072;
            v2073 = v2073_tmp(31, 0);	// L4009
            int32_t before;	// L4010
            before = v2073;	// L4011
            int32_t v2074 = before;	// L4012
            int32_t rectified;	// L4013
            rectified = v2074;	// L4014
            int32_t v2075 = rectified;	// L4015
            bool v2076 = v2075 < 0;	// L4016
            if (v2076) {	// L4017
              rectified = 0;	// L4018
            }
            int32_t v2077 = rectified;	// L4020
            ac_int<128, false> v2078 = write_word1;	// L4021
            ac_int<128, true> v2079;
            ap_int<128> v2079_tmp = v2078;
            v2079_tmp(31, 0) = v2077;
            v2079 = v2079_tmp;	// L4022
            write_word1 = v2079;	// L4023
            ac_int<128, false> v2080 = read_word;	// L4024
            uint32_t v2081;
            ap_int<128> v2081_tmp = v2080;
            v2081 = v2081_tmp(63, 32);	// L4025
            int32_t before1;	// L4026
            before1 = v2081;	// L4027
            int32_t v2082 = before1;	// L4028
            int32_t rectified1;	// L4029
            rectified1 = v2082;	// L4030
            int32_t v2083 = rectified1;	// L4031
            bool v2084 = v2083 < 0;	// L4032
            if (v2084) {	// L4033
              rectified1 = 0;	// L4034
            }
            int32_t v2085 = rectified1;	// L4036
            ac_int<128, false> v2086 = write_word1;	// L4037
            ac_int<128, true> v2087;
            ap_int<128> v2087_tmp = v2086;
            v2087_tmp(63, 32) = v2085;
            v2087 = v2087_tmp;	// L4038
            write_word1 = v2087;	// L4039
            ac_int<128, false> v2088 = read_word;	// L4040
            uint32_t v2089;
            ap_int<128> v2089_tmp = v2088;
            v2089 = v2089_tmp(95, 64);	// L4041
            int32_t before2;	// L4042
            before2 = v2089;	// L4043
            int32_t v2090 = before2;	// L4044
            int32_t rectified2;	// L4045
            rectified2 = v2090;	// L4046
            int32_t v2091 = rectified2;	// L4047
            bool v2092 = v2091 < 0;	// L4048
            if (v2092) {	// L4049
              rectified2 = 0;	// L4050
            }
            int32_t v2093 = rectified2;	// L4052
            ac_int<128, false> v2094 = write_word1;	// L4053
            ac_int<128, true> v2095;
            ap_int<128> v2095_tmp = v2094;
            v2095_tmp(95, 64) = v2093;
            v2095 = v2095_tmp;	// L4054
            write_word1 = v2095;	// L4055
            ac_int<128, false> v2096 = read_word;	// L4056
            uint32_t v2097;
            ap_int<128> v2097_tmp = v2096;
            v2097 = v2097_tmp(127, 96);	// L4057
            int32_t before3;	// L4058
            before3 = v2097;	// L4059
            int32_t v2098 = before3;	// L4060
            int32_t rectified3;	// L4061
            rectified3 = v2098;	// L4062
            int32_t v2099 = rectified3;	// L4063
            bool v2100 = v2099 < 0;	// L4064
            if (v2100) {	// L4065
              rectified3 = 0;	// L4066
            }
            int32_t v2101 = rectified3;	// L4068
            ac_int<128, false> v2102 = write_word1;	// L4069
            ac_int<128, true> v2103;
            ap_int<128> v2103_tmp = v2102;
            v2103_tmp(127, 96) = v2101;
            v2103 = v2103_tmp;	// L4070
            write_word1 = v2103;	// L4071
          } else {
            do_write = 0;	// L4073
            uint32_t clipped_word;	// L4074
            clipped_word = 0;	// L4075
            ac_int<128, false> v2104 = read_word;	// L4076
            uint32_t v2105;
            ap_int<128> v2105_tmp = v2104;
            v2105 = v2105_tmp(31, 0);	// L4077
            int32_t retiring;	// L4078
            retiring = v2105;	// L4079
            int32_t v2106 = retiring;	// L4080
            bool v2107 = v2106 > 127;	// L4081
            if (v2107) {	// L4082
              retiring = 127;	// L4083
            }
            int32_t v2108 = retiring;	// L4085
            bool v2109 = v2108 < -128;	// L4086
            if (v2109) {	// L4087
              retiring = -128;	// L4088
            }
            int32_t v2110 = retiring;	// L4090
            int8_t v2111 = v2110;	// L4091
            int8_t clipped;	// L4092
            clipped = v2111;	// L4093
            int8_t v2112 = clipped;	// L4094
            uint32_t v2113 = clipped_word;	// L4095
            int32_t v2114;
            ap_int<32> v2114_tmp = v2113;
            v2114_tmp(7, 0) = v2112;
            v2114 = v2114_tmp;	// L4096
            clipped_word = v2114;	// L4097
            ac_int<128, false> v2115 = read_word;	// L4098
            uint32_t v2116;
            ap_int<128> v2116_tmp = v2115;
            v2116 = v2116_tmp(63, 32);	// L4099
            int32_t retiring1;	// L4100
            retiring1 = v2116;	// L4101
            int32_t v2117 = retiring1;	// L4102
            bool v2118 = v2117 > 127;	// L4103
            if (v2118) {	// L4104
              retiring1 = 127;	// L4105
            }
            int32_t v2119 = retiring1;	// L4107
            bool v2120 = v2119 < -128;	// L4108
            if (v2120) {	// L4109
              retiring1 = -128;	// L4110
            }
            int32_t v2121 = retiring1;	// L4112
            int8_t v2122 = v2121;	// L4113
            int8_t clipped1;	// L4114
            clipped1 = v2122;	// L4115
            int8_t v2123 = clipped1;	// L4116
            uint32_t v2124 = clipped_word;	// L4117
            int32_t v2125;
            ap_int<32> v2125_tmp = v2124;
            v2125_tmp(15, 8) = v2123;
            v2125 = v2125_tmp;	// L4118
            clipped_word = v2125;	// L4119
            ac_int<128, false> v2126 = read_word;	// L4120
            uint32_t v2127;
            ap_int<128> v2127_tmp = v2126;
            v2127 = v2127_tmp(95, 64);	// L4121
            int32_t retiring2;	// L4122
            retiring2 = v2127;	// L4123
            int32_t v2128 = retiring2;	// L4124
            bool v2129 = v2128 > 127;	// L4125
            if (v2129) {	// L4126
              retiring2 = 127;	// L4127
            }
            int32_t v2130 = retiring2;	// L4129
            bool v2131 = v2130 < -128;	// L4130
            if (v2131) {	// L4131
              retiring2 = -128;	// L4132
            }
            int32_t v2132 = retiring2;	// L4134
            int8_t v2133 = v2132;	// L4135
            int8_t clipped2;	// L4136
            clipped2 = v2133;	// L4137
            int8_t v2134 = clipped2;	// L4138
            uint32_t v2135 = clipped_word;	// L4139
            int32_t v2136;
            ap_int<32> v2136_tmp = v2135;
            v2136_tmp(23, 16) = v2134;
            v2136 = v2136_tmp;	// L4140
            clipped_word = v2136;	// L4141
            ac_int<128, false> v2137 = read_word;	// L4142
            uint32_t v2138;
            ap_int<128> v2138_tmp = v2137;
            v2138 = v2138_tmp(127, 96);	// L4143
            int32_t retiring3;	// L4144
            retiring3 = v2138;	// L4145
            int32_t v2139 = retiring3;	// L4146
            bool v2140 = v2139 > 127;	// L4147
            if (v2140) {	// L4148
              retiring3 = 127;	// L4149
            }
            int32_t v2141 = retiring3;	// L4151
            bool v2142 = v2141 < -128;	// L4152
            if (v2142) {	// L4153
              retiring3 = -128;	// L4154
            }
            int32_t v2143 = retiring3;	// L4156
            int8_t v2144 = v2143;	// L4157
            int8_t clipped3;	// L4158
            clipped3 = v2144;	// L4159
            int8_t v2145 = clipped3;	// L4160
            uint32_t v2146 = clipped_word;	// L4161
            int32_t v2147;
            ap_int<32> v2147_tmp = v2146;
            v2147_tmp(31, 24) = v2145;
            v2147 = v2147_tmp;	// L4162
            clipped_word = v2147;	// L4163
            uint32_t v2148 = clipped_word;	// L4164
            v1802.write(v2148);	// L4165
          }
        }
      }
    }
    int32_t v2149 = do_write;	// L4170
    bool v2150 = v2149 == 1;	// L4171
    if (v2150) {	// L4172
      ac_int<128, false> v2151 = write_word1;	// L4173
      int32_t v2152 = write_row1;	// L4174
      int v2153 = v2152;	// L4175
      ar[v2153] = v2151;	// L4176
    }
  }
}

void dma_st_0(
  int8_t v2154[256],
  ac_channel< uint64_t >& v2155,
  ac_channel< uint32_t >& v2156
) {	// L4181
  uint64_t v2157 = v2155.read();	// L4205
  uint64_t count_word4;	// L4206
  count_word4 = v2157;	// L4207
  uint64_t v2158 = count_word4;	// L4208
  uint16_t v2159;
  ap_int<64> v2159_tmp = v2158;
  v2159 = v2159_tmp(15, 0);	// L4209
  int32_t v2160 = v2159;	// L4210
  int32_t n_row2;	// L4211
  n_row2 = v2160;	// L4212
  int32_t dram_row01;	// L4213
  dram_row01 = 0;	// L4214
  int32_t col_block1;	// L4215
  col_block1 = 0;	// L4216
  int32_t instr_rows3;	// L4217
  instr_rows3 = 0;	// L4218
  int32_t row20;	// L4219
  row20 = -1;	// L4220
  int32_t v2161 = n_row2;	// L4221
  int v2162 = v2161;	// L4222
  for (int v2163 = 0; v2163 < v2162; v2163 += 1) {	// L4223
    int32_t v2164 = row20;	// L4224
    ac_int<33, true> v2165 = v2164;	// L4225
    ac_int<33, true> v2166 = v2165 + 1;	// L4226
    int32_t v2167 = v2166;	// L4227
    row20 = v2167;	// L4228
    int32_t v2168 = row20;	// L4229
    int32_t v2169 = instr_rows3;	// L4230
    bool v2170 = v2168 >= v2169;	// L4231
    if (v2170) {	// L4232
      uint64_t v2171 = v2155.read();	// L4233
      uint64_t word4;	// L4234
      word4 = v2171;	// L4235
      uint64_t v2172 = word4;	// L4236
      ac_int<12, false> v2173;
      ap_int<64> v2173_tmp = v2172;
      v2173 = v2173_tmp(29, 18);	// L4237
      int32_t v2174 = v2173;	// L4238
      dram_row01 = v2174;	// L4239
      uint64_t v2175 = word4;	// L4240
      ac_int<12, false> v2176;
      ap_int<64> v2176_tmp = v2175;
      v2176 = v2176_tmp(41, 30);	// L4241
      int32_t v2177 = v2176;	// L4242
      col_block1 = v2177;	// L4243
      uint64_t v2178 = word4;	// L4244
      uint8_t v2179;
      ap_int<64> v2179_tmp = v2178;
      v2179 = v2179_tmp(61, 54);	// L4245
      int32_t v2180 = v2179;	// L4246
      instr_rows3 = v2180;	// L4247
      row20 = 0;	// L4248
    }
    uint32_t v2181 = v2156.read();	// L4250
    uint32_t clipped_word1;	// L4251
    clipped_word1 = v2181;	// L4252
    uint32_t v2182 = clipped_word1;	// L4253
    uint8_t v2183;
    ap_int<32> v2183_tmp = v2182;
    v2183 = v2183_tmp(7, 0);	// L4254
    int8_t lane_value;	// L4255
    lane_value = v2183;	// L4256
    int8_t v2184 = lane_value;	// L4257
    int32_t v2185 = dram_row01;	// L4258
    int32_t v2186 = row20;	// L4259
    ac_int<33, true> v2187 = v2185;	// L4260
    ac_int<33, true> v2188 = v2186;	// L4261
    ac_int<33, true> v2189 = v2187 + v2188;	// L4262
    ac_int<65, true> v2190 = v2189;	// L4263
    ac_int<65, true> v2191 = v2190 * 16;	// L4264
    int32_t v2192 = col_block1;	// L4265
    int64_t v2193 = v2192;	// L4266
    int64_t v2194 = v2193 * 4;	// L4267
    ac_int<66, true> v2195 = v2191;	// L4268
    ac_int<66, true> v2196 = v2194;	// L4269
    ac_int<66, true> v2197 = v2195 + v2196;	// L4270
    int v2198 = v2197;	// L4271
    v2154[v2198] = v2184;	// L4272
    uint32_t v2199 = clipped_word1;	// L4273
    uint8_t v2200;
    ap_int<32> v2200_tmp = v2199;
    v2200 = v2200_tmp(15, 8);	// L4274
    int8_t lane_value1;	// L4275
    lane_value1 = v2200;	// L4276
    int8_t v2201 = lane_value1;	// L4277
    int32_t v2202 = dram_row01;	// L4278
    int32_t v2203 = row20;	// L4279
    ac_int<33, true> v2204 = v2202;	// L4280
    ac_int<33, true> v2205 = v2203;	// L4281
    ac_int<33, true> v2206 = v2204 + v2205;	// L4282
    ac_int<65, true> v2207 = v2206;	// L4283
    ac_int<65, true> v2208 = v2207 * 16;	// L4284
    int32_t v2209 = col_block1;	// L4285
    int64_t v2210 = v2209;	// L4286
    int64_t v2211 = v2210 * 4;	// L4287
    ac_int<66, true> v2212 = v2208;	// L4288
    ac_int<66, true> v2213 = v2211;	// L4289
    ac_int<66, true> v2214 = v2212 + v2213;	// L4290
    ac_int<67, true> v2215 = v2214;	// L4291
    ac_int<67, true> v2216 = v2215 + 1;	// L4292
    int v2217 = v2216;	// L4293
    v2154[v2217] = v2201;	// L4294
    uint32_t v2218 = clipped_word1;	// L4295
    uint8_t v2219;
    ap_int<32> v2219_tmp = v2218;
    v2219 = v2219_tmp(23, 16);	// L4296
    int8_t lane_value2;	// L4297
    lane_value2 = v2219;	// L4298
    int8_t v2220 = lane_value2;	// L4299
    int32_t v2221 = dram_row01;	// L4300
    int32_t v2222 = row20;	// L4301
    ac_int<33, true> v2223 = v2221;	// L4302
    ac_int<33, true> v2224 = v2222;	// L4303
    ac_int<33, true> v2225 = v2223 + v2224;	// L4304
    ac_int<65, true> v2226 = v2225;	// L4305
    ac_int<65, true> v2227 = v2226 * 16;	// L4306
    int32_t v2228 = col_block1;	// L4307
    int64_t v2229 = v2228;	// L4308
    int64_t v2230 = v2229 * 4;	// L4309
    ac_int<66, true> v2231 = v2227;	// L4310
    ac_int<66, true> v2232 = v2230;	// L4311
    ac_int<66, true> v2233 = v2231 + v2232;	// L4312
    ac_int<67, true> v2234 = v2233;	// L4313
    ac_int<67, true> v2235 = v2234 + 2;	// L4314
    int v2236 = v2235;	// L4315
    v2154[v2236] = v2220;	// L4316
    uint32_t v2237 = clipped_word1;	// L4317
    uint8_t v2238;
    ap_int<32> v2238_tmp = v2237;
    v2238 = v2238_tmp(31, 24);	// L4318
    int8_t lane_value3;	// L4319
    lane_value3 = v2238;	// L4320
    int8_t v2239 = lane_value3;	// L4321
    int32_t v2240 = dram_row01;	// L4322
    int32_t v2241 = row20;	// L4323
    ac_int<33, true> v2242 = v2240;	// L4324
    ac_int<33, true> v2243 = v2241;	// L4325
    ac_int<33, true> v2244 = v2242 + v2243;	// L4326
    ac_int<65, true> v2245 = v2244;	// L4327
    ac_int<65, true> v2246 = v2245 * 16;	// L4328
    int32_t v2247 = col_block1;	// L4329
    int64_t v2248 = v2247;	// L4330
    int64_t v2249 = v2248 * 4;	// L4331
    ac_int<66, true> v2250 = v2246;	// L4332
    ac_int<66, true> v2251 = v2249;	// L4333
    ac_int<66, true> v2252 = v2250 + v2251;	// L4334
    ac_int<67, true> v2253 = v2252;	// L4335
    ac_int<67, true> v2254 = v2253 + 3;	// L4336
    int v2255 = v2254;	// L4337
    v2154[v2255] = v2239;	// L4338
  }
}

/// This is top function.
#pragma hls_design top
void tinytpu_isa(
  uint64_t v2256[56],
  int8_t v2257[256],
  int8_t v2258[256],
  int8_t v2259[256]
) {	// L4342
  #pragma hls_design dataflow
  static ac_channel< uint64_t > v2260;
	// L4343
  static ac_channel< uint64_t > v2261;
	// L4344
  static ac_channel< uint64_t > v2262;
	// L4345
  static ac_channel< uint64_t > v2263;
	// L4346
  static ac_channel< uint64_t > v2264;
	// L4347
  static ac_channel< uint32_t > v2265;
	// L4348
  static ac_channel< uint32_t > v2266;
	// L4349
  static ac_channel< uint32_t > v2267;
	// L4350
  static ac_channel< uint32_t > v2268;
	// L4351
  static ac_channel< uint32_t > v2269;
	// L4352
  static ac_channel< uint32_t > v2270;
	// L4353
  static ac_channel< uint32_t > v2271;
	// L4354
  static ac_channel< uint32_t > v2272;
	// L4355
  static ac_channel< uint32_t > v2273;
	// L4356
  static ac_channel< uint32_t > v2274;
	// L4357
  static ac_channel< uint32_t > v2275;
	// L4358
  static ac_channel< uint32_t > v2276;
	// L4359
  static ac_channel< uint32_t > v2277;
	// L4360
  static ac_channel< uint32_t > v2278;
	// L4361
  static ac_channel< uint32_t > v2279;
	// L4362
  static ac_channel< uint32_t > v2280;
	// L4363
  static ac_channel< uint32_t > v2281;
	// L4364
  static ac_channel< uint32_t > v2282;
	// L4365
  static ac_channel< uint32_t > v2283;
	// L4366
  static ac_channel< uint32_t > v2284;
	// L4367
  static ac_channel< uint32_t > v2285;
	// L4368
  static ac_channel< uint32_t > v2286;
	// L4369
  static ac_channel< uint32_t > v2287;
	// L4370
  static ac_channel< uint32_t > v2288;
	// L4371
  static ac_channel< uint32_t > v2289;
	// L4372
  static ac_channel< uint32_t > v2290;
	// L4373
  static ac_channel< uint32_t > v2291;
	// L4374
  static ac_channel< uint32_t > v2292;
	// L4375
  static ac_channel< int8_t > v2293;
	// L4376
  static ac_channel< int8_t > v2294;
	// L4377
  static ac_channel< int8_t > v2295;
	// L4378
  static ac_channel< int8_t > v2296;
	// L4379
  static ac_channel< int8_t > v2297;
	// L4380
  static ac_channel< int8_t > v2298;
	// L4381
  static ac_channel< int8_t > v2299;
	// L4382
  static ac_channel< int8_t > v2300;
	// L4383
  static ac_channel< int8_t > v2301;
	// L4384
  static ac_channel< int8_t > v2302;
	// L4385
  static ac_channel< int8_t > v2303;
	// L4386
  static ac_channel< int8_t > v2304;
	// L4387
  static ac_channel< int8_t > v2305;
	// L4388
  static ac_channel< int8_t > v2306;
	// L4389
  static ac_channel< int8_t > v2307;
	// L4390
  static ac_channel< int8_t > v2308;
	// L4391
  static ac_channel< int32_t > v2309;
	// L4392
  static ac_channel< int32_t > v2310;
	// L4393
  static ac_channel< int32_t > v2311;
	// L4394
  static ac_channel< int32_t > v2312;
	// L4395
  static ac_channel< int32_t > v2313;
	// L4396
  static ac_channel< int32_t > v2314;
	// L4397
  static ac_channel< int32_t > v2315;
	// L4398
  static ac_channel< int32_t > v2316;
	// L4399
  static ac_channel< int32_t > v2317;
	// L4400
  static ac_channel< int32_t > v2318;
	// L4401
  static ac_channel< int32_t > v2319;
	// L4402
  static ac_channel< int32_t > v2320;
	// L4403
  static ac_channel< int32_t > v2321;
	// L4404
  static ac_channel< int32_t > v2322;
	// L4405
  static ac_channel< int32_t > v2323;
	// L4406
  static ac_channel< int32_t > v2324;
	// L4407
  static ac_channel< ac_int<128, false> > v2325;
	// L4408
  static ac_channel< ac_int<128, false> > v2326;
	// L4409
  static ac_channel< ac_int<128, false> > v2327;
	// L4410
  static ac_channel< ac_int<128, false> > v2328;
	// L4411
  static ac_channel< uint32_t > v2329;
	// L4412
  static ac_channel< uint32_t > v2330;
	// L4413
  static ac_channel< uint32_t > v2331;
	// L4414
  static ac_channel< uint32_t > v2332;
	// L4415
  static ac_channel< uint32_t > v2333;
	// L4416
  static ac_channel< uint32_t > v2334;
	// L4417
  static ac_channel< uint32_t > v2335;
	// L4418
  static ac_channel< uint32_t > v2336;
	// L4419
  static ac_channel< uint32_t > v2337;
	// L4420
  static ac_channel< uint32_t > v2338;
	// L4421
  static ac_channel< uint32_t > v2339;
	// L4422
  static ac_channel< uint32_t > v2340;
	// L4423
  static ac_channel< uint32_t > v2341;
	// L4424
  static ac_channel< uint32_t > v2342;
	// L4425
  static ac_channel< uint32_t > v2343;
	// L4426
  static ac_channel< uint32_t > v2344;
	// L4427
  sequencer_0(v2256, v2260, v2261, v2262, v2263, v2264);	// L4428
  dma_ld_0(v2257, v2258, v2260, v2266, v2265);	// L4429
  spm_0(v2261, v2269, v2265, v2267);	// L4430
  vru_0(v2262, v2289, v2267, v2266);	// L4431
  wld_0_0(v2269, v2270, v2273, v2329);	// L4432
  wld_0_1(v2273, v2274, v2330);	// L4433
  wld_0_2(v2274, v2275, v2331);	// L4434
  wld_0_3(v2275, v2332);	// L4435
  wld_1_0(v2270, v2271, v2277, v2333);	// L4436
  wld_1_1(v2277, v2278, v2334);	// L4437
  wld_1_2(v2278, v2279, v2335);	// L4438
  wld_1_3(v2279, v2336);	// L4439
  wld_2_0(v2271, v2272, v2281, v2337);	// L4440
  wld_2_1(v2281, v2282, v2338);	// L4441
  wld_2_2(v2282, v2283, v2339);	// L4442
  wld_2_3(v2283, v2340);	// L4443
  wld_3_0(v2272, v2285, v2341);	// L4444
  wld_3_1(v2285, v2286, v2342);	// L4445
  wld_3_2(v2286, v2287, v2343);	// L4446
  wld_3_3(v2287, v2344);	// L4447
  pe_0_0(v2329, v2289, v2290, v2309, v2293);	// L4448
  pe_0_1(v2330, v2293, v2310, v2294);	// L4449
  pe_0_2(v2331, v2294, v2311, v2295);	// L4450
  pe_0_3(v2332, v2295, v2312);	// L4451
  pe_1_0(v2333, v2290, v2291, v2309, v2313, v2297);	// L4452
  pe_1_1(v2334, v2297, v2310, v2314, v2298);	// L4453
  pe_1_2(v2335, v2298, v2311, v2315, v2299);	// L4454
  pe_1_3(v2336, v2299, v2312, v2316);	// L4455
  pe_2_0(v2337, v2291, v2292, v2313, v2317, v2301);	// L4456
  pe_2_1(v2338, v2301, v2314, v2318, v2302);	// L4457
  pe_2_2(v2339, v2302, v2315, v2319, v2303);	// L4458
  pe_2_3(v2340, v2303, v2316, v2320);	// L4459
  pe_3_0(v2341, v2292, v2317, v2325, v2305);	// L4460
  pe_3_1(v2342, v2305, v2318, v2325, v2326, v2306);	// L4461
  pe_3_2(v2343, v2306, v2319, v2326, v2327, v2307);	// L4462
  pe_3_3(v2344, v2307, v2320, v2327, v2328);	// L4463
  accu_0(v2263, v2328, v2268);	// L4464
  dma_st_0(v2259, v2264, v2268);	// L4465
}

