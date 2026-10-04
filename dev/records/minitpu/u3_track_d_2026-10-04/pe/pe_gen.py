"""U3 ``mxu_pe`` (plan P1 ``bits``) for the open-HLS frontends, generated from
one body: ``examples/minitpu/units/mxu_pe.py::bits`` re-expressed on 32-bit
signed locals with masks (the bit-slice form is refused by both tools: U1
findings A2/A8 in AMC, no bit types in RTLGen), the 19-bit leading-zero count
as a 5-step priority search (U1 F4/A7: a loop with a flag is time in RTLGen
and mis-wired in AMC), 4-way booleans nested in pairs (AMC A5). The PE's and
the adder's registers are loop-carried scalars; outputs are the registers
after the edge, as the Allo ``bits`` writes them.

    python pe_gen.py rtlgen|amc <N> <out.py>
"""
import sys
tool, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = tool == "rtlgen"
I, U1, U8, U16, U32 = ("i32", "u8", "u8", "u16", "u32") if R else ("int32", "uint8", "uint8", "uint16", "uint32")
L = []
A = L.append
if R:
    A("from allo import kernel"); A("from allo.lang import i32, u8, u16, u32"); A(f"N = {N}"); A("")
    A("@kernel")
else:
    A("from allo.ir.types import int32, uint8, uint16, uint32"); A(f"N = {N}"); A("")
A(f"def pe(rst: {U1}[N], cmt: {U1}[N], cbk: {U1}[N], lhs: {U16}[N], lhv: {U1}[N], wgt: {U32}[N], wgv: {U8}[N], "
  f"psi: {U32}[N], psv: {U1}[N], cmo: {U1}[N], lho: {U16}[N], lvo: {U1}[N], wgo: {U32}[N], pso: {U32}[N], pvo: {U1}[N]):")
regs = ["active", "pend0", "pend1", "lhs_q", "lhs_valid_q", "commit_q", "product_q", "psum_q", "product_valid_q",
        "s1w", "s1v", "s2w", "s2v", "result", "vout"]
for r in regs:
    # RTLGen: loop-carried scalars. AMC: a carried scalar is a memory the
    # frontend never promotes; several accesses per iteration overflow its
    # port binding ("more concurrent accesses than the group's count"), so
    # each register is a 1-element array read once at the top and written
    # once at the bottom of the iteration (the finding's workaround).
    A(f"    {r}: {I} = 0" if R else f"    {r}_r: {I}[1] = 0")
A('    for t in range(N, name="t"):' if R else "    for t in range(N):")
B = []  # body lines (8-space indent added)
b = B.append
if not R:
    for r in regs:
        b(f"{r}: {I} = {r}_r[0]")
if R:
    b("ONE: i32 = 1"); b("ZERO: i32 = 0")
    one, zero = "ONE", "ZERO"
else:
    one, zero = "1", "0"
b(f"r: {I} = rst[t]"); b(f"cm: {I} = cmt[t]"); b(f"cb: {I} = cbk[t]"); b(f"x: {I} = lhs[t]"); b(f"xv: {I} = lhv[t]")
b(f"wg: {I} = wgt[t]"); b(f"wv: {I} = wgv[t]"); b(f"ps: {I} = psi[t]"); b(f"pv: {I} = psv[t]")
# ---- stage 3: round and pack (from s2w) ----
b(f"s2_mag: {I} = s2w & 0xFFFFF"); b(f"s2_exp: {I} = (s2w >> 20) & 0x1FF"); b(f"s2_sign: {I} = (s2w >> 29) & 1"); b(f"s2_special: {I} = (s2w >> 30) & 3")
b(f"guard_bit: {I} = (s2_mag >> 2) & 1"); b(f"round_bit: {I} = (s2_mag >> 1) & 1"); b(f"sticky_bit: {I} = s2_mag & 1")
b(f"round_up: {I} = guard_bit & (round_bit | sticky_bit | ((s2_mag >> 3) & 1))")
b(f"frac16: {I} = (s2_mag >> 3) & 0x7FFF"); b(f"rounded: {I} = frac16 + round_up"); b(f"inc_exp: {I} = s2_exp + 1")
b(f"s2_hid: {I} = (s2_mag >> 18) & 1")
b(f"packed: {I} = 0")
b("if s2_special == 3:"); b("    packed = 0x7FC000")
b("elif s2_special == 2:"); b("    packed = (s2_sign << 23) | (0xFF << 15)")
b("elif s2_special == 1:"); b("    if s2_hid == 1:"); b("        packed = (s2_sign << 23) | ((s2_exp & 0xFF) << 15) | ((s2_mag >> 3) & 0x7FFF)")
b("    else:"); b("        packed = (s2_sign << 23) | ((s2_mag >> 3) & 0x7FFF)")
b("elif s2_mag == 0:"); b("    packed = 0")
b("elif ((rounded >> 15) & 1) == 1:"); b("    if inc_exp >= 255:"); b("        packed = (s2_sign << 23) | (0xFF << 15)")
b("    else:"); b("        packed = (s2_sign << 23) | ((inc_exp & 0xFF) << 15)")
b("elif s2_exp >= 255:"); b("    packed = (s2_sign << 23) | (0xFF << 15)")
b("elif s2_exp <= 1 and s2_hid == 0:"); b("    packed = (s2_sign << 23) | (rounded & 0x7FFF)")
b("else:"); b("    packed = (s2_sign << 23) | ((s2_exp & 0xFF) << 15) | (rounded & 0x7FFF)")
# ---- stage 2: normalize (from s1w) ----
b(f"s1_mag: {I} = s1w & 0xFFFFF"); b(f"s1_exp: {I} = (s1w >> 20) & 0x1FF"); b(f"s1_sign: {I} = (s1w >> 29) & 1"); b(f"s1_special: {I} = (s1w >> 30) & 3")
b(f"norm_overflow: {I} = (s1_mag >> 19) & 1"); b(f"s1_hid: {I} = (s1_mag >> 18) & 1")
b(f"norm_needed: {I} = 0")
b("if s1_mag != 0 and (s1_hid == 0 and norm_overflow == 0):"); b("    norm_needed = 1")
b(f"mag19: {I} = s1_mag & 0x7FFFF")
b(f"lx: {I} = mag19"); b(f"lzc: {I} = 0")
b("if lx == 0:"); b("    lzc = 19")
b("else:")
b("    if lx < 8:"); b("        lzc = lzc + 16"); b("        lx = lx << 16")
b("    if lx < 2048:"); b("        lzc = lzc + 8"); b("        lx = lx << 8")
b("    if lx < 32768:"); b("        lzc = lzc + 4"); b("        lx = lx << 4")
b("    if lx < 131072:"); b("        lzc = lzc + 2"); b("        lx = lx << 2")
b("    if lx < 262144:"); b("        lzc = lzc + 1")
b(f"max_ns: {I} = 18 if s1_exp > 19 else s1_exp - 1")
b(f"normalize_shift: {I} = 0")
b("if norm_needed == 1:"); b("    normalize_shift = lzc if lzc < max_ns else max_ns")
b(f"mag_normalized: {I} = (mag19 << normalize_shift) & 0x7FFFF")
b(f"mag_s2: {I} = s1_mag"); b(f"exp_s2: {I} = s1_exp")
b("if s1_special == 1:"); b("    exp_s2 = s1_exp")
b("elif norm_overflow == 1:"); b("    mag_s2 = (mag_s2 | ((mag_s2 & 1) << 1)) >> 1"); b("    exp_s2 = s1_exp + 1")
b("elif norm_needed == 1:"); b("    mag_s2 = mag_normalized"); b("    exp_s2 = s1_exp - normalize_shift")
b(f"w2n: {I} = (mag_s2 & 0xFFFFF) | ((exp_s2 & 0x1FF) << 20) | (s1_sign << 29) | (s1_special << 30)")
# ---- stage 1: classify, align (jam), add (product_q + psum_q) ----
b(f"a_i: {I} = product_q"); b(f"b_i: {I} = psum_q")
b(f"sign_a: {I} = (a_i >> 23) & 1"); b(f"sign_b: {I} = (b_i >> 23) & 1")
b(f"exp_a: {I} = (a_i >> 15) & 0xFF"); b(f"exp_b: {I} = (b_i >> 15) & 0xFF")
b(f"frac_a: {I} = a_i & 0x7FFF"); b(f"frac_b: {I} = b_i & 0x7FFF")
b(f"hid_a: {I} = {one} if exp_a != 0 else {zero}"); b(f"hid_b: {I} = {one} if exp_b != 0 else {zero}")
b(f"sig_a: {I} = (hid_a << 15) | frac_a"); b(f"sig_b: {I} = (hid_b << 15) | frac_b")
b(f"same_sign: {I} = {one} if sign_a == sign_b else {zero}")
b(f"key_a: {I} = a_i & 0x7FFFFF"); b(f"key_b: {I} = b_i & 0x7FFFFF")
b(f"a_is_large: {I} = {one} if key_a >= key_b else {zero}")
b(f"sign_large: {I} = sign_b"); b(f"exp_large: {I} = 0"); b(f"exp_small: {I} = 0"); b(f"sig_large: {I} = 0"); b(f"sig_small: {I} = 0")
b("if a_is_large == 1:"); b("    sign_large = sign_a"); b("    exp_large = 1 if exp_a == 0 else exp_a"); b("    exp_small = 1 if exp_b == 0 else exp_b")
b("    sig_large = sig_a"); b("    sig_small = sig_b")
b("else:"); b("    exp_large = 1 if exp_b == 0 else exp_b"); b("    exp_small = 1 if exp_a == 0 else exp_a"); b("    sig_large = sig_b"); b("    sig_small = sig_a")
b(f"exp_diff: {I} = exp_large - exp_small")
b(f"align_shift: {I} = 19 if exp_diff >= 19 else exp_diff")
b(f"wide: {I} = sig_small << 3"); b(f"amask: {I} = (1 << align_shift) - 1")
b(f"jam: {I} = {one} if (wide & amask) != 0 else {zero}")
b(f"small_aligned: {I} = (wide >> align_shift) | jam")
b(f"mant_large: {I} = sig_large << 3")
b(f"magnitude_s1: {I} = 0")
b("if same_sign == 1:"); b("    magnitude_s1 = (mant_large + small_aligned) & 0xFFFFF")
b("else:"); b("    magnitude_s1 = (mant_large - small_aligned) & 0xFFFFF")
b(f"special_s1: {I} = 0")
b(f"a_nan: {I} = {one} if (exp_a == 0xFF and frac_a != 0) else {zero}"); b(f"b_nan: {I} = {one} if (exp_b == 0xFF and frac_b != 0) else {zero}")
b(f"inf_cancel: {I} = {one} if ((exp_a == 0xFF and exp_b == 0xFF) and sign_a != sign_b) else {zero}")
b("if (a_nan == 1 or b_nan == 1) or inf_cancel == 1:"); b("    special_s1 = 3")
b("elif exp_a == 0xFF or exp_b == 0xFF:"); b("    special_s1 = 2")
b("elif key_a == 0:"); b("    special_s1 = 1"); b("    sign_large = sign_b")
b("elif key_b == 0:"); b("    special_s1 = 1"); b("    sign_large = sign_a")
b(f"w1n: {I} = (magnitude_s1 & 0xFFFFF) | ((exp_large & 0x1FF) << 20) | (sign_large << 29) | (special_s1 << 30)")
# ---- the multiplier (mxu_bf16_mul_acc24.sv): lhs_i x active ----
b(f"ma: {I} = x & 0xFFFF"); b(f"mb: {I} = active & 0xFFFF")
b(f"msign: {I} = ((ma >> 15) ^ (mb >> 15)) & 1")
b(f"mexp_a: {I} = (ma >> 7) & 0xFF"); b(f"mexp_b: {I} = (mb >> 7) & 0xFF")
b(f"mfrac_a: {I} = ma & 0x7F"); b(f"mfrac_b: {I} = mb & 0x7F")
b(f"mant_a: {I} = 0x80 | mfrac_a"); b(f"mant_b: {I} = 0x80 | mfrac_b")
b(f"mprod: {I} = mant_a * mant_b")
b(f"exp_sum: {I} = mexp_a + mexp_b"); b(f"exp_low: {I} = exp_sum - 127"); b(f"exp_high: {I} = exp_sum - 126")
b(f"finite_low: {I} = (msign << 23) | ((exp_low & 0xFF) << 15) | ((mprod & 0x3FFF) << 1)")
b("if exp_sum <= 127:"); b("    finite_low = msign << 23")
b("elif exp_sum >= 382:"); b("    finite_low = (msign << 23) | (0xFF << 15)")
b(f"finite_high: {I} = (msign << 23) | ((exp_high & 0xFF) << 15) | (mprod & 0x7FFF)")
b("if exp_sum <= 126:"); b("    finite_high = msign << 23")
b("elif exp_sum >= 381:"); b("    finite_high = (msign << 23) | (0xFF << 15)")
b(f"finite_result: {I} = finite_high if ((mprod >> 15) & 1) == 1 else finite_low")
b(f"ma_nan: {I} = {one} if (mexp_a == 0xFF and mfrac_a != 0) else {zero}"); b(f"mb_nan: {I} = {one} if (mexp_b == 0xFF and mfrac_b != 0) else {zero}")
b(f"inf_zero: {I} = {one} if ((mexp_a == 0xFF and (mb & 0x7FFF) == 0) or (mexp_b == 0xFF and (ma & 0x7FFF) == 0)) else {zero}")
b(f"product: {I} = 0")
b("if (ma_nan == 1 or mb_nan == 1) or inf_zero == 1:"); b("    product = 0x7FC000")
b("elif mexp_a == 0xFF or mexp_b == 0xFF:"); b("    product = (msign << 23) | (0xFF << 15)")
b("elif mexp_a == 0 or mexp_b == 0:"); b("    product = msign << 23")
b("else:"); b("    product = finite_result")
# ---- the edge: adder stages (reset forces class and valid) ----
b("if r == 0:"); b("    result = 0"); b("    vout = 0"); b("    s2w = w2n & 0x3FFFFFFF"); b("    s2v = 0"); b("    s1w = w1n & 0x3FFFFFFF"); b("    s1v = 0")
b("else:"); b("    result = packed"); b("    vout = s2v"); b("    s2w = w2n"); b("    s2v = s1v"); b("    s1w = w1n"); b("    s1v = product_valid_q")
# ---- the edge: the PE's registers ----
b("product_q = product"); b("psum_q = ps & 0xFFFFFF"); b("lhs_q = x")
b("if r == 0:"); b("    active = 0"); b("    pend0 = 0"); b("    pend1 = 0"); b("    product_valid_q = 0"); b("    lhs_valid_q = 0"); b("    commit_q = 0")
b("else:"); b("    lhs_valid_q = xv")
b("    if cm == 1:"); b("        active = pend1 if cb == 1 else pend0")
b("    if (wv & 1) == 1:"); b("        pend0 = wg & 0xFFFF")
b("    if ((wv >> 1) & 1) == 1:"); b("        pend1 = (wg >> 16) & 0xFFFF")
b("    commit_q = cm"); b("    product_valid_q = xv & pv")
# ---- outputs: the registers after the edge ----
b("cmo[t] = commit_q"); b("lho[t] = lhs_q"); b("lvo[t] = lhs_valid_q")
b(f"wo: {I} = pend0 | (pend1 << 16)"); b("wgo[t] = wo"); b("pso[t] = result"); b("pvo[t] = vout")
if not R:
    for r in regs:
        b(f"{r}_r[0] = {r}")
L += ["        " + ln for ln in B]
open(out, "w").write("\n".join(L) + "\n")
print(f"GENERATED {out}: {len(L)} lines ({tool}, N={N})")
