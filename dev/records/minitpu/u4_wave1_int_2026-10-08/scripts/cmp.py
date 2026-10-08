import re, glob, os, sys
REC = "/work/shared/users/phd/sk3463/scratch/wt-u4int/dev/records/minitpu/"
NEW = "/work/shared/users/phd/sk3463/scratch/u4int_out/logs/"  # the run's output; copied to ../logs/
pairs = []
for f in sorted(glob.glob(NEW + "ac/A_*.log")):
    pairs.append((f, REC + "u4_track_a_2026-10-08/" + os.path.basename(f)[2:]))
for f in sorted(glob.glob(NEW + "ac/C_*.log")):
    b = os.path.basename(f)[2:]
    pairs.append((f, REC + "u4_track_c_2026-10-08/" + b))
for u in ("seq_issue", "vpu_cmd", "vpu_wb"):
    pairs.append((NEW + f"check_{u}.log", REC + f"u4_track_b_2026-10-08/logs/check_{u}.log"))
PAT = re.compile(r"^(UNIT-|CONTRACT-|DERIVED|REFUSED|ACCEPTED|\[item|CALENDAR|DELTA|GEOM|.*-MATCH|.*-DIFF)")
def norm(l):
    l = re.sub(r"\((build [^)]*|[0-9.]+s)\)", "", l)
    l = re.sub(r"\b[0-9.]+s\b", "", l)
    l = re.sub(r"/[^ ]*/(wt-[a-z0-9]+|scratch)[^ ]*", "<path>", l)
    return " ".join(l.split())
def lines(f):
    if not os.path.exists(f): return None
    return [norm(l) for l in open(f, errors="replace") if PAT.match(l) and "no verdict, killed" not in l]
tot = same = 0
for new, old in pairs:
    a, b = lines(new), lines(old)
    if b is None:
        print(f"NO-RECORD-LOG {os.path.basename(new)} ({len(a)} lines)"); continue
    sa, sb = sorted(a), sorted(b)
    tot += 1
    if sa == sb:
        same += 1; print(f"EQUAL {os.path.basename(new)}: {len(a)} verdict lines")
    else:
        print(f"CHANGED {os.path.basename(new)}:")
        for x in sorted(set(sa) - set(sb)): print("   + " + x[:230])
        for x in sorted(set(sb) - set(sa)): print("   - " + x[:230])
print(f"{same}/{tot} logs equal")
