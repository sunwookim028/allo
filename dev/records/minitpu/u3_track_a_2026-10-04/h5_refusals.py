"""H5: each wrong parameter set against tree_legality at the composition."""
from examples.minitpu.units import xlu_tree_compose as tc
ok = tc.architecture(4, 4, 8)
print(f"legal 4x4: composed, {len(ok.units)} unit declarations, instances",
      [u.instances for u in ok.units])
ok64 = tc.architecture(16, 4, 8)
print("legal 16x4: composed")
for case, msg in tc.refusals().items():
    tag = "ACCEPTED (BAD)" if msg == "ACCEPTED" else "refused"
    print(f"{tag:14s} {case}\n               -> {msg[:150]}")
