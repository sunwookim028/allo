# Full-chip assembly planning

This node validates the normalized RTL, final Allo ASIC manifest, and hardened
macro registry as one consistent dataset. It records every concrete instance
replacement and its member-to-canonical named-port map, while carrying the
whole-region channel graph forward for later placement/orientation planning.
It makes no physical placement choice itself.

For FIFO-folded entries, the plan carries explicit canonical wrapper
connections and the exact producer-owned FIFO module/instance removals. Stable
semantic instance names use `<kernel>_<pid...>` when supplied by the macro
plan. Derived FIFO-wrapper classes retain their source compiler equivalence
class for manifest validation.
