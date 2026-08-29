# Allo collateral trimming

This terminal node optionally removes a conservative allowlist of recomputable
scratch after full-chip verification, simulation, activity extraction, and
power analysis complete. It is disabled by default with `trim_collateral`.

The retained archive includes final physical collateral, published macro views,
Innovus checkpoints, timing/area/congestion reports, concise power and activity
reports, simulation logs and result JSON, manifests, metrics, and flow scripts.
The removed set includes full-chip LVS `svdb`, detailed PrimeTime pre-activity
dumps and saved sessions, FFGL/BAGL waveforms and compiled simulator databases,
completed worker scratch, and generated Python/pytest caches.

`trim-metrics.json` records pre-trim and post-trim apparent sizes plus every
removed path. A trimmed build cannot reopen the deleted LVS, PrimeTime, or VCS
sessions without regenerating those nodes.
