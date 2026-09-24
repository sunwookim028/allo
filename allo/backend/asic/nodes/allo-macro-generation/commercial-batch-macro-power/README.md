# Commercial batch macro power

This node reuses the standard `synopsys-pt-power` scripts for one representative
of every published macro class. It consumes instance-scoped SAIF files produced
from BAGL activity, runs class jobs concurrently up to
`macro_power_max_workers`, weights each result by the registry reuse count, and
adds the weighted macro contribution to the full-chip shell power estimate.

The output directory retains every class's PrimeTime reports. The JSON summary
is the machine-readable result; `aggregate-power.rpt` is the concise report for
inspection. A failed or unparsable class job fails the node rather than silently
omitting its contribution.
