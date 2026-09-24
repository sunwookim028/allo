# Commercial batch synthesis

This node consumes the indexed `macro-batch` contract and launches one isolated
copy of the existing Synopsys Design Compiler flow per canonical macro class.
All entries run concurrently. Each worker has separate inputs, logs, reports,
results, status, and output views; the ADK and optional SRAM libraries are shared
read-only.

The node deliberately uses separate DC processes. A single process would
amortize library setup but serialize the designs and make failure recovery and
state isolation harder. Shared `.alib` data already avoids much of the repeated
library-analysis cost.

Required output links are created only after resolving a real DC artifact.
Optional name-map and UPF files are omitted when DC does not emit them, avoiding
dangling wildcard symlinks. The collector resolves every valid worker symlink
relative to its real filesystem location and copies the target data into a
self-contained batch, while skipping only links that are genuinely dangling.

After the parallel workers finish, every worker `*.log` is replayed to the node's
stdout in deterministic class/path order with explicit begin/end delimiters.
mflowgen therefore captures the complete batch transcript in `mflowgen-run.log`
on both success and failure, while the original per-worker log files remain
available for focused inspection.
