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
