# Commercial macro PNR

This Stage 1 node launches one isolated Innovus process for every synthesized
canonical macro class. The existing single-node Innovus implementation is
reused inside each worker, but all work directories and reports remain separate.
The complete selected batch is launched concurrently.

Signal pins are placed from the planner's whole-region Allo channel graph rather
than split arbitrarily between two edges. Each semantic stream's complete Vitis
handshake bundle stays together on its requested N/S/E/W side; controls and
auxiliary interfaces follow explicit planner policies. Unknown, duplicate, or
unassigned RTL ports are fatal.

The macro power plan is a sparse edge mesh, not a core ring. One VDD/VSS stripe
pair spans each of two orthogonal upper-metal layers and is exported as LEF PG
geometry. `sroute` connects standard-cell rails to this mesh. This retains the
internal distribution required to power placed cells while avoiding a private,
wide ring and its four large margins around every tiled PE.

After all workers stop, every worker `*.log` is replayed to stdout in stable
class/path order, with begin/end delimiters and from a `finally` path. Thus the
node's `mflowgen-run.log` is complete on success and failure without removing
the individual logs.
