# Commercial batch physical verification

This node runs Calibre DRC and LVS for every hardened macro. Macro classes run
concurrently, and DRC/LVS also run concurrently within each isolated class work
directory. Only verified PNR entries are forwarded to timing/model signoff.

After all workers stop, every DRC/LVS worker `*.log` is replayed to stdout in
stable class/path order, with begin/end delimiters and from a `finally` path.
The node's `mflowgen-run.log` therefore remains useful even after a failure.
