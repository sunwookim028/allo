# Commercial batch physical verification

This node runs Calibre DRC and LVS for every hardened macro. Macro classes run
concurrently, and DRC/LVS also run concurrently within each isolated class work
directory. Only verified PNR entries are forwarded to timing/model signoff.
