#!/bin/bash
# usage: hashes.sh <worktree> <outdir> : TinyTPU emission hashes (sync5 gates.sh's `hashes` stage)
WT=$1 L=$2; mkdir -p $L
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate allo >/dev/null 2>&1
export OMP_NUM_THREADS=8 PYTHONPATH=$WT; cd $WT
$CONDA_PREFIX/bin/python -c "
import hashlib
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
for T in ('vhls','catapult','systemc'):
    s=customize(tinytpu_isa); schedule(s); t=str(s.build(target=T))
    open('$L/emit_'+T+'.txt','w').write(t)
    print('HASH',T,hashlib.sha256(t.encode()).hexdigest(),len(t))
" 2>&1 | grep HASH
