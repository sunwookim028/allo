#!/bin/bash
F=/work/shared/users/phd/sk3463/scratch/fix5; cd $F/f1/pretest
source $(conda info --base)/etc/profile.d/conda.sh; conda activate allo; export OMP_NUM_THREADS=8 PYTHONPATH=$F/pre
$CONDA_PREFIX/bin/python -m pytest -p no:cacheprovider -q --collect-only test_wide_literal.py 2>/dev/null | grep "::" > ids.txt
./run_each.sh
