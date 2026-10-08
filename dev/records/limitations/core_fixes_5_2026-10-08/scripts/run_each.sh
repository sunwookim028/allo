#!/bin/bash
# one pytest process per test id, 120 s each, against the tree on $PYTHONPATH
cd $(dirname $0); : > results.txt
while read id; do
  out=$(timeout 120 $CONDA_PREFIX/bin/python -m pytest -p no:cacheprovider -q "$id" 2>&1); rc=$?
  case $rc in 0) v=pass;; 1) v=FAIL;; 124) v=TIMEOUT;; *) v="CRASH(rc=$rc)";; esac
  echo "$v $id" >> results.txt
done < ids.txt
