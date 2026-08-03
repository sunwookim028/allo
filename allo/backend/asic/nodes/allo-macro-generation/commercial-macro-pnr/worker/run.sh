#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs outputs
if [[ "${enable_gui:-False}" == "True" ]]; then
  gui_flag=""
else
  gui_flag="-nowin"
fi
innovus -overwrite -64 $gui_flag -init START.tcl -log logs/run.log

cd outputs
[[ ! -e ../checkpoints/design.checkpoint ]] || ln -sf ../checkpoints/design.checkpoint design.checkpoint
compgen -G "../results/*.gds.gz" >/dev/null || exit 1
ln -sf ../results/*.gds.gz design.gds.gz
compgen -G "../results/*-merged.gds" >/dev/null || exit 1
ln -sf ../results/*-merged.gds design-merged.gds
compgen -G "../results/*.vcs.v" >/dev/null || exit 1
ln -sf ../results/*.vcs.v design.vcs.v
compgen -G "../results/*.lef" >/dev/null || exit 1
ln -sf ../results/*.lef design.lef
compgen -G "../results/*.pt.sdc" >/dev/null || exit 1
ln -sf ../results/*.pt.sdc design.pt.sdc
[[ ! -e ../typical.spef.gz ]] || ln -sf ../typical.spef.gz design.spef.gz
if compgen -G "../results/*.sdf" >/dev/null; then ln -sf ../results/*.sdf design.sdf; fi
compgen -G "../results/*.lvs.v" >/dev/null || exit 1
ln -sf ../results/*.lvs.v design.lvs.v
if compgen -G "../results/*.def.gz" >/dev/null; then ln -sf ../results/*.def.gz design.def.gz; fi
