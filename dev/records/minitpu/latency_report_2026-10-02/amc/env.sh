# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Source (bash) on zhang-21. AMC build at scratch/amc (amc-dialect fe60c121).
AMC=/work/shared/users/phd/sk3463/scratch/amc
U1=/work/shared/users/phd/sk3463/scratch/wt-lat/dev/records/minitpu/u1_bf16_add_amc
export PATH=$AMC/env/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH
export PYTHONPATH=$AMC/amc-dialect/allo:$AMC/amc-dialect/build/tools/amc/python_packages/amc_core:$U1:$AMC/latency
# Local XFS disk (not NFS): AMCModule's tempfile.mkdtemp + rmtree breaks on NFS.
mkdir -p /scratch/sk3463/amc_latency_tmp && export TMPDIR=/scratch/sk3463/amc_latency_tmp
