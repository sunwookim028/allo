# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A copy of the harness glue, for the instance. ``microarch_isa.py`` here
exports the names the frozen gates import from
``examples.tinytpu.microarch_isa``; ``run_gates.py`` mounts this directory
in front of ``examples.tinytpu`` (as ``mutate.py`` mounts a mutant tree) so
the gates run unedited on the build ``TPU_INSTANCE`` selects."""
