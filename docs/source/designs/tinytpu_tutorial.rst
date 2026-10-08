..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

##################################################
Tutorial: A PyTorch MLP on TinyTPU, Then Change It
##################################################

The walk-through lives with the example it runs, as a regular example:
`examples/tinytpu/README.md
<https://github.com/sunwookim028/allo/blob/main/examples/tinytpu/README.md>`_.

It takes a two-layer PyTorch MLP through ACT onto the TinyTPU design and runs
it there, then changes the machine twice -- a larger systolic array
(``TPU_T=8``), and a fused ``mvoutrelu`` instruction added to the ISA, the
hardware, the toolchain and the compiler by one patch
(``examples/tinytpu/mvoutrelu.patch``) -- and ends with the gates. Every
command in it is listed with its complete output, as run on this fork's host.
The fused instruction stays off the shipped design (README decision D-4).

The flow and its design choices are :doc:`workload_suite`; the machine is
:doc:`tinytpu_isa`.
