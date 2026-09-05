# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The composed TinyTPU schedule keeps all three accepted backend choices."""

from examples.accelerator.tinytpu.microarch import export_backend


def test_tinytpu_cpu_compiles():
    module = export_backend("cpu").compile()
    assert "llvm.func @tinytpu" in str(module)


def test_tinytpu_vitis_exports_relu_nan_semantics():
    code = export_backend("vitis").hls_code
    assert 'extern "C" void tinytpu' in code
    assert "hls::isnan" in code


def test_tinytpu_rtlgen_exports_verilog():
    rtl = export_backend("rtl")
    assert "module tinytpu" in rtl.verilog
    assert rtl.schedule().func("tinytpu") is not None
