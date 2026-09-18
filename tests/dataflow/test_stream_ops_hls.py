# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

def test_vhls_stream_nb():
    @df.region()
    def top():
        S0: Stream[int32, 2][1]
        S1: Stream[int32, 2][1]

        @df.kernel(mapping=[1])
        def kernel():
            # try_get
            data, success = S0[0].try_get()
            if success:
                # try_put
                ok = S1[0].try_put(data)
                if ok:
                    pass
            # empty/full
            if not S0[0].empty() and not S1[0].full():
                pass

    mod = allo.customize(top)
    hls_mod = mod.build(target="vhls")
    hls_code = hls_mod.hls_code
    
    assert ".read_nb(" in hls_code
    assert ".write_nb(" in hls_code
    assert ".empty()" in hls_code
    assert ".full()" in hls_code
    print("Level 2 VHLS Stream NB Test Passed!")

# The TAPA backend has no non-blocking stream codegen. EmitTapaHLS.cpp's visitor
# dispatches only StreamConstructOp/StreamGetOp/StreamPutOp, and the base hooks
# emitStreamTryGet/emitStreamTryPut/emitStreamEmpty/emitStreamFull in
# mlir/include/allo/Translation/EmitBaseHLS.h are empty bodies. So StreamTryGetOp
# falls through to visitUnhandledOp, TapaModuleEmitter::emitBlock reports
# "can't be correctly emitted", and mod.build(target="tapa") raises RuntimeError
# before ever producing code to assert on. `.try_read(`/`.try_write(` -- TAPA's
# real non-blocking API, and what this test asserts -- appear nowhere in the repo.
#
# strict=True on purpose: if someone implements the codegen, this XPASSes and
# fails the suite, which is the prompt to drop the marker. raises=RuntimeError on
# purpose too: if the build starts succeeding but does not emit try_read/try_write,
# the AssertionError is not the expected exception and the test fails loudly
# rather than being quietly excused.
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="TAPA backend has no non-blocking stream codegen: EmitTapaHLS.cpp "
    "dispatches only construct/get/put and the emitStreamTryGet/emitStreamTryPut "
    "hooks in EmitBaseHLS.h are empty, so target='tapa' fails to emit try_get/try_put",
)
def test_tapa_stream_nb():
    @df.region()
    def top():
        S0: Stream[int32, 2][1]

        @df.kernel(mapping=[1])
        def kernel():
            data, success = S0[0].try_get()
            if success:
                S0[0].try_put(data)

    mod = allo.customize(top)
    hls_mod = mod.build(target="tapa")
    hls_code = hls_mod.hls_code

    assert ".try_read(" in hls_code, "Expected .try_read() for try_get"
    assert ".try_write(" in hls_code, "Expected .try_write() for try_put"
    print("Level 2 Tapa Stream NB Test Passed!")

if __name__ == "__main__":
    test_vhls_stream_nb()
    # Expected to fail; see the xfail marker above.
    try:
        test_tapa_stream_nb()
        print("Level 2 Tapa Stream NB Test UNEXPECTEDLY PASSED -- drop the xfail")
    except RuntimeError as err:
        print(f"Level 2 Tapa Stream NB Test xfailed as expected: {err}")
