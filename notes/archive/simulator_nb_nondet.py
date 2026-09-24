# =====================================================================================
# ARCHIVED SNAPSHOT of allo/backend/simulator.py, taken BEFORE the DAM-lite timing layer.
#
# WHAT IT IS. 1,490 lines against the current simulator's 2,389. It lacks the whole timing
# layer -- _emit_read_barrier, _emit_deadlock_guard, _emit_cycle_harvest,
# _emit_blocked_delta, _declare_sim_global, _collect_top_loop_bodies.
#
# WHY IT IS KEPT. That missing read barrier is exactly why it is useful: the CURRENT
# simulator cannot run a mesh. Its per-PE clocks diverge (a router body charging ~270
# cycles/pass against a collector's ~2), so `try_get` never sees the producer's timestamps
# and packets sit in the FIFO, correctly written and forever unread -- no deadlock, no
# error, zero delivered. See notes/SIMULATOR.md section 3.2. This snapshot has no barrier,
# so it runs meshes. The price is in the name: non-blocking is NON-DETERMINISTIC here
# (23 distinct outcomes in 30 runs on nb_nondeterminism.py).
#
# HOW TO USE IT. It is NOT importable from here -- the relative imports below
# (`from ..backend.llvm import ...`) only resolve inside the allo package. To use it:
#
#     cp allo/backend/simulator.py /tmp/simulator_timed.py.bak      # keep the real one
#     cp notes/archive/simulator_nb_nondet.py allo/backend/simulator.py
#     ... run the mesh ...
#     cp /tmp/simulator_timed.py.bak allo/backend/simulator.py      # put it back
#
# Results from this simulator are ordering-dependent and must not be quoted as
# deterministic. It is a means of getting a mesh to execute at all, nothing more.
# =====================================================================================

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, super-init-not-called, too-many-nested-blocks, too-many-branches
# pylint: disable=consider-using-enumerate, no-value-for-parameter, too-many-function-args, redefined-variable-type

import os
from ..backend.llvm import LLVMModule
from .._mlir.ir import (
    Location,
    UnitAttr,
    InsertionPoint,
    Module,
    Context,
    Region,
    RegionSequence,
    Block,
    BlockArgument,
    BlockArgumentList,
    OpView,
    OpResult,
    OpOperandList,
    Operation,
    Value,
    TypeAttr,
    StringAttr,
    AffineMapAttr,
    AffineMap,
    AffineExpr,
    FunctionType,
    MemRefType,
    IntegerType,
    FloatType,
    IndexType,
    FlatSymbolRefAttr,
)
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    openmp as openmp_d,
    arith as arith_d,
    index as index_d,
    affine as affine_d,
    scf as scf_d,
    llvm as llvm_d,
)
from .._mlir.passmanager import PassManager
from .._mlir.execution_engine import ExecutionEngine
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function
from ..utils import get_func_inputs_outputs


# The `walk` function
def recursive_collect_ops(
    top_op: Operation, target_op_type: tuple[type], res_list: list
):
    if isinstance(top_op, target_op_type):
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops(op, target_op_type, res_list)


# Useful when searching for omp operations after lowering
def recursive_collect_ops_by_name(
    top_op: Operation, target_op_name: str, res_list: list
):
    if top_op.name == target_op_name:
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops_by_name(op, target_op_name, res_list)


# ---------------------------------------------------------------------------
# Phase 1: per-PE simulated-time clock (default "all-ones" cost model).
# ---------------------------------------------------------------------------
def _op_latency(op):
    """Static latency (abstract cycles) of one op for the per-PE clock.
    Default cost model: every op costs 1 tick. This is the single place to
    refine later toward the HLS II / latency schedule."""
    return 1


def _block_latency(block):
    """Sum of _op_latency over ops DIRECTLY in this block (nested regions get
    their own increments, so a loop body is charged once per iteration)."""
    return sum(_op_latency(op) for op in block.operations)


def _collect_blocks(op, res):
    """Collect every Block in op's regions, recursively -- but NOT inside spin-wait
    scf.while loops: their iteration count is scheduler-dependent, so charging the
    clock per spin would make it non-deterministic. The while op still counts once
    in its enclosing block (via _block_latency)."""
    for region in op.regions:
        for block in region.blocks:
            res.append(block)
            for inner in block.operations:
                if inner.name == "scf.while":
                    continue
                _collect_blocks(inner, res)


def _insert_clock_increments(func, clock_arg, module):
    """Insert `clock_arg += static_block_latency` at the start of every block of
    the PE function (recursively), so the PE's clock tracks simulated time under
    the default cost model. Inert until the clock is read (Phase 3)."""
    i64 = IntegerType.get_signless(64, module.context)
    blocks = []
    _collect_blocks(func, blocks)
    for block in blocks:
        lat = _block_latency(block)
        if lat == 0:
            continue
        ip = InsertionPoint(beforeOperation=block.operations[0])
        cur = memref_d.LoadOp(memref=clock_arg, indices=[], ip=ip)
        inc = arith_d.ConstantOp(i64, lat, ip=ip)
        nxt = arith_d.AddIOp(lhs=cur.result, rhs=inc.result, ip=ip)
        memref_d.StoreOp(nxt, clock_arg, [], ip=ip)


def _add_pe_clock_args(module, top_func_name):
    """Reorder step: give each PE a trailing memref<i64> clock arg and thread it
    through the (recreated) calls, BEFORE streams are lowered -- so put/get lowering
    can stamp/advance the clock. Each clocked PE func is tagged 'sim.clock'; its
    clock is the last block arg. Per-block increments come later (after lowering)."""
    i64 = IntegerType.get_signless(64, module.context)
    clk_ty = MemRefType.get([], i64)
    funcs = {
        str(op.sym_name).strip('"'): op
        for op in module.body.operations
        if isinstance(op, func_d.FuncOp) and len(op.body.blocks) > 0
    }
    clocked = set()
    for caller in funcs.values():
        calls = []
        recursive_collect_ops(caller, (func_d.CallOp,), calls)
        entry_ip = InsertionPoint.at_block_begin(caller.body.blocks[0])
        for call_op in calls:
            callee_name = str(call_op.callee)[1:]
            callee = funcs.get(callee_name)
            if callee is None or callee_name.startswith(
                ("load_buf", "store_res", "usleep")
            ):
                continue
            if callee_name not in clocked:
                clocked.add(callee_name)
                callee.body.blocks[0].add_argument(clk_ty, Location.unknown())
                old_ty = callee.type
                new_ty = FunctionType.get(
                    list(old_ty.inputs) + [clk_ty], list(old_ty.results)
                )
                callee.attributes["function_type"] = TypeAttr.get(new_ty)
                callee.attributes["sim.clock"] = UnitAttr.get()
            clk = memref_d.AllocOp(clk_ty, [], [], ip=entry_ip)
            c0 = arith_d.ConstantOp(i64, 0, ip=entry_ip)
            memref_d.StoreOp(c0, clk, [], ip=entry_ip)
            func_d.CallOp(
                [], call_op.callee,
                list(call_op.operands_) + [clk.result],
                ip=InsertionPoint(beforeOperation=call_op),
            )
            call_op.operation.erase()


def _insert_pe_clock_increments(module):
    """Insert per-block clock increments into every clocked PE (tagged sim.clock),
    using its last block arg as the clock. Runs AFTER stream lowering so spin-wait
    scf.while loops are present and get skipped by _collect_blocks."""
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and "sim.clock" in op.attributes:
            clock_arg = op.body.blocks[0].arguments[-1]
            _insert_clock_increments(op, clock_arg, module)


def _clock_of(func):
    """The PE's clock block arg (its last arg) if func is a clocked PE (tagged
    sim.clock), else None."""
    if "sim.clock" in func.attributes:
        return func.body.blocks[0].arguments[-1]
    return None


def _stamp_put_ts(ts_ptr, slot_index, clock_arg, ip):
    """On put: ts_ring[slot] = clock -- stamp the element with the producer's
    current simulated time. No-op when the enclosing func has no clock."""
    if clock_arg is None:
        return
    clk_val = memref_d.LoadOp(memref=clock_arg, indices=[], ip=ip)
    memref_d.StoreOp(clk_val, ts_ptr, [slot_index], ip=ip)


def _advance_get_ts(ts_ptr, slot_index, clock_arg, ip):
    """On get: clock = max(clock, ts_ring[slot]) -- advance the consumer clock to
    when the dequeued element was produced. No-op when there is no clock."""
    if clock_arg is None:
        return
    ts_val = memref_d.LoadOp(memref=ts_ptr, indices=[slot_index], ip=ip)
    clk_val = memref_d.LoadOp(memref=clock_arg, indices=[], ip=ip)
    mx = arith_d.MaxSIOp(lhs=clk_val.result, rhs=ts_val.result, ip=ip)
    memref_d.StoreOp(mx, clock_arg, [], ip=ip)


def _lower_nb_stream_op(
    stream_access_op, head_ptr, tail_ptr, fifo_ptr,
    stream_type, const_one, const_fifo_depth, module, replace_ip,
    ts_ptr, clock_arg,
):
    """Lower a non-blocking / status stream op (empty/full/try_put/try_get)
    to its ring-buffer implementation. Returns True if it handled the op,
    False for a blocking put/get (left to the caller). Shared by the
    cross-call and local lowering paths (Phase 0 refactor)."""
    if isinstance(stream_access_op, allo_d.StreamEmptyOp):
        openmp_d.FlushOp([], ip=replace_ip)
        head_val = memref_d.LoadOp(memref=head_ptr, indices=[], ip=replace_ip)
        tail_val = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=replace_ip)
        cmp_op = arith_d.CmpIOp(0, lhs=head_val, rhs=tail_val, ip=replace_ip)
        stream_access_op.results[0].replace_all_uses_with(cmp_op.result)
        stream_access_op.operation.erase()
        return True
    if isinstance(stream_access_op, allo_d.StreamFullOp):
        openmp_d.FlushOp([], ip=replace_ip)
        tail_val = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=replace_ip)
        tail_inc = arith_d.AddIOp(
            lhs=tail_val.result, rhs=const_one.result, ip=replace_ip
        )
        tail_next = arith_d.RemUIOp(
            lhs=tail_inc.result, rhs=const_fifo_depth.result, ip=replace_ip
        )
        head_val = memref_d.LoadOp(memref=head_ptr, indices=[], ip=replace_ip)
        cmp_op = arith_d.CmpIOp(
            0, lhs=tail_next.result, rhs=head_val.result, ip=replace_ip
        )
        stream_access_op.results[0].replace_all_uses_with(cmp_op.result)
        stream_access_op.operation.erase()
        return True
    if isinstance(stream_access_op, allo_d.StreamTryPutOp):
        openmp_d.FlushOp([], ip=replace_ip)
        tail_val_op = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=replace_ip)
        tail_inc_op = arith_d.AddIOp(
            lhs=tail_val_op.result, rhs=const_one.result, ip=replace_ip
        )
        tail_next_op = arith_d.RemUIOp(
            lhs=tail_inc_op.result, rhs=const_fifo_depth.result, ip=replace_ip
        )
        head_val_op = memref_d.LoadOp(memref=head_ptr, indices=[], ip=replace_ip)
        is_full = arith_d.CmpIOp(
            0, lhs=head_val_op.result, rhs=tail_next_op.result, ip=replace_ip
        )
        is_not_full = arith_d.CmpIOp(
            1, lhs=head_val_op.result, rhs=tail_next_op.result, ip=replace_ip
        )
        if_op = scf_d.IfOp(
            is_not_full.result,
            [IntegerType.get_signless(1, module.context)],
            has_else=True,
            ip=replace_ip,
        )
        # Then block (Not Full)
        then_ip = InsertionPoint(if_op.then_block)
        data = stream_access_op.data
        tail_index_op = index_d.CastUOp(
            output=IndexType.get(module.context), input=tail_val_op, ip=then_ip
        )
        if isinstance(data.type, MemRefType):
            element_type = data.type.element_type
            rank = data.type.rank
            for_ip = then_ip
            for_induction_vars = []
            for_ips = []
            for i in range(rank):
                dim_size = data.type.get_dim_size(i)
                for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                for_induction_vars.append(for_loop_op.induction_variable)
                for_ip = InsertionPoint(for_loop_op.body)
                for_ips.append(for_ip)
            element_dim_map = AffineMap.get(
                dim_count=rank,
                symbol_count=0,
                exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                context=module.context,
            )
            element_load_op = affine_d.AffineLoadOp(
                result=element_type,
                memref=data,
                indices=for_induction_vars,
                map=AffineMapAttr.get(element_dim_map),
                ip=for_ip,
            )
            memref_d.StoreOp(
                value=element_load_op,
                memref=fifo_ptr,
                indices=[tail_index_op] + for_induction_vars,
                ip=for_ip,
            )
            for ip in for_ips:
                affine_d.AffineYieldOp([], ip=ip)
        else:
            fifo_element_type = stream_type.element_type
            store_value = data
            if data.type != fifo_element_type:
                if (
                    isinstance(data.type, (IntegerType, IndexType))
                    and isinstance(fifo_element_type, (IntegerType, IndexType))
                ):
                    if isinstance(data.type, IndexType):
                        store_value = index_d.CastSOp(
                            fifo_element_type, data, ip=then_ip
                        )
                    elif isinstance(fifo_element_type, IndexType):
                        store_value = index_d.CastSOp(
                            IndexType.get(module.context), data, ip=then_ip
                        )
                    elif data.type.width > fifo_element_type.width:
                        store_value = arith_d.TruncIOp(
                            fifo_element_type, data, ip=then_ip
                        )
                    elif data.type.width < fifo_element_type.width:
                        if data.type.is_signed:
                            store_value = arith_d.ExtSIOp(
                                fifo_element_type, data, ip=then_ip
                            )
                        else:
                            store_value = arith_d.ExtUIOp(
                                fifo_element_type, data, ip=then_ip
                            )
            memref_d.StoreOp(
                value=store_value,
                memref=fifo_ptr,
                indices=[tail_index_op],
                ip=then_ip,
            )
        _stamp_put_ts(ts_ptr, tail_index_op, clock_arg, then_ip)
        critical_op = openmp_d.CriticalOp(ip=then_ip)
        critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
        memref_d.StoreOp(tail_next_op, tail_ptr, [], ip=critical_ip)
        openmp_d.TerminatorOp(ip=critical_ip)
        openmp_d.FlushOp([], ip=then_ip)
        true_val = arith_d.ConstantOp(
            IntegerType.get_signless(1, module.context), 1, ip=then_ip
        )
        scf_d.YieldOp(results_=[true_val.result], ip=then_ip)
        # Else block (Full)
        else_ip = InsertionPoint(if_op.else_block)
        false_val = arith_d.ConstantOp(
            IntegerType.get_signless(1, module.context), 0, ip=else_ip
        )
        scf_d.YieldOp(results_=[false_val.result], ip=else_ip)
        stream_access_op.results[0].replace_all_uses_with(if_op.results[0])
        stream_access_op.operation.erase()
        return True
    if isinstance(stream_access_op, allo_d.StreamTryGetOp):
        openmp_d.FlushOp([], ip=replace_ip)
        head_val_op = memref_d.LoadOp(memref=head_ptr, indices=[], ip=replace_ip)
        tail_val_op = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=replace_ip)
        is_empty = arith_d.CmpIOp(
            0, lhs=head_val_op.result, rhs=tail_val_op.result, ip=replace_ip
        )
        is_not_empty = arith_d.CmpIOp(
            1, lhs=head_val_op.result, rhs=tail_val_op.result, ip=replace_ip
        )
        orig_got_val = stream_access_op.results[0]
        expected_type = orig_got_val.type
        if_op = scf_d.IfOp(
            is_not_empty.result,
            [expected_type, IntegerType.get_signless(1, module.context)],
            has_else=True,
            ip=replace_ip,
        )
        # Then block (Not Empty)
        then_ip = InsertionPoint(if_op.then_block)
        head_index_op = index_d.CastUOp(
            output=IndexType.get(module.context), input=head_val_op, ip=then_ip
        )
        head_inc_op = arith_d.AddIOp(
            lhs=head_val_op.result, rhs=const_one.result, ip=then_ip
        )
        head_next_op = arith_d.RemUIOp(
            lhs=head_inc_op.result, rhs=const_fifo_depth.result, ip=then_ip
        )
        if isinstance(expected_type, MemRefType):
            element_alloc_op = memref_d.AllocOp(
                memref=expected_type,
                dynamicSizes=[],
                symbolOperands=[],
                ip=then_ip,
            )
            rank = expected_type.rank
            for_ip = then_ip
            for_induction_vars = []
            for_ips = []
            for i in range(rank):
                for_loop_op = affine_d.AffineForOp(
                    0, expected_type.get_dim_size(i), ip=for_ip
                )
                for_induction_vars.append(for_loop_op.induction_variable)
                for_ip = InsertionPoint(for_loop_op.body)
                for_ips.append(for_ip)
            element_dim_map = AffineMap.get(
                dim_count=rank,
                symbol_count=0,
                exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                context=module.context,
            )
            element_load_op = memref_d.LoadOp(
                memref=fifo_ptr,
                indices=[head_index_op] + for_induction_vars,
                ip=for_ip,
            )
            affine_d.AffineStoreOp(
                value=element_load_op,
                memref=element_alloc_op,
                indices=for_induction_vars,
                map=AffineMapAttr.get(element_dim_map),
                ip=for_ip,
            )
            for ip in for_ips:
                affine_d.AffineYieldOp([], ip=ip)
            data_val = element_alloc_op.result
        else:
            new_get_op = memref_d.LoadOp(
                memref=fifo_ptr, indices=[head_index_op], ip=then_ip
            )
            loaded_value = new_get_op.result
            if loaded_value.type != expected_type:
                if isinstance(loaded_value.type, IntegerType) and isinstance(
                    expected_type, IntegerType
                ):
                    if loaded_value.type.width < expected_type.width:
                        if loaded_value.type.is_signed:
                            loaded_value = arith_d.ExtSIOp(
                                expected_type, loaded_value, ip=then_ip
                            )
                        else:
                            loaded_value = arith_d.ExtUIOp(
                                expected_type, loaded_value, ip=then_ip
                            )
                    elif loaded_value.type.width > expected_type.width:
                        loaded_value = arith_d.TruncIOp(
                            expected_type, loaded_value, ip=then_ip
                        )
            data_val = loaded_value
        _advance_get_ts(ts_ptr, head_index_op, clock_arg, then_ip)
        critical_op = openmp_d.CriticalOp(ip=then_ip)
        critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
        memref_d.StoreOp(head_next_op, head_ptr, [], ip=critical_ip)
        openmp_d.TerminatorOp(ip=critical_ip)
        openmp_d.FlushOp([], ip=then_ip)
        true_val = arith_d.ConstantOp(
            IntegerType.get_signless(1, module.context), 1, ip=then_ip
        )
        scf_d.YieldOp(results_=[data_val, true_val.result], ip=then_ip)
        # Else block (Empty)
        else_ip = InsertionPoint(if_op.else_block)
        if isinstance(expected_type, MemRefType):
            dummy_data = memref_d.AllocOp(
                memref=expected_type,
                dynamicSizes=[],
                symbolOperands=[],
                ip=else_ip,
            )
            dummy_data_val = dummy_data.result
        elif isinstance(expected_type, IntegerType):
            dummy_data_val = arith_d.ConstantOp(expected_type, 0, ip=else_ip).result
        elif isinstance(expected_type, FloatType):
            dummy_data_val = arith_d.ConstantOp(
                expected_type, 0.0, ip=else_ip
            ).result
        else:
            raise NotImplementedError(
                f"Unsupported stream type for dummy data: {expected_type}"
            )
        false_val = arith_d.ConstantOp(
            IntegerType.get_signless(1, module.context), 0, ip=else_ip
        )
        scf_d.YieldOp(results_=[dummy_data_val, false_val.result], ip=else_ip)
        stream_access_op.results[0].replace_all_uses_with(if_op.results[0])
        stream_access_op.results[1].replace_all_uses_with(if_op.results[1])
        stream_access_op.operation.erase()
        return True
    return False


def _process_function_streams(
    module: Module,
    func: func_d.FuncOp,
    processed_funcs: set,
    all_pe_calls_by_func: dict,
):
    """
    Process streams and PE calls within a single function.
    Returns (stream_struct_table, stream_type_table, pe_call_define_ops, stream_construct_ops)
    for use by the caller.
    """
    func_name = str(func.sym_name).strip('"')
    if func_name in processed_funcs:
        return {}, {}, {}, {}
    processed_funcs.add(func_name)

    if not isinstance(func.body, Region) or len(func.body.blocks) == 0:
        return {}, {}, {}, {}

    func_ops = func.body.blocks[0].operations
    pe_call_define_ops: dict[func_d.CallOp, func_d.FuncOp] = {}
    stream_construct_ops: dict[str, allo_d.StreamConstructOp] = {}

    # Collect PE calls and stream construct ops in this function.
    # The top-level scan handles direct (non-nested) calls and local stream
    # constructs, which is sufficient to identify top-level parallel PE calls.
    for op in func_ops:
        if isinstance(op, memref_d.AllocOp):
            continue
        if isinstance(op, func_d.CallOp):
            callee_name = str(op.callee)[1:]
            if not callee_name.startswith(("load_buf", "store_res")):
                for mod_op in module.body.operations:
                    if isinstance(mod_op, func_d.FuncOp):
                        if callee_name == str(mod_op.sym_name).strip('"'):
                            pe_call_define_ops[op] = mod_op
                            _process_function_streams(
                                module, mod_op, processed_funcs, all_pe_calls_by_func
                            )
                            break
        elif isinstance(op, allo_d.StreamConstructOp):
            stream_name = str(op.attributes["name"]).strip('"')
            stream_construct_ops[stream_name] = op

    # Deep scan: also reach func.call ops nested inside affine.for / scf.if /
    # other control-flow regions. Without this, a sub-region call like
    # ``inner(buf)`` placed inside ``for _ in range(N): inner(buf)`` is not
    # discovered by the top-level scan above, and the callee's own
    # ``allo.stream_put`` / ``allo.stream_get`` ops survive into LLVM
    # lowering -- ``convert-func-to-llvm`` then fails with
    # "cannot be converted to LLVM IR: missing LLVMTranslationDialectInterface
    # registration for dialect for op: func.func".
    #
    # We do not add nested calls to ``pe_call_define_ops`` here: nested
    # calls do not pass parent-region streams as call args (those would
    # have been visible at the top level), so there is no parent-side
    # arg-mapping to perform. We only need to ensure the callee gets
    # processed so its internal streams are lowered.
    nested_calls: list = []
    recursive_collect_ops(func, func_d.CallOp, nested_calls)
    for call_op in nested_calls:
        if call_op in pe_call_define_ops:
            continue
        callee_name = str(call_op.callee)[1:]
        if callee_name.startswith(("load_buf", "store_res", "usleep")):
            continue
        for mod_op in module.body.operations:
            if isinstance(mod_op, func_d.FuncOp):
                if callee_name == str(mod_op.sym_name).strip('"'):
                    _process_function_streams(
                        module, mod_op, processed_funcs, all_pe_calls_by_func
                    )
                    break

    # If no streams, nothing to do for this function
    if not stream_construct_ops:
        return {}, {}, pe_call_define_ops, {}

    # Construct Memref variables for pipes
    stream_struct_table: dict[str, OpResult] = {}  # stream name: stream struct
    stream_type_table: dict[str, MemRefType] = {}
    int_type = IntegerType.get_signless(32, module.context)
    memref_scalar_int_type = MemRefType.get([], int_type)
    empty_map = AffineMapAttr.get(AffineMap.get(0, 0, []))
    const_0_defined = False
    const_zero = None

    for stream_access_op in stream_construct_ops.values():
        stream_name = stream_access_op.attributes["name"]
        stream_type = allo_d.StreamType(stream_access_op.result.type)
        stream_item_type = stream_type.base_type
        stream_depth = stream_type.depth
        assert isinstance(stream_item_type, (MemRefType, IntegerType, FloatType))
        assert isinstance(stream_depth, int)
        ip = InsertionPoint(beforeOperation=stream_access_op)
        if isinstance(stream_item_type, MemRefType):
            item_element_type = stream_item_type.element_type
            if not isinstance(item_element_type, (IntegerType, FloatType)):
                raise NotImplementedError()
            memref_stream_type = MemRefType.get(
                shape=[stream_depth + 1] + stream_item_type.shape,
                element_type=item_element_type,
            )
        else:
            memref_stream_type = MemRefType.get(
                shape=[stream_depth + 1], element_type=stream_item_type
            )
        stream_memref_op = memref_d.AllocOp(memref_stream_type, [], [], ip=ip)
        stream_head_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
        stream_tail_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
        # Phase 2: parallel timestamp ring -- ts_ring[i] holds the producer's
        # simulated-time clock at the moment element i was put. Same length as
        # the data ring; i64 to match the per-PE clock.
        i64_type = IntegerType.get_signless(64, func.context)
        ts_ring_type = MemRefType.get([stream_depth + 1], i64_type)
        ts_ring_op = memref_d.AllocOp(ts_ring_type, [], [], ip=ip)
        if not const_0_defined:
            const_zero = arith_d.ConstantOp(int_type, 0, ip=ip)
            const_0_defined = True
        memref_d.StoreOp(value=const_zero, memref=stream_head_op, indices=[], ip=ip)
        memref_d.StoreOp(value=const_zero, memref=stream_tail_op, indices=[], ip=ip)
        fifo_struct_type = allo_d.StructType.get(
            members=[
                memref_stream_type,
                memref_scalar_int_type,
                memref_scalar_int_type,
                ts_ring_type,
            ],
            context=func.context,
        )
        fifo_struct_op = allo_d.StructConstructOp(
            output=fifo_struct_type,
            input=[stream_memref_op, stream_head_op, stream_tail_op, ts_ring_op],
            ip=ip,
        )
        fifo_struct_memref_type = MemRefType.get([], fifo_struct_type)
        stream_memref_op = memref_d.AllocOp(fifo_struct_memref_type, [], [], ip=ip)
        stream_memref_op.attributes["name"] = stream_name
        affine_d.AffineStoreOp(
            value=fifo_struct_op,
            memref=stream_memref_op,
            indices=[],
            map=empty_map,
            ip=ip,
        )
        stream_name_str = str(stream_name).strip('"')
        stream_head_op.attributes["name"] = StringAttr.get(f"{stream_name_str}_head")
        stream_tail_op.attributes["name"] = StringAttr.get(f"{stream_name_str}_tail")
        stream_memref_op.attributes["name"] = stream_name
        stream_struct_table[stream_name_str] = stream_memref_op.result
        stream_type_table[stream_name_str] = memref_stream_type

    # Transform the stream operations in function calls
    for call_op, func_def_op in pe_call_define_ops.items():
        # Get the correspondence between arguments and passed pipes
        arg_stream_table: dict[BlockArgument, str] = {}  # arg: stream name
        assert isinstance(call_op.operands_, OpOperandList)
        assert isinstance(func_def_op.arguments, BlockArgumentList)
        assert len(call_op.operands_) == len(func_def_op.arguments)
        # 1. Update this call site and callee signature for all stream arguments
        for i in range(len(call_op.operands_)):
            arg_instance = call_op.operands_[i]
            for stream_name, stream_construct_op in stream_construct_ops.items():
                if Value(stream_construct_op.result) == arg_instance:
                    arg_def = func_def_op.arguments[i]
                    stream_memref = stream_struct_table[stream_name]
                    arg_stream_table[arg_def] = stream_name
                    # Change argument definitions
                    arg_def.set_type(stream_memref.type)
                    old_func_type = func_def_op.type
                    new_inputs = list(old_func_type.inputs)
                    new_inputs[arg_def.arg_number] = stream_memref.type
                    new_func_type = FunctionType.get(
                        inputs=new_inputs,
                        results=old_func_type.results,
                        context=old_func_type.context,
                    )
                    func_def_op.attributes["function_type"] = TypeAttr.get(
                        new_func_type, module.context
                    )
                    call_op.operands_[arg_def.arg_number] = stream_memref
        # Collect and replace `stream_get`s and `stream_put`s
        func_stream_ops = []
        recursive_collect_ops(
            func_def_op,
            (
                allo_d.StreamGetOp,
                allo_d.StreamPutOp,
                allo_d.StreamTryGetOp,
                allo_d.StreamTryPutOp,
                allo_d.StreamEmptyOp,
                allo_d.StreamFullOp,
            ),
            func_stream_ops,
        )
        for stream_access_op in func_stream_ops:
            assert isinstance(
                stream_access_op,
                (
                    allo_d.StreamGetOp,
                    allo_d.StreamPutOp,
                    allo_d.StreamTryGetOp,
                    allo_d.StreamTryPutOp,
                    allo_d.StreamEmptyOp,
                    allo_d.StreamFullOp,
                ),
            )
            replace_ip = InsertionPoint(beforeOperation=stream_access_op)
            # Have to leverage weak typing here
            stream = stream_access_op.stream
            # Check if this stream is a block argument (passed from caller)
            # If not (e.g., local stream_construct), skip it
            try:
                stream_arg = BlockArgument(stream)
            except ValueError:
                # Not a block argument, skip - will be handled elsewhere
                continue
            # Check if this stream is in our arg_stream_table
            if stream_arg not in arg_stream_table:
                continue
            stream_name = arg_stream_table[stream_arg]
            stream_type = stream_type_table[stream_name]
            stream_memref = stream_struct_table[stream_name]
            # FIFO access
            # Spin and wait for the FIFO to be not full
            assert isinstance(stream_memref.type, MemRefType)
            stream_struct = affine_d.AffineLoadOp(
                result=stream_memref.type.element_type,
                memref=stream_arg,
                indices=[],
                map=empty_map,
                ip=replace_ip,
            )
            head_ptr = allo_d.StructGetOp(
                output=memref_scalar_int_type,
                input=stream_struct,
                index=1,
                ip=replace_ip,
            )
            tail_ptr = allo_d.StructGetOp(
                output=memref_scalar_int_type,
                input=stream_struct,
                index=2,
                ip=replace_ip,
            )
            fifo_ptr = allo_d.StructGetOp(
                output=stream_type, input=stream_struct, index=0, ip=replace_ip
            )
            ts_ring_type = MemRefType.get(
                [stream_type.get_dim_size(0)],
                IntegerType.get_signless(64, module.context),
            )
            ts_ptr = allo_d.StructGetOp(
                output=ts_ring_type, input=stream_struct, index=3, ip=replace_ip
            )
            clock_arg = _clock_of(func_def_op)
            const_one = arith_d.ConstantOp(int_type, 1, ip=replace_ip)
            const_fifo_depth = arith_d.ConstantOp(
                int_type, stream_type.get_dim_size(0), ip=replace_ip
            )
            if _lower_nb_stream_op(
                stream_access_op, head_ptr, tail_ptr, fifo_ptr,
                stream_type, const_one, const_fifo_depth, module, replace_ip,
                ts_ptr, clock_arg,
            ):
                continue
            if isinstance(stream_access_op, allo_d.StreamPutOp):
                openmp_d.FlushOp([], ip=replace_ip)
                tail_val_op = memref_d.LoadOp(
                    memref=tail_ptr, indices=[], ip=replace_ip
                )

                tail_inc_op = arith_d.AddIOp(
                    lhs=tail_val_op.result, rhs=const_one.result, ip=replace_ip
                )
                tail_next_op = arith_d.RemUIOp(
                    lhs=tail_inc_op.result,
                    rhs=const_fifo_depth.result,
                    ip=replace_ip,
                )
            else:
                assert isinstance(stream_access_op, allo_d.StreamGetOp)
                head_val_op = memref_d.LoadOp(
                    memref=head_ptr, indices=[], ip=replace_ip
                )
                head_inc_op = arith_d.AddIOp(
                    lhs=head_val_op.result, rhs=const_one.result, ip=replace_ip
                )
                head_next_op = arith_d.RemUIOp(
                    lhs=head_inc_op.result,
                    rhs=const_fifo_depth.result,
                    ip=replace_ip,
                )
            spin_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
            assert isinstance(spin_while_op.before, Region)
            assert isinstance(spin_while_op.after, Region)
            before_block = Block.create_at_start(
                parent=spin_while_op.before, arg_types=[]
            )
            before_ip = InsertionPoint(before_block)
            openmp_d.FlushOp([], ip=before_ip)
            after_block = Block.create_at_start(
                parent=spin_while_op.after, arg_types=[]
            )
            after_ip = InsertionPoint(after_block)
            openmp_d.TaskyieldOp(ip=after_ip)
            # Inject usleep(1) to prevent CPU starvation
            c1 = arith_d.ConstantOp(
                IntegerType.get_signless(32, module.context), 1, ip=after_ip
            )
            func_d.CallOp([], FlatSymbolRefAttr.get("usleep"), [c1], ip=after_ip)
            scf_d.YieldOp(results_=[], ip=after_ip)
            if isinstance(stream_access_op, allo_d.StreamPutOp):
                head_val_op = memref_d.LoadOp(memref=head_ptr, indices=[], ip=before_ip)
                cmp_op = arith_d.CmpIOp(
                    predicate=0, lhs=head_val_op, rhs=tail_next_op, ip=before_ip
                )
                scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                data = stream_access_op.data
                assert isinstance(data, Value)  # Vector or scalar
                tail_index_op = index_d.CastUOp(
                    output=IndexType.get(module.context),
                    input=tail_val_op,
                    ip=replace_ip,
                )
                if isinstance(data.type, MemRefType):  # Vector
                    # Data is an `alloc` pointer and should be loaded first
                    element_type = data.type.element_type
                    if not isinstance(element_type, (IntegerType, FloatType)):
                        # May get StructType involved in the future
                        raise NotImplementedError()
                    rank = data.type.rank
                    for_ip = replace_ip
                    for_induction_vars = []
                    for_ips: list[InsertionPoint] = (
                        []
                    )  # Reserved to insert affine.yield ops later
                    for i in range(rank):
                        dim_size = data.type.get_dim_size(i)
                        for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                        for_induction_vars.append(for_loop_op.induction_variable)
                        for_ip = InsertionPoint(for_loop_op.body)
                        for_ips.append(for_ip)
                    element_dim_map = AffineMap.get(
                        dim_count=rank,
                        symbol_count=0,
                        exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                        context=module.context,
                    )
                    element_load_op = affine_d.AffineLoadOp(
                        result=element_type,
                        memref=data,
                        indices=for_induction_vars,
                        map=AffineMapAttr.get(element_dim_map),
                        ip=for_ip,
                    )  # Fetch the element
                    memref_d.StoreOp(
                        value=element_load_op,
                        memref=fifo_ptr,
                        indices=[tail_index_op] + for_induction_vars,
                        ip=for_ip,
                    )  # Put the element to the stream
                    for ip in for_ips:
                        affine_d.AffineYieldOp([], ip=ip)
                else:  # Scalar
                    # Ensure data type matches the memref element type
                    fifo_element_type = stream_type.element_type
                    store_value = data
                    if data.type != fifo_element_type:
                        # Cast the data to match the expected element type
                        if isinstance(data.type, IntegerType) and isinstance(
                            fifo_element_type, IntegerType
                        ):
                            if data.type.width > fifo_element_type.width:
                                store_value = arith_d.TruncIOp(
                                    fifo_element_type, data, ip=replace_ip
                                )
                            elif data.type.width < fifo_element_type.width:
                                if data.type.is_signed:
                                    store_value = arith_d.ExtSIOp(
                                        fifo_element_type, data, ip=replace_ip
                                    )
                                else:
                                    store_value = arith_d.ExtUIOp(
                                        fifo_element_type, data, ip=replace_ip
                                    )
                    memref_d.StoreOp(
                        value=store_value,
                        memref=fifo_ptr,
                        indices=[tail_index_op],
                        ip=replace_ip,
                    )
                # Atomic update of tail
                _stamp_put_ts(ts_ptr, tail_index_op, clock_arg, replace_ip)
                critical_op = openmp_d.CriticalOp(ip=replace_ip)
                critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
                memref_d.StoreOp(tail_next_op, tail_ptr, [], ip=critical_ip)
                openmp_d.TerminatorOp(ip=critical_ip)
                openmp_d.FlushOp([], ip=replace_ip)
            else:
                assert isinstance(stream_access_op, allo_d.StreamGetOp)
                tail_val_op = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=before_ip)
                cmp_op = arith_d.CmpIOp(
                    0, lhs=head_val_op, rhs=tail_val_op, ip=before_ip
                )
                scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                orig_got_val = stream_access_op.res
                assert isinstance(orig_got_val, OpResult)
                head_index_op = index_d.CastUOp(
                    output=IndexType.get(module.context),
                    input=head_val_op,
                    ip=replace_ip,
                )
                if isinstance(orig_got_val.type, MemRefType):
                    element_type = orig_got_val.type.element_type
                    if not isinstance(element_type, (IntegerType, FloatType)):
                        raise NotImplementedError()
                    rank = orig_got_val.type.rank
                    assert rank > 0
                    # Create a memref for the loaded element
                    element_alloc_op = memref_d.AllocOp(
                        memref=orig_got_val.type,
                        dynamicSizes=[],
                        symbolOperands=[],
                        ip=replace_ip,
                    )
                    orig_got_val.replace_all_uses_with(element_alloc_op.result)
                    # Create the element load/store loop
                    for_ip = replace_ip
                    for_induction_vars = []
                    for_ips: list[InsertionPoint] = []
                    for i in range(rank):
                        for_loop_op = affine_d.AffineForOp(
                            0,
                            orig_got_val.type.get_dim_size(i),
                            ip=for_ip,
                        )
                        for_induction_vars.append(for_loop_op.induction_variable)
                        for_ip = InsertionPoint(for_loop_op.body)
                        for_ips.append(for_ip)
                    element_dim_map = AffineMap.get(
                        dim_count=rank,
                        symbol_count=0,
                        exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                        context=module.context,
                    )
                    element_load_op = memref_d.LoadOp(
                        memref=fifo_ptr,
                        indices=[head_index_op] + for_induction_vars,
                        ip=for_ip,  # The innermost Loop body
                    )
                    affine_d.AffineStoreOp(
                        value=element_load_op,
                        memref=element_alloc_op,
                        indices=for_induction_vars,
                        map=AffineMapAttr.get(element_dim_map),
                        ip=for_ip,
                    )
                    for ip in for_ips:
                        affine_d.AffineYieldOp([], ip=ip)
                else:  # Scalar
                    new_get_op = memref_d.LoadOp(
                        memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                    )
                    # Ensure loaded type matches the expected result type
                    loaded_value = new_get_op.result
                    expected_type = orig_got_val.type
                    if loaded_value.type != expected_type:
                        # Cast the loaded value to match the expected type
                        if isinstance(loaded_value.type, IntegerType) and isinstance(
                            expected_type, IntegerType
                        ):
                            if loaded_value.type.width < expected_type.width:
                                if loaded_value.type.is_signed:
                                    loaded_value = arith_d.ExtSIOp(
                                        expected_type, loaded_value, ip=replace_ip
                                    )
                                else:
                                    loaded_value = arith_d.ExtUIOp(
                                        expected_type, loaded_value, ip=replace_ip
                                    )
                            elif loaded_value.type.width > expected_type.width:
                                loaded_value = arith_d.TruncIOp(
                                    expected_type, loaded_value, ip=replace_ip
                                )
                    orig_got_val.replace_all_uses_with(loaded_value)
                _advance_get_ts(ts_ptr, head_index_op, clock_arg, replace_ip)
                critical_op = openmp_d.CriticalOp(ip=replace_ip)
                critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
                memref_d.StoreOp(head_next_op, head_ptr, [], ip=critical_ip)
                openmp_d.TerminatorOp(ip=critical_ip)
            stream_access_op.operation.erase()

    # Also handle local stream operations within this function directly
    # (streams defined locally and used locally via stream_get/put, not passed to callees)
    local_stream_ops = []
    recursive_collect_ops(
        func,
        (
            allo_d.StreamGetOp,
            allo_d.StreamPutOp,
            allo_d.StreamTryGetOp,
            allo_d.StreamTryPutOp,
            allo_d.StreamEmptyOp,
            allo_d.StreamFullOp,
        ),
        local_stream_ops,
    )

    for stream_access_op in local_stream_ops:
        # Get the stream this op uses
        stream = stream_access_op.stream
        # Check if this stream is one of our locally-defined streams
        stream_name = None
        for sname, sop in stream_construct_ops.items():
            if Value(sop.result) == stream:
                stream_name = sname
                break
        if stream_name is None:
            continue  # Not a local stream we're processing
        if stream_name not in stream_struct_table:
            continue  # Stream wasn't processed (shouldn't happen)

        stream_type = stream_type_table[stream_name]
        stream_memref = stream_struct_table[stream_name]
        replace_ip = InsertionPoint(beforeOperation=stream_access_op)

        # FIFO access - transform the local stream operation
        assert isinstance(stream_memref.type, MemRefType)
        stream_struct = affine_d.AffineLoadOp(
            result=stream_memref.type.element_type,
            memref=stream_memref,
            indices=[],
            map=empty_map,
            ip=replace_ip,
        )
        head_ptr = allo_d.StructGetOp(
            output=memref_scalar_int_type,
            input=stream_struct,
            index=1,
            ip=replace_ip,
        )
        tail_ptr = allo_d.StructGetOp(
            output=memref_scalar_int_type,
            input=stream_struct,
            index=2,
            ip=replace_ip,
        )
        fifo_ptr = allo_d.StructGetOp(
            output=stream_type, input=stream_struct, index=0, ip=replace_ip
        )
        ts_ring_type = MemRefType.get(
            [stream_type.get_dim_size(0)],
            IntegerType.get_signless(64, module.context),
        )
        ts_ptr = allo_d.StructGetOp(
            output=ts_ring_type, input=stream_struct, index=3, ip=replace_ip
        )
        clock_arg = _clock_of(func)
        const_one = arith_d.ConstantOp(int_type, 1, ip=replace_ip)
        const_fifo_depth = arith_d.ConstantOp(
            int_type, stream_type.get_dim_size(0), ip=replace_ip
        )
        if _lower_nb_stream_op(
            stream_access_op, head_ptr, tail_ptr, fifo_ptr,
            stream_type, const_one, const_fifo_depth, module, replace_ip,
            ts_ptr, clock_arg,
        ):
            continue
        if isinstance(stream_access_op, allo_d.StreamPutOp):
            openmp_d.FlushOp([], ip=replace_ip)
            tail_val_op = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=replace_ip)
            tail_inc_op = arith_d.AddIOp(
                lhs=tail_val_op.result, rhs=const_one.result, ip=replace_ip
            )
            tail_next_op = arith_d.RemUIOp(
                lhs=tail_inc_op.result,
                rhs=const_fifo_depth.result,
                ip=replace_ip,
            )
            spin_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
            assert isinstance(spin_while_op.before, Region)
            assert isinstance(spin_while_op.after, Region)
            before_block = Block.create_at_start(
                parent=spin_while_op.before, arg_types=[]
            )
            before_ip = InsertionPoint(before_block)
            openmp_d.FlushOp([], ip=before_ip)
            after_block = Block.create_at_start(
                parent=spin_while_op.after, arg_types=[]
            )
            after_ip = InsertionPoint(after_block)
            openmp_d.TaskyieldOp(ip=after_ip)
            # Inject usleep(1) to prevent CPU starvation
            c1 = arith_d.ConstantOp(
                IntegerType.get_signless(32, module.context), 1, ip=after_ip
            )
            func_d.CallOp([], FlatSymbolRefAttr.get("usleep"), [c1], ip=after_ip)
            scf_d.YieldOp(results_=[], ip=after_ip)
            head_val_op = memref_d.LoadOp(memref=head_ptr, indices=[], ip=before_ip)
            cmp_op = arith_d.CmpIOp(
                predicate=0, lhs=head_val_op, rhs=tail_next_op, ip=before_ip
            )
            scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
            data = stream_access_op.data
            assert isinstance(data, Value)
            tail_index_op = index_d.CastUOp(
                output=IndexType.get(module.context),
                input=tail_val_op,
                ip=replace_ip,
            )
            if isinstance(data.type, MemRefType):
                element_type = data.type.element_type
                if not isinstance(element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                rank = data.type.rank
                for_ip = replace_ip
                for_induction_vars = []
                for_ips: list[InsertionPoint] = []
                for i in range(rank):
                    dim_size = data.type.get_dim_size(i)
                    for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                    for_induction_vars.append(for_loop_op.induction_variable)
                    for_ip = InsertionPoint(for_loop_op.body)
                    for_ips.append(for_ip)
                element_dim_map = AffineMap.get(
                    dim_count=rank,
                    symbol_count=0,
                    exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                    context=module.context,
                )
                element_load_op = affine_d.AffineLoadOp(
                    result=element_type,
                    memref=data,
                    indices=for_induction_vars,
                    map=AffineMapAttr.get(element_dim_map),
                    ip=for_ip,
                )
                memref_d.StoreOp(
                    value=element_load_op,
                    memref=fifo_ptr,
                    indices=[tail_index_op] + for_induction_vars,
                    ip=for_ip,
                )
                for ip in for_ips:
                    affine_d.AffineYieldOp([], ip=ip)
            else:
                fifo_element_type = stream_type.element_type
                store_value = data
                if data.type != fifo_element_type:
                    if isinstance(data.type, IntegerType) and isinstance(
                        fifo_element_type, IntegerType
                    ):
                        if data.type.width > fifo_element_type.width:
                            store_value = arith_d.TruncIOp(
                                fifo_element_type, data, ip=replace_ip
                            )
                        elif data.type.width < fifo_element_type.width:
                            if data.type.is_signed:
                                store_value = arith_d.ExtSIOp(
                                    fifo_element_type, data, ip=replace_ip
                                )
                            else:
                                store_value = arith_d.ExtUIOp(
                                    fifo_element_type, data, ip=replace_ip
                                )
                memref_d.StoreOp(
                    value=store_value,
                    memref=fifo_ptr,
                    indices=[tail_index_op],
                    ip=replace_ip,
                )
            _stamp_put_ts(ts_ptr, tail_index_op, clock_arg, replace_ip)
            critical_op = openmp_d.CriticalOp(ip=replace_ip)
            critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
            memref_d.StoreOp(tail_next_op, tail_ptr, [], ip=critical_ip)
            openmp_d.TerminatorOp(ip=critical_ip)
            openmp_d.FlushOp([], ip=replace_ip)
        else:
            assert isinstance(stream_access_op, allo_d.StreamGetOp)
            head_val_op = memref_d.LoadOp(memref=head_ptr, indices=[], ip=replace_ip)
            head_inc_op = arith_d.AddIOp(
                lhs=head_val_op.result, rhs=const_one.result, ip=replace_ip
            )
            head_next_op = arith_d.RemUIOp(
                lhs=head_inc_op.result,
                rhs=const_fifo_depth.result,
                ip=replace_ip,
            )
            spin_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
            assert isinstance(spin_while_op.before, Region)
            assert isinstance(spin_while_op.after, Region)
            before_block = Block.create_at_start(
                parent=spin_while_op.before, arg_types=[]
            )
            before_ip = InsertionPoint(before_block)
            openmp_d.FlushOp([], ip=before_ip)
            after_block = Block.create_at_start(
                parent=spin_while_op.after, arg_types=[]
            )
            after_ip = InsertionPoint(after_block)
            openmp_d.TaskyieldOp(ip=after_ip)
            # Inject usleep(1) to prevent CPU starvation
            c1 = arith_d.ConstantOp(
                IntegerType.get_signless(32, module.context), 1, ip=after_ip
            )
            func_d.CallOp([], FlatSymbolRefAttr.get("usleep"), [c1], ip=after_ip)
            scf_d.YieldOp(results_=[], ip=after_ip)
            tail_val_op = memref_d.LoadOp(memref=tail_ptr, indices=[], ip=before_ip)
            cmp_op = arith_d.CmpIOp(0, lhs=head_val_op, rhs=tail_val_op, ip=before_ip)
            scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
            orig_got_val = stream_access_op.res
            assert isinstance(orig_got_val, OpResult)
            head_index_op = index_d.CastUOp(
                output=IndexType.get(module.context),
                input=head_val_op,
                ip=replace_ip,
            )
            if isinstance(orig_got_val.type, MemRefType):
                element_type = orig_got_val.type.element_type
                if not isinstance(element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                rank = orig_got_val.type.rank
                assert rank > 0
                element_alloc_op = memref_d.AllocOp(
                    memref=orig_got_val.type,
                    dynamicSizes=[],
                    symbolOperands=[],
                    ip=replace_ip,
                )
                orig_got_val.replace_all_uses_with(element_alloc_op.result)
                for_ip = replace_ip
                for_induction_vars = []
                for_ips: list[InsertionPoint] = []
                for i in range(rank):
                    for_loop_op = affine_d.AffineForOp(
                        0, orig_got_val.type.get_dim_size(i), ip=for_ip
                    )
                    for_induction_vars.append(for_loop_op.induction_variable)
                    for_ip = InsertionPoint(for_loop_op.body)
                    for_ips.append(for_ip)
                element_dim_map = AffineMap.get(
                    dim_count=rank,
                    symbol_count=0,
                    exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                    context=module.context,
                )
                element_load_op = memref_d.LoadOp(
                    memref=fifo_ptr,
                    indices=[head_index_op] + for_induction_vars,
                    ip=for_ip,
                )
                affine_d.AffineStoreOp(
                    value=element_load_op,
                    memref=element_alloc_op,
                    indices=for_induction_vars,
                    map=AffineMapAttr.get(element_dim_map),
                    ip=for_ip,
                )
                for ip in for_ips:
                    affine_d.AffineYieldOp([], ip=ip)
            else:
                new_get_op = memref_d.LoadOp(
                    memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                )
                loaded_value = new_get_op.result
                expected_type = orig_got_val.type
                if loaded_value.type != expected_type:
                    if isinstance(loaded_value.type, IntegerType) and isinstance(
                        expected_type, IntegerType
                    ):
                        if loaded_value.type.width < expected_type.width:
                            if loaded_value.type.is_signed:
                                loaded_value = arith_d.ExtSIOp(
                                    expected_type, loaded_value, ip=replace_ip
                                )
                            else:
                                loaded_value = arith_d.ExtUIOp(
                                    expected_type, loaded_value, ip=replace_ip
                                )
                        elif loaded_value.type.width > expected_type.width:
                            loaded_value = arith_d.TruncIOp(
                                expected_type, loaded_value, ip=replace_ip
                            )
                orig_got_val.replace_all_uses_with(loaded_value)
            _advance_get_ts(ts_ptr, head_index_op, clock_arg, replace_ip)
            critical_op = openmp_d.CriticalOp(ip=replace_ip)
            critical_ip = InsertionPoint(Block.create_at_start(critical_op.region))
            memref_d.StoreOp(head_next_op, head_ptr, [], ip=critical_ip)
            openmp_d.TerminatorOp(ip=critical_ip)
        stream_access_op.operation.erase()

    # Erase stream construct ops for this function
    for op in stream_construct_ops.values():
        op.operation.erase()

    # Accumulate PE calls keyed by function for recursive OMP injection
    if pe_call_define_ops:
        all_pe_calls_by_func[func_name] = pe_call_define_ops

    return (
        stream_struct_table,
        stream_type_table,
        pe_call_define_ops,
        stream_construct_ops,
    )


def _inject_omp_parallel_sections(pe_call_define_ops):
    """Wrap a set of func.call ops in omp.parallel > omp.sections > omp.section blocks."""
    assert len(pe_call_define_ops) > 0
    omp_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])
    omp_parallel_op = openmp_d.ParallelOp([], [], [], [], ip=omp_ip)
    assert isinstance(omp_parallel_op.region, Region)
    omp_parallel_block = Block.create_at_start(omp_parallel_op.region, [])

    # Add `omp.sections`
    ip_omp_parallel = InsertionPoint(omp_parallel_block)
    omp_sections_op = openmp_d.SectionsOp([], [], [], [], ip=ip_omp_parallel)
    omp_sections_block = Block.create_at_start(omp_sections_op.region, [])
    openmp_d.TerminatorOp(ip=ip_omp_parallel)

    # Add `omp.section`s for PE calls
    ip_omp_sections = InsertionPoint(omp_sections_block)
    for call_op in pe_call_define_ops:
        assert isinstance(call_op, OpView)
        omp_section_op = openmp_d.SectionOp(ip=ip_omp_sections)
        omp_section_block = Block.create_at_start(omp_section_op.region, [])
        ip_omp_section = InsertionPoint(omp_section_block)
        omp_term_op = openmp_d.TerminatorOp(ip=ip_omp_section)
        call_op.operation.move_before(omp_term_op.operation)
    openmp_d.TerminatorOp(ip=ip_omp_sections)


def build_dataflow_simulator(module: Module, top_func_name: str):
    # Enable nested OpenMP parallelism so that peer kernels calling
    # sub-regions (which have their own omp.parallel/sections) don't
    # deadlock.  This is safe because the simulator already controls
    # thread counts via omp.sections.
    if os.environ.get("OMP_MAX_ACTIVE_LEVELS") is None:
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "4"
    with module.context, Location.unknown():
        # Declare usleep for spinloop yielding
        found_usleep = False
        for op in module.body.operations:
            if (
                isinstance(op, func_d.FuncOp)
                and op.attributes["sym_name"].value == "usleep"
            ):
                found_usleep = True
                break
        if not found_usleep:
            usleep_type = FunctionType.get(
                [IntegerType.get_signless(32, module.context)],
                [],
            )
            # pylint: disable=unexpected-keyword-arg
            usleep_op = func_d.FuncOp(
                name="usleep",
                type=usleep_type,
                ip=InsertionPoint(module.body),
            )
            usleep_op.attributes["sym_visibility"] = StringAttr.get("private")

        # Process all functions with streams recursively, starting from top
        processed_funcs: set = set()
        all_pe_calls_by_func: dict = {}
        func = find_func_in_module(module, top_func_name)
        assert isinstance(func.body, Region)

        # Phase 1/2 reorder: thread a per-PE clock arg through the calls BEFORE
        # stream lowering, so put/get lowering can stamp/advance the clock.
        _add_pe_clock_args(module, top_func_name)

        # Recursively process the top function and all its callees
        _, _, pe_call_define_ops, _ = _process_function_streams(
            module, func, processed_funcs, all_pe_calls_by_func
        )

        # If no PE calls were found in top function, collect them again from the processed functions
        if not pe_call_define_ops:
            top_func_ops = func.body.blocks[0].operations
            for op in top_func_ops:
                if isinstance(op, func_d.CallOp):
                    callee_name = str(op.callee)[1:]
                    if not callee_name.startswith(("load_buf", "store_res")):
                        for mod_op in module.body.operations:
                            if isinstance(mod_op, func_d.FuncOp):
                                if callee_name == str(mod_op.sym_name).strip('"'):
                                    pe_call_define_ops[op] = mod_op
                                    break
            all_pe_calls_by_func[top_func_name] = pe_call_define_ops

        # Phase 1: insert the per-PE clock increments (after stream lowering, so
        # spin-wait scf.while loops exist and are skipped). Clock args were already
        # threaded before lowering by _add_pe_clock_args above.
        _insert_pe_clock_increments(module)

        # Inject omp.parallel/sections into every function that has PE calls
        for func_pe_calls in all_pe_calls_by_func.values():
            if func_pe_calls:
                _inject_omp_parallel_sections(func_pe_calls)


# This pass is only meant to run on fully lowered MLIR code
# Note: OpenMP operations in lowered IR are not the original operation types anymore
def convert_critical_write_to_atomic_write(module: Module):
    with module.context, Location.unknown():
        omp_critical_ops = []
        for op in module.body:
            if not isinstance(op, llvm_d.LLVMFuncOp):
                continue
            recursive_collect_ops_by_name(op, "omp.critical", omp_critical_ops)
        for critical_op in omp_critical_ops:
            # Transform a critical area with only the store op and omp.terminator
            assert isinstance(critical_op.regions, RegionSequence)
            if len(critical_op.regions) != 1:
                continue
            region = critical_op.regions[0]
            if len(region.blocks) != 1:
                continue
            block = region.blocks[0]
            if len(block.operations) != 2:
                continue
            if (
                not isinstance(block.operations[0], llvm_d.StoreOp)
                or block.operations[1].name != "omp.terminator"
            ):
                continue
            store_op = block.operations[0]
            assert isinstance(store_op, llvm_d.StoreOp)
            store_ip = InsertionPoint(critical_op)
            openmp_d.AtomicWriteOp(x=store_op.addr, expr=store_op.value, ip=store_ip)
            critical_op.operation.erase()


class LLVMOMPModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_dataflow_simulator(self.module, self.top_func_name)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # Start lowering
            # Lower linalg for AIE
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize,"
                "expand-strided-metadata,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            # Lower StructType
            allo_d.lower_composite_type(self.module)
            # Lower bit ops
            allo_d.lower_bit_ops(self.module)
            # Reference: https://discourse.llvm.org/t/help-lowering-affine-loop-to-openmp/72441/9
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-arith-to-llvm,"
                "convert-cf-to-llvm,"
                "convert-openmp-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)
            convert_critical_write_to_atomic_write(self.module)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(os.getenv("LLVM_BUILD_DIR"), "lib", "libomp.so"),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )
