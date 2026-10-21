"""convert-toy-to-csl: lower a toy function to a one-PE Cerebras CSL program.

The output uses the CSL dialects that ship with xDSL (csl_wrapper, csl) plus
linalg on memrefs. xDSL's own passes finish the job and print CSL source:

    df-opt examples/toy_linear_pe.mlir \
        -p convert-toy-to-csl,linalg-to-csl,lower-csl-wrapper,memref-to-dsd -t csl

How each piece maps (everything runs on one processing element, PE):

    function argument  ->  buffer in PE memory, exported so the host can fill it
    toy.matmul         ->  loop nest of csl.fmacs: row i of C += a[i,k] * row k of B
    toy.add            ->  linalg.add, which xDSL's linalg-to-csl turns into @fadds
    toy.relu           ->  linalg.max with 0, which linalg-to-csl turns into @fmaxs
    function result    ->  buffer exported as "out"

Buffers are flattened to 1D, row-major. Only f32 is supported.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, csl, csl_wrapper, func, memref, scf
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    StridedLayoutAttr,
    StringAttr,
    TensorType,
    f32,
)
from xdsl.dialects.linalg.ops import AddOp as LinalgAddOp
from xdsl.dialects.linalg.ops import MaxOp as LinalgMaxOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from df_dialect.toy import AddOp, MatmulOp, ReluOp

PE_PTR = csl.PtrType.get(f32, is_single=False, is_const=False)
"""Type of an exported buffer pointer: [*]f32 in CSL."""


def _shape(value: SSAValue) -> tuple[int, ...]:
    t = value.type
    if not isinstance(t, TensorType) or t.element_type != f32:
        raise PassFailedException(f"convert-toy-to-csl: only f32 tensors are supported, got {t}")
    shape = t.get_shape()
    if any(d < 0 for d in shape):
        raise PassFailedException(f"convert-toy-to-csl: dynamic shapes are not supported: {t}")
    return shape


def _numel(value: SSAValue) -> int:
    n = 1
    for d in _shape(value):
        n *= d
    return n


def _index(block: Block, value: int) -> SSAValue:
    op = arith.ConstantOp(IntegerAttr(value, IndexType()))
    block.add_op(op)
    return op.result


def _row(block: Block, buf: SSAValue, offset: SSAValue, row_len: int) -> SSAValue:
    """Add a view of `buf[offset : offset + row_len]` to `block` and return it."""
    view = memref.SubviewOp(
        buf,
        offsets=[offset],
        sizes=[],
        strides=[],
        static_offsets=[DYNAMIC_INDEX],
        static_sizes=[row_len],
        static_strides=[1],
        result_type=MemRefType(f32, [row_len], StridedLayoutAttr([1], None)),
    )
    block.add_op(view)
    return view.result


def _emit_matmul(body: Block, op: MatmulOp, a: SSAValue, b: SSAValue, c: SSAValue) -> None:
    """C = A @ B, as: C = 0; for i: for k: C[i,:] += A[i,k] * B[k,:]."""
    m, k_dim = _shape(op.lhs)
    _, n = _shape(op.rhs)

    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    body.add_ops([zero, csl.FmovsOp(operands=[[c, zero.result]])])
    c0, c1 = _index(body, 0), _index(body, 1)
    c_m, c_k, c_n = _index(body, m), _index(body, k_dim), _index(body, n)

    # for i in 0..M: c_row = C[i*N : i*N + N]
    i_block = Block(arg_types=[IndexType()])
    i = i_block.args[0]
    i.name_hint = "i"
    c_off = arith.MuliOp(i, c_n)
    a_off = arith.MuliOp(i, c_k)
    i_block.add_ops([c_off, a_off])
    c_row = _row(i_block, c, c_off.result, n)

    # for k in 0..K: c_row += A[i*K + k] * B[k*N : k*N + N]
    k_block = Block(arg_types=[IndexType()])
    k = k_block.args[0]
    k.name_hint = "k"
    a_index = arith.AddiOp(a_off, k)
    a_ik = memref.LoadOp.get(a, [a_index])
    b_off = arith.MuliOp(k, c_n)
    k_block.add_ops([a_index, a_ik, b_off])
    b_row = _row(k_block, b, b_off.result, n)
    k_block.add_ops([csl.FmacsOp(operands=[[c_row, c_row, b_row, a_ik.res]]), scf.YieldOp()])

    i_block.add_ops([scf.ForOp(c0, c_k, c1, [], k_block), scf.YieldOp()])
    body.add_op(scf.ForOp(c0, c_m, c1, [], i_block))


def _emit_add(body: Block, a: SSAValue, b: SSAValue, out: SSAValue) -> None:
    body.add_op(LinalgAddOp(inputs=[a, b], outputs=[out], res=[]))


def _emit_relu(body: Block, a: SSAValue, out: SSAValue) -> None:
    out_type = out.type
    assert isinstance(out_type, MemRefType)
    zeros = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(out_type, [0.0]))
    body.add_ops([zeros, LinalgMaxOp(inputs=[a, zeros.result], outputs=[out], res=[])])


@dataclass(frozen=True)
class ConvertToyToCSLPass(ModulePass):
    """Replace each toy function with a one-PE csl_wrapper.module program."""

    name = "convert-toy-to-csl"

    pe_memory_kb: int = 48
    """Local memory of one PE in KB. 48 matches a WSE-3 PE."""

    target: str = "wse3"
    """CSL target architecture: wse2 or wse3."""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for fn in list(op.ops):
            if isinstance(fn, func.FuncOp) and any(
                isinstance(o, MatmulOp | AddOp | ReluOp) for o in fn.body.walk()
            ):
                Rewriter.replace_op(fn, self._lower_function(fn))

    def _lower_function(self, fn: func.FuncOp) -> csl_wrapper.ModuleOp:
        name = fn.sym_name.data
        ops = list(fn.body.block.ops)
        ret = ops[-1]
        if not isinstance(ret, func.ReturnOp) or len(ret.arguments) != 1:
            raise PassFailedException(f"@{name}: must return exactly one tensor")
        for o in ops[:-1]:
            if not isinstance(o, MatmulOp | AddOp | ReluOp):
                raise PassFailedException(f"@{name}: unsupported op {o.name}")

        # Every tensor value gets its own buffer: arguments first, then op results.
        values: list[SSAValue] = [*fn.body.block.args, *(o.results[0] for o in ops[:-1])]
        total = sum(_numel(v) * 4 for v in values)
        if total > self.pe_memory_kb * 1024:
            raise PassFailedException(
                f"@{name} needs {total // 1024} KB of buffers; one PE has "
                f"{self.pe_memory_kb} KB. Use smaller tensors."
            )

        if self.target not in ("wse2", "wse3"):
            raise PassFailedException(f"target must be wse2 or wse3, got {self.target}")
        wrapper = csl_wrapper.ModuleOp(width=1, height=1, target=StringAttr(self.target))
        wrapper.properties["program_name"] = StringAttr(name)

        # Layout: one PE; fetch the memcpy parameters for it.
        layout = wrapper.layout_module.block
        for arg, hint in zip(layout.args, ["x", "y", "width", "height"], strict=True):
            arg.name_hint = hint
        get_params = csl_wrapper.ImportOp(
            "<memcpy/get_params>", {"width": layout.args[2], "height": layout.args[3]}
        )
        memcpy_params = csl.MemberCallOp(
            "get_params", csl.ComptimeStructType(), get_params, [layout.args[0]]
        )
        assert memcpy_params.result is not None
        memcpy_params.result.name_hint = "memcpy_params"
        layout.add_ops(
            [
                get_params,
                memcpy_params,
                csl_wrapper.YieldOp.from_field_name_mapping({"memcpy_params": memcpy_params}),
            ]
        )
        wrapper.update_program_block_args()

        # Program: buffers, exports, and the compute function.
        program = wrapper.program_module.block
        for arg, hint in zip(program.args, ["width", "height", "memcpy_params"], strict=True):
            arg.name_hint = hint
        memcpy_mod = csl_wrapper.ImportOp("<memcpy/memcpy>", {"": program.args[2]})
        program.add_op(memcpy_mod)

        buffers: dict[SSAValue, SSAValue] = {}
        result = ret.arguments[0]
        for index, value in enumerate(values):
            alloc = memref.AllocOp.get(f32, shape=[_numel(value)])
            program.add_op(alloc)
            buffers[value] = alloc.memref
            if value is result:
                export = "out"
            elif index < len(fn.body.block.args):
                export = value.name_hint or f"arg{index}"
            else:
                alloc.memref.name_hint = f"t{index - len(fn.body.block.args)}_buf"
                continue
            alloc.memref.name_hint = f"{export}_buf"
            ptr = csl.AddressOfOp(alloc.memref, PE_PTR)
            program.add_ops([ptr, csl.SymbolExportOp(export, ptr.res)])

        program.add_op(csl.SymbolExportOp(name, FunctionType.from_lists([], [])))
        body = Block()
        for o in ops[:-1]:
            out = buffers[o.results[0]]
            match o:
                case MatmulOp():
                    _emit_matmul(body, o, buffers[o.lhs], buffers[o.rhs], out)
                case AddOp():
                    _emit_add(body, buffers[o.lhs], buffers[o.rhs], out)
                case ReluOp():
                    _emit_relu(body, buffers[o.input], out)
                case _:
                    raise AssertionError(o)
        body.add_ops([csl.MemberCallOp("unblock_cmd_stream", None, memcpy_mod, []), csl.ReturnOp()])
        program.add_ops(
            [
                csl.FuncOp(name, ((), None), Region(body)),
                csl_wrapper.YieldOp.from_field_name_mapping({}),
            ]
        )
        return wrapper
