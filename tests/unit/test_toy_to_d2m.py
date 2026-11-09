from dataclasses import dataclass
from io import StringIO
from itertools import product
from math import prod
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir.affine import AffineDimExpr
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import PassFailedException

from df_dialect import d2m
from df_dialect.passes.convert_toy_to_d2m import ConvertToyToD2MPass
from df_dialect.toy import Toy
from df_dialect.ttcore import TileType, TTCore


def context():
    ctx = Context()
    for dialect in (builtin.Builtin, func.Func, arith.Arith, linalg.Linalg, Toy, d2m.D2M, TTCore):
        ctx.load_dialect(dialect)
    return ctx


def array(data, shape):
    return ShapedArray(TypedPtr.new_float32(data), list(shape))


@dataclass
class TileTensor:
    shape: tuple[int, ...]
    tiles: list[list[float]]

    def load(self, index):
        return self.tiles[index[0] * self.shape[1] + index[1]]

    def store(self, index, tile):
        self.tiles[index[0] * self.shape[1] + index[1]] = tile

    def copy(self):
        return TileTensor(self.shape, [tile.copy() for tile in self.tiles])


@register_impls
class D2MFunctions(InterpreterFunctions):
    @impl(d2m.EmptyOp)
    def empty(self, interpreter, op, args):
        shape = op.result.type.get_shape()

        if isinstance(op.result.type.element_type, TileType):
            return (TileTensor(shape, [[float("nan")] * 1024 for _ in range(prod(shape))]),)
        return (array([float("nan")] * prod(shape), shape),)

    @impl(d2m.GenericOp)
    def generic(self, interpreter, op, args):
        return interpreter.run_ssacfg_region(op.body, args)

    @impl(d2m.WaitOp)
    def wait(self, interpreter, op, args):
        return args

    @impl(d2m.ReserveOp)
    def reserve(self, interpreter, op, args):
        return args

    @impl_terminator(d2m.YieldOp)
    def yield_(self, interpreter, op, args):
        return ReturnedValues(args), ()

    @impl(d2m.TileTilizeBlockOp)
    def tilize(self, interpreter, op, args):
        source, destination = args
        result = destination.copy()
        for i, j in product(*(range(d) for d in result.shape)):
            result.store(
                (i, j),
                [source.load((i * 32 + r, j * 32 + c)) for r in range(32) for c in range(32)],
            )
        return (result,)

    @impl(d2m.TileUntilizeBlockOp)
    def untilize(self, interpreter, op, args):
        source, destination = args
        result = destination.copy()
        for i, j in product(*(range(d) for d in source.shape)):
            tile = source.load((i, j))
            for r, c in product(range(32), repeat=2):
                result.store((i * 32 + r, j * 32 + c), tile[r * 32 + c])
        return (result,)

    @impl(d2m.TileFillOp)
    def fill(self, interpreter, op, args):
        return ([args[0]] * 1024,)

    @impl(d2m.TileAddOp)
    def add(self, interpreter, op, args):
        return ([a + b for a, b in zip(*args, strict=True)],)

    @impl(d2m.TileReluOp)
    def relu(self, interpreter, op, args):
        return ([max(value, 0.0) for value in args[0]],)

    @impl(d2m.TileMatmulOp)
    def matmul(self, interpreter, op, args):
        a, b, accumulator = args
        return (
            [
                accumulator[i * 32 + j] + sum(a[i * 32 + k] * b[k * 32 + j] for k in range(32))
                for i in range(32)
                for j in range(32)
            ],
        )

    @impl(linalg.ops.GenericOp)
    def tile_loop(self, interpreter, op, args):
        maps = [attr.data for attr in op.indexing_maps]
        bounds = [0] * maps[0].num_dims
        output = args[-1].copy()
        tensors = (*args[:-1], output)
        for tensor, mapping in zip(tensors, maps, strict=True):
            for size, expr in zip(tensor.shape, mapping.results, strict=True):
                assert isinstance(expr, AffineDimExpr)
                assert bounds[expr.position] in (0, size)
                bounds[expr.position] = size
        for index in product(*(range(bound) for bound in bounds)):
            elements = tuple(
                tensor.load(mapping.eval(index, ()))
                for tensor, mapping in zip(tensors, maps, strict=True)
            )
            (value,) = interpreter.run_ssacfg_region(op.body, elements)
            output.store(maps[-1].eval(index, ()), value)
        return (output,)

    @impl_terminator(linalg.ops.YieldOp)
    def linalg_yield(self, interpreter, op, args):
        return ReturnedValues(args), ()


def lower(source):
    ctx = context()
    module = Parser(ctx, source).parse_module()
    ConvertToyToD2MPass().apply(ctx, module)
    module.verify()
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    text = stream.getvalue()
    assert "toy." not in text and "ttir." not in text
    assert "arith.mulf" not in text and "arith.addf" not in text and "arith.maximumf" not in text
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def execute(module, name, args):
    interpreter = Interpreter(module)
    for functions in (D2MFunctions(), ArithFunctions(), FuncFunctions()):
        interpreter.register_implementations(functions)
    return interpreter.call_op(name, args)


@pytest.mark.parametrize("m,k,n", [(32, 32, 32), (64, 96, 64)])
def test_linear_numerics_and_roundtrip(m, k, n):
    module = lower(f"""func.func @linear(%x: tensor<{m}x{k}xf32>, %w: tensor<{k}x{n}xf32>, %b: tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
      %0 = toy.matmul %x, %w : tensor<{m}x{n}xf32>
      %1 = toy.add %0, %b : tensor<{m}x{n}xf32>
      %2 = toy.relu %1 : tensor<{m}x{n}xf32>
      func.return %2 : tensor<{m}x{n}xf32>
    }}""")
    x = [float((i * 7 + i // k) % 11 - 5) for i in range(m * k)]
    w = [float((i * 3 + i // n) % 13 - 6) for i in range(k * n)]
    b = [float(i % 17 - 8) for i in range(m * n)]
    expected = [
        max(sum(x[i * k + t] * w[t * n + j] for t in range(k)) + b[i * n + j], 0)
        for i in range(m)
        for j in range(n)
    ]
    assert any(value > 0 for value in expected) and any(value == 0 for value in expected)
    (result,) = execute(module, "linear", (array(x, (m, k)), array(w, (k, n)), array(b, (m, n))))
    assert result.data == expected
    assert all(
        isinstance(op.body.block.args[0].type, TileType)
        for op in module.walk()
        if isinstance(op, linalg.ops.GenericOp)
    )


@pytest.mark.parametrize(
    "typ",
    [
        "tensor<32x32xf16>",
        "tensor<32x32xi32>",
        "tensor<?x32xf32>",
        "tensor<0x32xf32>",
        "tensor<16x32xf32>",
        "tensor<33x64xf32>",
        "tensor<f32>",
        "tensor<32xf32>",
        "tensor<1x32x32xf32>",
        'tensor<32x32xf32, "layout">',
    ],
)
def test_unsupported_tensors_leave_module_unchanged(typ):
    ctx = context()
    module = Parser(
        ctx,
        f"""func.func @bad(%good: tensor<32x32xf32>, %x: {typ}) -> {typ} {{
        %0 = toy.relu %good : tensor<32x32xf32>
        %1 = toy.relu %x : {typ}
        func.return %1 : {typ}
    }}""",
    ).parse_module()
    original = module.clone()
    with pytest.raises(PassFailedException, match="convert-toy-to-d2m:"):
        ConvertToyToD2MPass().apply(ctx, module)
    assert module.is_structurally_equivalent(original)


def test_example_and_idempotence():
    source = Path(__file__).resolve().parents[2] / "examples/toy_linear.mlir"
    module = lower(source.read_text())
    original = module.clone()
    ConvertToyToD2MPass().apply(context(), module)
    assert module.is_structurally_equivalent(original)
    assert len([op for op in module.walk() if isinstance(op, d2m.GenericOp)]) == 3
    for cls in (d2m.TileMatmulOp, d2m.TileAddOp, d2m.TileReluOp):
        assert len([op for op in module.walk() if isinstance(op, cls)]) == 1


@pytest.mark.parametrize("shape", [(32, 32), (32, 64), (64, 32)])
def test_elementwise_shared_input(shape):
    typ = f"tensor<{shape[0]}x{shape[1]}xf32>"
    module = lower(f"""func.func @twice(%x: {typ}) -> {typ} {{
        %0 = toy.add %x, %x : {typ}
        %1 = toy.relu %0 : {typ}
        func.return %1 : {typ}
    }}""")
    values = [float(i % 9 - 4) for i in range(prod(shape))]
    input_array = array(values, shape)
    (result,) = execute(module, "twice", (input_array,))
    assert result.data == [max(2 * value, 0) for value in values]
    assert input_array.data == values
