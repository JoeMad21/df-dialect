from collections.abc import Callable

from xdsl.dialects import arith
from xdsl.dialects.builtin import AffineMapAttr, FloatAttr, NoneAttr, TensorType, f32
from xdsl.dialects.linalg import ops as linalg
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import IRDLOperation
from xdsl.utils.exceptions import PassFailedException

from df_dialect import d2m
from df_dialect.ttcore import TileType


def verify_tile_tensor(typ: Attribute) -> None:
    if not isinstance(typ, TensorType) or typ.element_type != f32:
        raise PassFailedException(f"convert-toy-to-d2m: only f32 tensors are supported, got {typ}")
    if (
        len(typ.get_shape()) != 2
        or any(dim <= 0 or dim % 32 for dim in typ.get_shape())
        or typ.encoding != NoneAttr()
    ):
        raise PassFailedException(
            f"convert-toy-to-d2m: expected unencoded rank-2 tensors with positive static dimensions divisible by 32, got {typ}"
        )


def _tile_loop(
    inputs: list[SSAValue], output: SSAValue, body: Block, *, matmul: bool = False
) -> linalg.GenericOp:
    maps = (
        [
            AffineMap.from_callable(f)
            for f in (lambda i, j, k: (i, k), lambda i, j, k: (k, j), lambda i, j, k: (i, j))
        ]
        if matmul
        else [AffineMap.identity(2)] * (len(inputs) + 1)
    )
    iterators = [IteratorType.PARALLEL] * 2 + ([IteratorType.REDUCTION] if matmul else [])
    return linalg.GenericOp(
        inputs,
        [output],
        Region(body),
        [AffineMapAttr(m) for m in maps],
        [IteratorTypeAttr(it) for it in iterators],
        [output.type],
    )


def build_tile_compute(
    operands: list[SSAValue],
    result_type: TensorType,
    compute_tile: Callable[..., IRDLOperation],
) -> tuple[d2m.EmptyOp, d2m.GenericOp]:
    matmul = compute_tile is d2m.TileMatmulOp
    empty = d2m.EmptyOp(result_type)
    block = Block(arg_types=[d2m.CBType(value.type) for value in (*operands, empty.result)])
    inputs = []
    for arg in block.args[:-1]:
        wait = d2m.WaitOp(arg)
        tiles = d2m.EmptyOp(TensorType(TileType(), [d // 32 for d in wait.result.type.get_shape()]))
        tilize = d2m.TileTilizeBlockOp(wait.result, tiles.result)
        block.add_ops([wait, tiles, tilize])
        inputs.append(tilize.result)
    destination = d2m.ReserveOp(block.args[-1])
    tiles = d2m.EmptyOp(TensorType(TileType(), [d // 32 for d in result_type.get_shape()]))
    block.add_ops([destination, tiles])
    output = tiles.result
    if matmul:
        fill_body = Block(arg_types=[TileType()])
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        fill_tile = d2m.TileFillOp(zero.result)
        fill_body.add_ops([zero, fill_tile, linalg.YieldOp(fill_tile.result)])
        fill = _tile_loop([], output, fill_body)
        block.add_op(fill)
        output = fill.results[0]

    body = Block(arg_types=[TileType()] * (len(inputs) + 1))
    tile = compute_tile(*(body.args if matmul else body.args[:-1]))
    body.add_ops([tile, linalg.YieldOp(tile.results[0])])
    compute = _tile_loop(inputs, output, body, matmul=matmul)
    untilize = d2m.TileUntilizeBlockOp(compute.results[0], destination.result)
    block.add_ops([compute, untilize, d2m.YieldOp(untilize.result)])
    generic = d2m.GenericOp(operands, empty.result, Region(block))
    return empty, generic
