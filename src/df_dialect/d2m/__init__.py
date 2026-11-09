from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    Float32Type,
    IntegerAttr,
    TensorType,
    i64,
)
from xdsl.dialects.utils import EnumAttribute
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, Region, SSAValue, TypeAttribute
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

from df_dialect.ttcore import GridAttr, IteratorType, IteratorTypeAttr, TileType


class ThreadType(StrEnum):
    UNIFIED = "unified"


@irdl_attr_definition
class ThreadAttr(EnumAttribute[ThreadType]):
    name = "d2m.thread"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> ThreadType:
        with parser.in_angle_brackets():
            return ThreadType(parser.parse_identifier())

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"<{self.data.value}>")


@irdl_attr_definition
class CBType(ParametrizedAttribute, TypeAttribute):
    name = "d2m.cb"
    underlying: TensorType

    def __init__(self, underlying: Attribute):
        assert isinstance(underlying, TensorType)
        super().__init__(underlying)


@irdl_op_definition
class EmptyOp(IRDLOperation):
    name = "d2m.empty"
    result = result_def(TensorType)

    def __init__(self, result_type: TensorType):
        super().__init__(result_types=[result_type])


class _CBOp(IRDLOperation):
    name = "d2m.wait"
    cb = operand_def(CBType)
    result = result_def(TensorType)

    def __init__(self, cb: SSAValue):
        assert isinstance(cb.type, CBType)
        super().__init__(operands=[cb], result_types=[cb.type.underlying])

    def verify_(self) -> None:
        assert isinstance(self.cb.type, CBType)
        if self.result.type != self.cb.type.underlying:
            raise VerifyException("result type must match circular buffer's wrapped tensor")


@irdl_op_definition
class WaitOp(_CBOp):
    name = "d2m.wait"


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "d2m.yield"
    value = operand_def(TensorType)
    traits = traits_def(IsTerminator())

    def __init__(self, value: SSAValue):
        super().__init__(operands=[value])


@irdl_op_definition
class ReserveOp(_CBOp):
    name = "d2m.reserve"


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "d2m.generic"
    inputs = var_operand_def(TensorType)
    output = operand_def(TensorType)
    result = result_def(TensorType)
    body = region_def("single_block")
    grid = attr_def(GridAttr)
    block_factors = attr_def(ArrayAttr[IntegerAttr])
    indexing_maps = attr_def(ArrayAttr[AffineMapAttr])
    iterator_types = attr_def(ArrayAttr[IteratorTypeAttr])
    threads = attr_def(ArrayAttr[ThreadAttr])
    irdl_options = (AttrSizedOperandSegments(),)

    def __init__(self, inputs: list[SSAValue], output: SSAValue, body: Region):
        super().__init__(
            operands=[inputs, output],
            result_types=[output.type],
            regions=[body],
            attributes={
                "grid": GridAttr(1, 1),
                "block_factors": ArrayAttr([IntegerAttr(1, i64)] * 2),
                "indexing_maps": ArrayAttr(
                    [AffineMapAttr(AffineMap.identity(2))] * (len(inputs) + 1)
                ),
                "iterator_types": ArrayAttr([IteratorTypeAttr(IteratorType.PARALLEL)] * 2),
                "threads": ArrayAttr([ThreadAttr(ThreadType.UNIFIED)]),
            },
        )

    def verify_(self) -> None:
        if self.result.type != self.output.type:
            raise VerifyException("generic result must match destination type")
        if tuple(arg.type for arg in self.body.block.args) != tuple(
            CBType(value.type) for value in self.operands
        ):
            raise VerifyException("generic arguments must wrap the operand tensor types")
        terminator = self.body.block.last_op
        if not isinstance(terminator, YieldOp) or terminator.value.type != self.result.type:
            raise VerifyException("generic must yield its result tensor type")


class _TileConvertOp(IRDLOperation):
    input = operand_def(TensorType)
    output = operand_def(TensorType)
    result = result_def(TensorType)

    def __init__(self, input: SSAValue, output: SSAValue):
        super().__init__(operands=[input, output], result_types=[output.type])

    def verify_(self) -> None:
        if self.result.type != self.output.type:
            raise VerifyException("result type must match output parameter type")
        scalar, tiled = self.input.type, self.output.type
        if self.name == "d2m.tile_untilize_block":
            scalar, tiled = tiled, scalar
        assert isinstance(scalar, TensorType) and isinstance(tiled, TensorType)
        if not isinstance(scalar.element_type, Float32Type) or not isinstance(
            tiled.element_type, TileType
        ):
            raise VerifyException("expected f32 tensor and tensor of ttcore tiles")
        if (
            len(scalar.get_shape()) != 2
            or tuple(d * 32 for d in tiled.get_shape()) != scalar.get_shape()
        ):
            raise VerifyException("scalar shape must equal tiled shape multiplied by 32")


@irdl_op_definition
class TileTilizeBlockOp(_TileConvertOp):
    name = "d2m.tile_tilize_block"


@irdl_op_definition
class TileUntilizeBlockOp(_TileConvertOp):
    name = "d2m.tile_untilize_block"


@irdl_op_definition
class TileFillOp(IRDLOperation):
    name = "d2m.tile_fill"
    value = operand_def(Float32Type)
    result = result_def(TileType)

    def __init__(self, value: SSAValue):
        super().__init__(operands=[value], result_types=[TileType()])


@irdl_op_definition
class TileMatmulOp(IRDLOperation):
    name = "d2m.tile_matmul"
    a = operand_def(TileType)
    b = operand_def(TileType)
    c = operand_def(TileType)
    result = result_def(TileType)

    def __init__(self, a: SSAValue, b: SSAValue, c: SSAValue):
        super().__init__(operands=[a, b, c], result_types=[c.type])


@irdl_op_definition
class TileAddOp(IRDLOperation):
    name = "d2m.tile_add"
    lhs = operand_def(TileType)
    rhs = operand_def(TileType)
    result = result_def(TileType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class TileReluOp(IRDLOperation):
    name = "d2m.tile_relu"
    input = operand_def(TileType)
    result = result_def(TileType)

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input], result_types=[input.type])


D2M = Dialect(
    "d2m",
    [
        EmptyOp,
        WaitOp,
        ReserveOp,
        YieldOp,
        GenericOp,
        TileTilizeBlockOp,
        TileUntilizeBlockOp,
        TileFillOp,
        TileMatmulOp,
        TileAddOp,
        TileReluOp,
    ],
    [CBType, ThreadAttr],
)
