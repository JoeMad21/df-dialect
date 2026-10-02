"""A minimal mirror of Tenstorrent's TTIR dialect (tt-mlir), output only.

Only the ops the toy lowering emits are defined. They have no custom syntax,
so xDSL prints them in MLIR generic form, which is how tt-mlir's own tests
write them, for example:

    %0 = "ttir.matmul"(%a, %b) : (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>

Operand form matches tt-mlir commit a6bd5d2 (2026-09-16): no output operand.
The real definitions live in tt-mlir: include/ttmlir/Dialect/TTIR/IR/TTIROps.td.
"""

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def


@irdl_op_definition
class TTIRMatmulOp(IRDLOperation):
    """ttir.matmul: matrix multiply."""

    name = "ttir.matmul"

    a = operand_def(TensorType)
    b = operand_def(TensorType)
    result = result_def(TensorType)

    def __init__(self, a: SSAValue, b: SSAValue, result_type: Attribute):
        super().__init__(operands=[a, b], result_types=[result_type])


@irdl_op_definition
class TTIRAddOp(IRDLOperation):
    """ttir.add: elementwise addition."""

    name = "ttir.add"

    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute):
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


@irdl_op_definition
class TTIRReluOp(IRDLOperation):
    """ttir.relu: elementwise max(x, 0)."""

    name = "ttir.relu"

    input = operand_def(TensorType)
    result = result_def(TensorType)

    def __init__(self, input: SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])


TTIR = Dialect("ttir", [TTIRMatmulOp, TTIRAddOp, TTIRReluOp], [])
