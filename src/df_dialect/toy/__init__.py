"""The toy dialect: three tensor ops used to demo lowering into Tenstorrent TTIR.

This is a sandbox, separate from the df dialect design in docs/SPEC.md.
Lower it with:  df-opt input.mlir -p convert-toy-to-ttir
See docs/TOY.md.
"""

from typing import ClassVar

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """2D matrix multiply.

    Syntax:
        %c = toy.matmul %a, %b : tensor<64x64xf32>

    Verifier: both operands are rank 2, the inner dimensions match, the result
    is MxN, and all element types are equal.
    """

    name = "toy.matmul"

    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    traits = traits_def(Pure())

    def __init__(self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute):
        super().__init__(operands=[lhs, rhs], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> "MatmulOp":
        lhs = parser.parse_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        op = cls(lhs, rhs, parser.parse_type())
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)

    def verify_(self) -> None:
        a, b, c = self.lhs.type, self.rhs.type, self.result.type
        assert isinstance(a, TensorType)
        assert isinstance(b, TensorType)
        assert isinstance(c, TensorType)
        if len(a.get_shape()) != 2 or len(b.get_shape()) != 2:
            raise VerifyException("toy.matmul operands must be rank 2")
        (m, k), (k2, n) = a.get_shape(), b.get_shape()
        if k != k2:
            raise VerifyException(f"toy.matmul inner dimensions differ: {k} vs {k2}")
        if not (a.element_type == b.element_type == c.element_type):
            raise VerifyException("toy.matmul operands and result must share an element type")
        if c.get_shape() != (m, n):
            raise VerifyException(f"toy.matmul result should be tensor<{m}x{n}x{c.element_type}>")


@irdl_op_definition
class AddOp(IRDLOperation):
    """Elementwise addition of two tensors of the same type.

    Syntax:
        %c = toy.add %a, %b : tensor<64x64xf32>
    """

    name = "toy.add"

    T: ClassVar = VarConstraint("T", base(TensorType))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    traits = traits_def(Pure())
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class ReluOp(IRDLOperation):
    """Elementwise max(x, 0).

    Syntax:
        %y = toy.relu %x : tensor<64x64xf32>
    """

    name = "toy.relu"

    T: ClassVar = VarConstraint("T", base(TensorType))
    input = operand_def(T)
    result = result_def(T)

    traits = traits_def(Pure())
    assembly_format = "$input attr-dict `:` type($result)"

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input], result_types=[input.type])


Toy = Dialect("toy", [MatmulOp, AddOp, ReluOp], [])
