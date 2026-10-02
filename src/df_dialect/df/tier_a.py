"""Tier A ops: what to compute, with no cores or data movement.

Tier A ops work on tensors with value semantics. Managed targets (TTNN, Furiosa
TCL, the CPU reference) consume Tier A directly. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, TensorType
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from df_dialect.df.einsum import EinsumError, parse_einsum


@irdl_op_definition
class ContractOp(IRDLOperation):
    """Tensor contraction described by an einsum spec.

    Syntax:
        %c = df.contract "mk,kn->mn" %a, %b : tensor<64x64xbf16>

    The spec covers matmul ("mk,kn->mn"), batched matmul ("bmk,bkn->bmn") and
    the contractions inside attention. Both operands and the result must be
    ranked tensors with the same element type, and their shapes must agree
    with the spec.
    """

    name = "df.contract"

    spec = prop_def(StringAttr)
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    traits = traits_def(Pure())

    def __init__(self, spec: str, lhs: SSAValue, rhs: SSAValue, result_type: Attribute):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            properties={"spec": StringAttr(spec)},
        )

    @classmethod
    def parse(cls, parser: Parser) -> ContractOp:
        spec = parser.parse_str_literal("einsum spec")
        lhs = parser.parse_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        op = cls(spec, lhs, rhs, result_type)
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_string_literal(self.spec.data)
        printer.print_string(" ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)

    def verify_(self) -> None:
        lhs_t, rhs_t, res_t = self.lhs.type, self.rhs.type, self.result.type
        assert isinstance(lhs_t, TensorType)
        assert isinstance(rhs_t, TensorType)
        assert isinstance(res_t, TensorType)
        if not (lhs_t.element_type == rhs_t.element_type == res_t.element_type):
            raise VerifyException("operands and result of df.contract must share an element type")
        try:
            einsum = parse_einsum(self.spec.data)
            expected = einsum.result_shape(lhs_t.get_shape(), rhs_t.get_shape())
        except EinsumError as e:
            raise VerifyException(f"df.contract: {e}") from e
        if expected != res_t.get_shape():
            shape = "x".join(str(d) for d in expected)
            raise VerifyException(
                f"df.contract result should be tensor<{shape}x{res_t.element_type}>"
            )
