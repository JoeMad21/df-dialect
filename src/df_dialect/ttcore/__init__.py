from xdsl.dialects.builtin import Float32Type, IntAttr, f32
from xdsl.dialects.utils import EnumAttribute
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


@irdl_attr_definition
class TileType(ParametrizedAttribute, TypeAttribute):
    name = "ttcore.tile"
    rows: IntAttr
    cols: IntAttr
    element_type: Float32Type

    def __init__(self):
        super().__init__(IntAttr(32), IntAttr(32), f32)

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            dims = parser.parse_dimension_list()
            if len(dims) != 2:
                parser.raise_error("expected a two-dimensional tile")
            parser.parse_punctuation(",")
            element_type = parser.parse_type()
        return (IntAttr(dims[0]), IntAttr(dims[1]), element_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.rows.data}x{self.cols.data}, ")
        printer.print_attribute(self.element_type)
        printer.print_string(">")

    def verify(self) -> None:
        if (self.rows.data, self.cols.data) != (32, 32):
            raise VerifyException("only 32x32 tiles are supported")


@irdl_attr_definition
class GridAttr(ParametrizedAttribute):
    name = "ttcore.grid"
    rows: IntAttr
    cols: IntAttr

    def __init__(self, rows: int, cols: int):
        super().__init__(IntAttr(rows), IntAttr(cols))

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            dims = parser.parse_dimension_list()
            if len(dims) != 2:
                parser.raise_error("expected a two-dimensional grid")
        return (IntAttr(dims[0]), IntAttr(dims[1]))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.rows.data}x{self.cols.data}>")


class IteratorType(StrEnum):
    PARALLEL = "parallel"


@irdl_attr_definition
class IteratorTypeAttr(EnumAttribute[IteratorType]):
    name = "ttcore.iterator_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> IteratorType:
        with parser.in_angle_brackets():
            return IteratorType(parser.parse_identifier())

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"<{self.data.value}>")


TTCore = Dialect("ttcore", [], [GridAttr, IteratorTypeAttr, TileType])
