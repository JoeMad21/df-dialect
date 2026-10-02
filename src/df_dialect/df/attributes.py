"""Attributes and types of the df dialect.

Every class here is registered in `df_dialect/df/__init__.py`. If you add a new
attribute or type, add it to the `ATTRIBUTES` list there, document it in
`docs/SPEC.md`, and add a round-trip test under `tests/filecheck/dialect/`.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

DEFAULT_CHANNEL_DEPTH = 2

ALL_TILES = -1
"""Sentinel stored in `row_end` and `col_end` of TileRangeAttr for `#df.tiles<all>`."""


def _int(value: int | IntAttr) -> IntAttr:
    return value if isinstance(value, IntAttr) else IntAttr(value)


# --------------------------------------------------------------------------- grid


@irdl_attr_definition
class GridAttr(ParametrizedAttribute):
    """A rectangular grid of cores, written `#df.grid<ROWSxCOLS>`.

    Example: `#df.grid<2x2>` is a 2 by 2 grid of four cores.
    """

    name = "df.grid"

    rows: IntAttr
    cols: IntAttr

    def __init__(self, rows: int | IntAttr, cols: int | IntAttr):
        super().__init__(_int(rows), _int(cols))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            pos = parser.pos
            dims = parser.parse_dimension_list()
            if len(dims) != 2:
                parser.raise_error("expected a 2D grid such as <2x2>", pos, parser.pos)
        return (IntAttr(dims[0]), IntAttr(dims[1]))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.rows.data}x{self.cols.data}>")

    def verify(self) -> None:
        if self.rows.data < 1 or self.cols.data < 1:
            raise VerifyException(
                f"grid dimensions must be positive, got {self.rows.data}x{self.cols.data}"
            )

    @property
    def num_cores(self) -> int:
        return self.rows.data * self.cols.data


# --------------------------------------------------------------------------- tiles


@irdl_attr_definition
class TileRangeAttr(ParametrizedAttribute):
    """The cores an actor is placed on.

    `#df.tiles<all>` means every core of the enclosing grid.
    `#df.tiles<0:2, 1:2>` means rows 0 to 1 and column 1 (half-open ranges).
    """

    name = "df.tiles"

    row_begin: IntAttr
    row_end: IntAttr
    col_begin: IntAttr
    col_end: IntAttr

    def __init__(
        self,
        row_begin: int | IntAttr,
        row_end: int | IntAttr,
        col_begin: int | IntAttr,
        col_end: int | IntAttr,
    ):
        super().__init__(_int(row_begin), _int(row_end), _int(col_begin), _int(col_end))

    @staticmethod
    def all() -> TileRangeAttr:
        return TileRangeAttr(0, ALL_TILES, 0, ALL_TILES)

    def is_all(self) -> bool:
        return self.row_end.data == ALL_TILES and self.col_end.data == ALL_TILES

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("all") is not None:
                return (IntAttr(0), IntAttr(ALL_TILES), IntAttr(0), IntAttr(ALL_TILES))
            r0 = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation(":")
            r1 = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation(",")
            c0 = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation(":")
            c1 = parser.parse_integer(allow_boolean=False)
        return (IntAttr(r0), IntAttr(r1), IntAttr(c0), IntAttr(c1))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.range_str()}>")

    def range_str(self) -> str:
        if self.is_all():
            return "all"
        return (
            f"{self.row_begin.data}:{self.row_end.data}, {self.col_begin.data}:{self.col_end.data}"
        )

    def verify(self) -> None:
        if self.is_all():
            return
        if not (0 <= self.row_begin.data < self.row_end.data):
            raise VerifyException(f"invalid row range in #df.tiles<{self.range_str()}>")
        if not (0 <= self.col_begin.data < self.col_end.data):
            raise VerifyException(f"invalid column range in #df.tiles<{self.range_str()}>")

    def fits_in(self, grid: GridAttr) -> bool:
        if self.is_all():
            return True
        return self.row_end.data <= grid.rows.data and self.col_end.data <= grid.cols.data


# --------------------------------------------------------------------------- channel type


@irdl_attr_definition
class ChannelType(ParametrizedAttribute, TypeAttribute):
    """A typed FIFO between cores, written `!df.chan<ELEMENT>`.

    `depth` is how many elements the FIFO can hold. It defaults to 2 and is
    only printed when it differs: `!df.chan<tensor<64x64xbf16>, depth = 4>`.
    """

    name = "df.chan"

    element_type: Attribute
    depth: IntAttr

    def __init__(self, element_type: Attribute, depth: int | IntAttr = DEFAULT_CHANNEL_DEPTH):
        super().__init__(element_type, _int(depth))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            element_type = parser.parse_type()
            depth = DEFAULT_CHANNEL_DEPTH
            if parser.parse_optional_punctuation(",") is not None:
                parser.parse_keyword("depth")
                parser.parse_punctuation("=")
                depth = parser.parse_integer(allow_boolean=False, allow_negative=False)
        return (element_type, IntAttr(depth))

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.element_type)
            if self.depth.data != DEFAULT_CHANNEL_DEPTH:
                printer.print_string(f", depth = {self.depth.data}")

    def verify(self) -> None:
        if not isinstance(self.element_type, TypeAttribute):
            raise VerifyException("channel element must be a type")
        if self.depth.data < 1:
            raise VerifyException(f"channel depth must be at least 1, got {self.depth.data}")


# --------------------------------------------------------------------------- enums


class BcastKind(StrEnum):
    """Which cores receive each value sent on a channel."""

    ROW = "row"
    COL = "col"
    ALL = "all"


@irdl_attr_definition
class BcastAttr(EnumAttribute[BcastKind], SpacedOpaqueSyntaxAttribute):
    """Broadcast mode of a channel: `row`, `col` or `all`."""

    name = "df.bcast"


class SplitKind(StrEnum):
    """How a host array is cut into pieces for the grid."""

    ROWS = "rows"
    COLS = "cols"
    GRID = "grid"


@irdl_attr_definition
class SplitAttr(EnumAttribute[SplitKind], SpacedOpaqueSyntaxAttribute):
    """Split mode of df.scatter and df.gather: `rows`, `cols` or `grid`."""

    name = "df.split"
