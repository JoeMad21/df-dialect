"""Tier B ops: where work runs and how data moves.

A Tier B program declares a grid of cores, typed channels between them, the
actors that run on each core, and a host section that moves arrays in and out.
Explicit targets (Cerebras CSL, TT-Metalium kernels, AMD AIE) consume Tier B.
See docs/SPEC.md for the full syntax and docs/ARCHITECTURE.md for the design.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import MemRefType, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsolatedFromAbove, NoTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException

from df_dialect.df.attributes import (
    BcastAttr,
    BcastKind,
    ChannelType,
    GridAttr,
    SplitAttr,
    SplitKind,
    TileRangeAttr,
)


def _name(value: SSAValue) -> str:
    """Readable name of an SSA value for error messages."""
    return f"%{value.name_hint}" if value.name_hint else "<unnamed value>"


def _parse_symbol(parser: Parser) -> StringAttr:
    return parser.parse_symbol_name()


def _ensure_block(region: Region) -> Region:
    """An empty `{ }` parses as a region with no blocks; give it one empty block."""
    if not region.blocks:
        region.add_block(Block())
    return region


def _require_ancestor(op: Operation, kind: type[Operation]) -> None:
    """Raise unless `op` is nested (at any depth) inside an op of type `kind`."""
    parent = op.parent_op()
    while parent is not None:
        if isinstance(parent, kind):
            return
        parent = parent.parent_op()
    raise VerifyException(f"{op.name} must be nested inside {kind.name}")


# --------------------------------------------------------------------------- program


@irdl_op_definition
class ProgramOp(IRDLOperation):
    """A dataflow program: a grid of cores and everything that runs on it.

    Syntax:
        df.program @matmul attributes {grid = #df.grid<2x2>} {
          ...channels, actors, one df.host...
        }

    Verifier: every channel declared in the program must be written at least
    once (df.send or df.scatter) and read at least once (df.recv or df.gather).
    """

    name = "df.program"

    sym_name = prop_def(StringAttr)
    grid = attr_def(GridAttr)
    body = region_def("single_block")

    traits = traits_def(NoTerminator(), IsolatedFromAbove(), SymbolOpInterface())

    def __init__(self, sym_name: str, grid: GridAttr, body: Region | None = None):
        super().__init__(
            properties={"sym_name": StringAttr(sym_name)},
            attributes={"grid": grid},
            regions=[body if body is not None else Region(Block())],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ProgramOp:
        sym_name = _parse_symbol(parser)
        attrs = parser.parse_optional_attr_dict_with_keyword()
        body = _ensure_block(parser.parse_region())
        return cls.create(
            properties={"sym_name": sym_name},
            attributes=dict(attrs.data) if attrs else {},
            regions=[body],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print_string(" ")
        printer.print_region(self.body, print_entry_block_args=False, print_empty_block=False)

    def verify_(self) -> None:
        for op in self.body.ops:
            if not isinstance(op, ChannelOp):
                continue
            chan = op.result
            users = [use.operation for use in chan.uses]
            written = any(isinstance(u, SendOp | ScatterOp) and u.chan is chan for u in users)
            read = any(isinstance(u, RecvOp | GatherOp) and u.chan is chan for u in users)
            if not written:
                raise VerifyException(f"channel {_name(chan)} is read but never sent to")
            if not read:
                raise VerifyException(f"channel {_name(chan)} is sent to but never read")


# --------------------------------------------------------------------------- channel


@irdl_op_definition
class ChannelOp(IRDLOperation):
    """Declares a typed FIFO between cores.

    Syntax:
        %c = df.channel : !df.chan<tensor<64x64xbf16>>
        %a = df.channel bcast(row) : !df.chan<tensor<64x128xbf16>>

    `bcast(row)` delivers each value to every core in a grid row, `bcast(col)`
    to every core in a grid column, `bcast(all)` to every core.
    """

    name = "df.channel"

    result = result_def(ChannelType)
    bcast = opt_prop_def(BcastAttr)

    traits = traits_def(HasParent(ProgramOp))

    def __init__(self, chan_type: ChannelType, bcast: BcastKind | None = None):
        props: dict[str, Attribute] = {}
        if bcast is not None:
            props["bcast"] = BcastAttr(bcast)
        super().__init__(result_types=[chan_type], properties=props)

    @classmethod
    def parse(cls, parser: Parser) -> ChannelOp:
        bcast = None
        if parser.parse_optional_keyword("bcast") is not None:
            with parser.in_parens():
                bcast = parser.parse_str_enum(BcastKind)
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        pos = parser.pos
        chan_type = parser.parse_type()
        if not isinstance(chan_type, ChannelType):
            parser.raise_error("df.channel must produce a !df.chan type", pos, parser.pos)
        op = cls(chan_type, bcast)
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer) -> None:
        if self.bcast is not None:
            printer.print_string(f" bcast({self.bcast.data.value})")
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)


# --------------------------------------------------------------------------- actor


@irdl_op_definition
class ActorOp(IRDLOperation):
    """The code each core runs.

    Syntax:
        df.actor @tile on #df.tiles<all> ins(%a, %b) outs(%c) {
          ...df.recv, compute, df.send...
        }

    `on` places one copy of the actor on every listed core. `ins` and `outs`
    list the channels the body may read from and write to.

    Verifier: the tile range must fit inside the program grid, and the body
    may only df.recv from `ins` and df.send to `outs`.
    """

    name = "df.actor"

    sym_name = prop_def(StringAttr)
    tiles = prop_def(TileRangeAttr)
    ins = var_operand_def(ChannelType)
    outs = var_operand_def(ChannelType)
    body = region_def("single_block")

    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    traits = traits_def(NoTerminator(), HasParent(ProgramOp))

    def __init__(
        self,
        sym_name: str | StringAttr,
        tiles: TileRangeAttr,
        ins: Sequence[SSAValue],
        outs: Sequence[SSAValue],
        body: Region | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        super().__init__(
            operands=[ins, outs],
            properties={"sym_name": sym_name, "tiles": tiles},
            regions=[body if body is not None else Region(Block())],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ActorOp:
        sym_name = _parse_symbol(parser)
        parser.parse_keyword("on")
        pos = parser.pos
        tiles = parser.parse_attribute()
        if not isinstance(tiles, TileRangeAttr):
            parser.raise_error("expected #df.tiles<...> after 'on'", pos, parser.pos)
        parser.parse_keyword("ins")
        ins = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parser.parse_operand)
        parser.parse_keyword("outs")
        outs = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parser.parse_operand)
        attrs = parser.parse_optional_attr_dict_with_keyword()
        body = _ensure_block(parser.parse_region())
        op = cls(sym_name, tiles, ins, outs, body)
        if attrs:
            op.attributes.update(attrs.data)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        printer.print_string(" on ")
        printer.print_attribute(self.tiles)
        printer.print_string(" ins(")
        printer.print_list(self.ins, printer.print_operand)
        printer.print_string(") outs(")
        printer.print_list(self.outs, printer.print_operand)
        printer.print_string(")")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print_string(" ")
        printer.print_region(self.body, print_entry_block_args=False, print_empty_block=False)

    def verify_(self) -> None:
        program = self.parent_op()
        if isinstance(program, ProgramOp) and not self.tiles.fits_in(program.grid):
            g = program.grid
            raise VerifyException(
                f"tiles {self.tiles.range_str()} exceed #df.grid<{g.rows.data}x{g.cols.data}>"
            )
        ins, outs = set(self.ins), set(self.outs)
        for op in self.body.walk():
            if isinstance(op, RecvOp) and op.chan not in ins:
                raise VerifyException(
                    f"df.recv reads {_name(op.chan)}, which is not in ins(...) of @{self.sym_name.data}"
                )
            if isinstance(op, SendOp) and op.chan not in outs:
                raise VerifyException(
                    f"df.send writes {_name(op.chan)}, which is not in outs(...) of @{self.sym_name.data}"
                )


# --------------------------------------------------------------------------- host


@irdl_op_definition
class HostOp(IRDLOperation):
    """The host side of a program: moves arrays into and out of the grid.

    Syntax:
        df.host (%A: memref<128x128xbf16>, %C: memref<128x128xbf16>) {
          df.scatter %A -> %arow split(rows)
          df.gather %cout -> %C split(grid)
        }
    """

    name = "df.host"

    body = region_def("single_block")

    traits = traits_def(NoTerminator(), HasParent(ProgramOp))

    def __init__(self, body: Region):
        super().__init__(regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> HostOp:
        args = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parser.parse_argument)
        attrs = parser.parse_optional_attr_dict_with_keyword()
        body = _ensure_block(parser.parse_region(args))
        op = cls(body)
        if attrs:
            op.attributes.update(attrs.data)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_list(self.body.block.args, printer.print_block_argument)
        printer.print_string(")")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print_string(" ")
        printer.print_region(self.body, print_entry_block_args=False, print_empty_block=False)


# --------------------------------------------------------------------------- recv / send


def _check_element(op: Operation, chan: SSAValue, value_type: Attribute) -> None:
    chan_type = chan.type
    assert isinstance(chan_type, ChannelType)
    if chan_type.element_type != value_type:
        raise VerifyException(f"channel {_name(chan)} carries {chan_type.element_type}")


@irdl_op_definition
class RecvOp(IRDLOperation):
    """Waits for the next value on a channel and returns it.

    Syntax:
        %a = df.recv %arow : tensor<64x128xbf16>

    Verifier: the result type must equal the channel element type.
    """

    name = "df.recv"

    chan = operand_def(ChannelType)
    result = result_def()

    def __init__(self, chan: SSAValue, result_type: Attribute):
        super().__init__(operands=[chan], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> RecvOp:
        chan = parser.parse_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        op = cls(chan, parser.parse_type())
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.chan)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)

    def verify_(self) -> None:
        _require_ancestor(self, ActorOp)
        _check_element(self, self.chan, self.result.type)


@irdl_op_definition
class SendOp(IRDLOperation):
    """Sends a value on a channel.

    Syntax:
        df.send %cout, %c : tensor<64x64xbf16>

    Verifier: the value type must equal the channel element type.
    """

    name = "df.send"

    chan = operand_def(ChannelType)
    value = operand_def()

    def __init__(self, chan: SSAValue, value: SSAValue):
        super().__init__(operands=[chan, value])

    @classmethod
    def parse(cls, parser: Parser) -> SendOp:
        chan = parser.parse_operand()
        parser.parse_punctuation(",")
        value = parser.parse_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        pos = parser.pos
        written_type = parser.parse_type()
        if written_type != value.type:
            parser.raise_error(f"type does not match {_name(value)}: {value.type}", pos, parser.pos)
        op = cls(chan, value)
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.chan)
        printer.print_string(", ")
        printer.print_operand(self.value)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)

    def verify_(self) -> None:
        _require_ancestor(self, ActorOp)
        _check_element(self, self.chan, self.value.type)


# --------------------------------------------------------------------------- scatter / gather


def _parse_split(parser: Parser) -> SplitKind:
    parser.parse_keyword("split")
    with parser.in_parens():
        return parser.parse_str_enum(SplitKind)


@irdl_op_definition
class ScatterOp(IRDLOperation):
    """Cuts a host array into pieces and sends them into a channel.

    Syntax:
        df.scatter %A -> %arow split(rows)

    `split(rows)` sends row block i to grid row i, `split(cols)` sends column
    block j to grid column j, `split(grid)` sends one block per core.
    """

    name = "df.scatter"

    source = operand_def(MemRefType)
    chan = operand_def(ChannelType)
    split = prop_def(SplitAttr)

    traits = traits_def(HasParent(HostOp))

    def __init__(self, source: SSAValue, chan: SSAValue, split: SplitKind):
        super().__init__(operands=[source, chan], properties={"split": SplitAttr(split)})

    @classmethod
    def parse(cls, parser: Parser) -> ScatterOp:
        source = parser.parse_operand()
        parser.parse_punctuation("->")
        chan = parser.parse_operand()
        split = _parse_split(parser)
        op = cls(source, chan, split)
        op.attributes.update(parser.parse_optional_attr_dict())
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.source)
        printer.print_string(" -> ")
        printer.print_operand(self.chan)
        printer.print_string(f" split({self.split.data.value})")
        printer.print_op_attributes(self.attributes)


@irdl_op_definition
class GatherOp(IRDLOperation):
    """Receives pieces from a channel and assembles them into a host array.

    Syntax:
        df.gather %cout -> %C split(grid)
    """

    name = "df.gather"

    chan = operand_def(ChannelType)
    dest = operand_def(MemRefType)
    split = prop_def(SplitAttr)

    traits = traits_def(HasParent(HostOp))

    def __init__(self, chan: SSAValue, dest: SSAValue, split: SplitKind):
        super().__init__(operands=[chan, dest], properties={"split": SplitAttr(split)})

    @classmethod
    def parse(cls, parser: Parser) -> GatherOp:
        chan = parser.parse_operand()
        parser.parse_punctuation("->")
        dest = parser.parse_operand()
        split = _parse_split(parser)
        op = cls(chan, dest, split)
        op.attributes.update(parser.parse_optional_attr_dict())
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.chan)
        printer.print_string(" -> ")
        printer.print_operand(self.dest)
        printer.print_string(f" split({self.split.data.value})")
        printer.print_op_attributes(self.attributes)
