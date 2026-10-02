"""Build the 2x2 matmul from Python instead of parsing text.

This is the pattern a lowering pass uses to create df ops. If you change an op
constructor, update this test.
"""

from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import BFloat16Type, MemRefType, ModuleOp, TensorType
from xdsl.ir import Block, Region

from df_dialect.df import (
    ActorOp,
    BcastKind,
    ChannelOp,
    ChannelType,
    ContractOp,
    GatherOp,
    GridAttr,
    HostOp,
    ProgramOp,
    RecvOp,
    ScatterOp,
    SendOp,
    SplitKind,
    TileRangeAttr,
)
from df_dialect.passes.check_memory import actor_footprint


def build_matmul() -> tuple[ModuleOp, ActorOp]:
    bf16 = BFloat16Type()
    a_t, b_t, c_t = (TensorType(bf16, s) for s in ([64, 128], [128, 64], [64, 64]))
    full = MemRefType(bf16, [128, 128])

    program = ProgramOp("matmul", GridAttr(2, 2))
    with ImplicitBuilder(program.body):
        arow = ChannelOp(ChannelType(a_t), BcastKind.ROW).result
        bcol = ChannelOp(ChannelType(b_t), BcastKind.COL).result
        cout = ChannelOp(ChannelType(c_t)).result

        actor = ActorOp("tile", TileRangeAttr.all(), [arow, bcol], [cout])
        with ImplicitBuilder(actor.body):
            a = RecvOp(arow, a_t).result
            b = RecvOp(bcol, b_t).result
            c = ContractOp("mk,kn->mn", a, b, c_t).result
            SendOp(cout, c)

        host_block = Block(arg_types=[full, full, full])
        HostOp(Region(host_block))
        big_a, big_b, big_c = host_block.args
        with ImplicitBuilder(host_block):
            ScatterOp(big_a, arow, SplitKind.ROWS)
            ScatterOp(big_b, bcol, SplitKind.COLS)
            GatherOp(cout, big_c, SplitKind.GRID)

    return ModuleOp([program]), actor


def test_built_program_verifies():
    module, _ = build_matmul()
    module.verify()


def test_footprint_matches_slide():
    _, actor = build_matmul()
    assert actor_footprint(actor) == 40 * 1024
