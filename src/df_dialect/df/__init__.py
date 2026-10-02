"""The df dialect.

To add an op or attribute: define it in the right file, then add it to OPS or
ATTRIBUTES below. Nothing else needs registering. See docs/playbooks/ADD_OP.md.
"""

from xdsl.ir import Dialect

from df_dialect.df.attributes import (
    BcastAttr,
    BcastKind,
    ChannelType,
    GridAttr,
    SplitAttr,
    SplitKind,
    TileRangeAttr,
)
from df_dialect.df.tier_a import ContractOp
from df_dialect.df.tier_b import (
    ActorOp,
    ChannelOp,
    GatherOp,
    HostOp,
    ProgramOp,
    RecvOp,
    ScatterOp,
    SendOp,
)

OPS = [
    # Tier A
    ContractOp,
    # Tier B
    ProgramOp,
    ChannelOp,
    ActorOp,
    HostOp,
    RecvOp,
    SendOp,
    ScatterOp,
    GatherOp,
]

ATTRIBUTES = [
    GridAttr,
    TileRangeAttr,
    ChannelType,
    BcastAttr,
    SplitAttr,
]

Df = Dialect("df", OPS, ATTRIBUTES)

__all__ = [
    "ATTRIBUTES",
    "OPS",
    "ActorOp",
    "BcastAttr",
    "BcastKind",
    "ChannelOp",
    "ChannelType",
    "ContractOp",
    "Df",
    "GatherOp",
    "GridAttr",
    "HostOp",
    "ProgramOp",
    "RecvOp",
    "ScatterOp",
    "SendOp",
    "SplitAttr",
    "SplitKind",
    "TileRangeAttr",
]
