"""Registers the df dialect and passes with xDSL.

pyproject.toml points the `xdsl.universe` entry point here, so once the package
is installed, `xdsl-opt` and `df-opt` both know about `df` automatically.
"""

from xdsl.ir import Dialect
from xdsl.universe import Universe

from df_dialect.passes import ALL_PASSES


def _get_df() -> Dialect:
    from df_dialect.df import Df

    return Df


def _get_toy() -> Dialect:
    from df_dialect.toy import Toy

    return Toy


def _get_ttir() -> Dialect:
    from df_dialect.ttir import TTIR

    return TTIR


DF_UNIVERSE = Universe(
    all_dialects={"df": _get_df, "toy": _get_toy, "ttir": _get_ttir},
    all_passes=dict(ALL_PASSES),
)
