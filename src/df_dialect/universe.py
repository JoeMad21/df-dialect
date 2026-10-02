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


DF_UNIVERSE = Universe(
    all_dialects={"df": _get_df},
    all_passes=dict(ALL_PASSES),
)
