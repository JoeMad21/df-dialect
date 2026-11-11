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


def _get_d2m() -> Dialect:
    from df_dialect.d2m import D2M

    return D2M


def _get_ttcore() -> Dialect:
    from df_dialect.ttcore import TTCore

    return TTCore


DF_UNIVERSE = Universe(
    all_dialects={
        "df": _get_df,
        "toy": _get_toy,
        "ttir": _get_ttir,
        "d2m": _get_d2m,
        "ttcore": _get_ttcore,
    },
    all_passes=dict(ALL_PASSES),
)
