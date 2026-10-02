"""Passes for the df dialect.

To add a pass: create a module in this folder, then add one line to ALL_PASSES.
The key must equal the pass `name`. See docs/playbooks/ADD_PASS.md.
"""

from collections.abc import Callable

from xdsl.passes import ModulePass


def _check_memory() -> type[ModulePass]:
    from df_dialect.passes.check_memory import CheckMemoryPass

    return CheckMemoryPass


ALL_PASSES: dict[str, Callable[[], type[ModulePass]]] = {
    "df-check-memory": _check_memory,
}
