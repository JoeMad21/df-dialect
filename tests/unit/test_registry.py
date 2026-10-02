"""Coverage gate: every op and attribute must be documented and tested.

If this fails after you add an op, you are missing one of:
  1. a docstring on the class,
  2. a section in docs/SPEC.md that mentions the op name (for example `df.my_op`),
  3. at least one FileCheck test under tests/filecheck/ that uses it.
"""

from pathlib import Path

import pytest

from df_dialect.df import ATTRIBUTES, OPS, Df

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
FILECHECK = "\n".join(
    p.read_text(encoding="utf-8") for p in (ROOT / "tests" / "filecheck").rglob("*.mlir")
)


def _spelling(cls) -> str:
    """How the construct appears in IR text, for example df.recv or #df.grid."""
    from xdsl.ir import TypeAttribute

    if cls in OPS:
        return cls.name
    if issubclass(cls, TypeAttribute):
        return "!" + cls.name
    return "#" + cls.name


ALL = [*OPS, *ATTRIBUTES]


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_has_docstring(cls):
    assert cls.__doc__ and cls.__doc__.strip(), f"{cls.__name__} needs a docstring"


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_documented_in_spec(cls):
    assert cls.name in SPEC, f"add a section for {cls.name} to docs/SPEC.md"


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_used_in_filecheck_tests(cls):
    spelling = _spelling(cls)
    if cls.name in ("df.bcast", "df.split"):
        # Enum attributes appear inside op syntax as bcast(...) and split(...).
        spelling = cls.name.split(".")[1] + "("
    assert spelling in FILECHECK, f"add a FileCheck test that uses {spelling}"


def test_names_are_in_df_namespace():
    for cls in ALL:
        assert cls.name.startswith("df."), cls.name


def test_dialect_object_is_complete():
    assert set(Df.operations) == set(OPS)
