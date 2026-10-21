import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "split_csl", Path(__file__).resolve().parents[2] / "scripts" / "split_csl.py"
)
assert _spec and _spec.loader
split_csl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_csl)

SAMPLE = """// FILE: linear_program
fn linear() void {
  return;
}
// -----
// FILE: linear_layout
layout {
  @set_tile_code(x, y, "linear.csl", v8);
}
"""


def test_split_names_files_for_cslc():
    files = split_csl.split(SAMPLE)
    assert set(files) == {"linear.csl", "linear_layout.csl"}
    assert files["linear.csl"].startswith("fn linear() void {")
    assert "// -----" not in files["linear.csl"]
    assert "@set_tile_code" in files["linear_layout.csl"]


def test_split_rejects_non_csl_input():
    with pytest.raises(ValueError, match="FILE"):
        split_csl.split("builtin.module {}\n")
