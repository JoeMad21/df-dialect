"""Parsing and shape checking for einsum specs such as "mk,kn->mn".

This module is plain Python with no xDSL imports, so it is easy to unit test.
See `tests/unit/test_einsum.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


class EinsumError(ValueError):
    """Raised when a spec is malformed or does not match the operand shapes."""


@dataclass(frozen=True)
class EinsumSpec:
    inputs: tuple[str, ...]
    output: str

    def result_shape(self, *shapes: tuple[int, ...]) -> tuple[int, ...]:
        """Return the result shape for the given operand shapes, or raise EinsumError."""
        if len(shapes) != len(self.inputs):
            raise EinsumError(f"spec has {len(self.inputs)} inputs but got {len(shapes)} operands")
        sizes: dict[str, int] = {}
        for index, (labels, shape) in enumerate(zip(self.inputs, shapes, strict=True)):
            if len(labels) != len(shape):
                raise EinsumError(
                    f"operand {index} has rank {len(shape)} but spec '{labels}' has rank {len(labels)}"
                )
            for label, size in zip(labels, shape, strict=True):
                if sizes.setdefault(label, size) != size:
                    raise EinsumError(
                        f"index '{label}' is {sizes[label]} in one operand and {size} in another"
                    )
        return tuple(sizes[label] for label in self.output)


def parse_einsum(spec: str) -> EinsumSpec:
    """Parse "ab,bc->ac" into an EinsumSpec. Only explicit '->' form is accepted."""
    if spec.count("->") != 1:
        raise EinsumError(f"spec '{spec}' must contain exactly one '->'")
    lhs, output = spec.split("->")
    inputs = tuple(lhs.split(","))
    for part in (*inputs, output):
        if not part.isalpha() or not part.islower() or not part.isascii():
            raise EinsumError(f"spec '{spec}' must use lowercase letters a-z only")
        if len(set(part)) != len(part):
            raise EinsumError(f"spec '{spec}' repeats an index within one operand")
    used = set("".join(inputs))
    missing = [label for label in output if label not in used]
    if missing:
        raise EinsumError(f"output index '{missing[0]}' does not appear in any input")
    return EinsumSpec(inputs, output)
