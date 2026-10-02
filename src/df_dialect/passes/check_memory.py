"""df-check-memory: fail if any actor needs more local memory than a core has.

Usage:
    df-opt input.mlir -p df-check-memory{budget-kb=48}

The footprint of an actor is the sum of the sizes of every tensor value its
body produces (df.recv results and compute results). This is the estimate
used on the Verifier Checks slide: A 16 KB + B 16 KB + C 8 KB = 40 KB.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from df_dialect.df.tier_b import ActorOp


def element_bytes(element_type: Attribute) -> int:
    """Size in bytes of one element, rounded up (i1 counts as 1 byte)."""
    for name in ("bitwidth", "width"):
        bits = getattr(element_type, name, None)
        if bits is not None:
            bits = bits.data if hasattr(bits, "data") else bits
            return max(1, (int(bits) + 7) // 8)
    raise PassFailedException(f"df-check-memory: unknown element size for {element_type}")


def tensor_bytes(t: TensorType) -> int:
    count = 1
    for dim in t.get_shape():
        if dim < 0:
            raise PassFailedException(f"df-check-memory: dynamic shape in {t}")
        count *= dim
    return count * element_bytes(t.element_type)


def actor_footprint(actor: ActorOp) -> int:
    total = 0
    for op in actor.body.walk():
        for result in op.results:
            if isinstance(result.type, TensorType):
                total += tensor_bytes(result.type)
    return total


@dataclass(frozen=True)
class CheckMemoryPass(ModulePass):
    """Fail if any df.actor holds more than `budget_kb` kilobytes of tensors."""

    name = "df-check-memory"

    budget_kb: int = 48
    """Per-core local memory budget in KB. 48 matches one Cerebras WSE-3 PE."""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for actor in op.walk():
            if not isinstance(actor, ActorOp):
                continue
            used = actor_footprint(actor)
            if used > self.budget_kb * 1024:
                raise PassFailedException(
                    f"actor @{actor.sym_name.data} holds {used // 1024} KB; "
                    f"budget is {self.budget_kb} KB per core"
                )
