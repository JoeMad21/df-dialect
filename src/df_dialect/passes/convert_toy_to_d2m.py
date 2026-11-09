from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from df_dialect.d2m import TileAddOp, TileMatmulOp, TileReluOp
from df_dialect.d2m.builders import build_tile_compute, verify_tile_tensor
from df_dialect.toy import AddOp, MatmulOp, ReluOp


class LowerMatmul(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, build_tile_compute([op.lhs, op.rhs], op.result.type, TileMatmulOp))


class LowerAdd(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, build_tile_compute([op.lhs, op.rhs], op.result.type, TileAddOp))


class LowerRelu(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReluOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, build_tile_compute([op.input], op.result.type, TileReluOp))


@dataclass(frozen=True)
class ConvertToyToD2MPass(ModulePass):
    name = "convert-toy-to-d2m"

    def apply(self, ctx: Context, op: ModuleOp) -> None:

        for toy_op in op.walk():
            if not isinstance(toy_op, MatmulOp | AddOp | ReluOp):
                continue
            for value in (*toy_op.operands, toy_op.result):
                verify_tile_tensor(value.type)
        patterns = GreedyRewritePatternApplier([LowerMatmul(), LowerAdd(), LowerRelu()])
        PatternRewriteWalker(patterns).rewrite_module(op)
