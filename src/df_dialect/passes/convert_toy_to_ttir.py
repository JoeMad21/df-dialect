"""convert-toy-to-ttir: lower every toy op to the matching TTIR op.

Usage:
    df-opt examples/toy_linear.mlir -p convert-toy-to-ttir

The output can be fed to tt-mlir, for example:
    ttmlir-opt --ttir-to-ttnn-runtime-pipeline="system-desc-path=$DESC" out.mlir
"""

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

from df_dialect.toy import AddOp, MatmulOp, ReluOp
from df_dialect.ttir import TTIRAddOp, TTIRMatmulOp, TTIRReluOp


class LowerMatmul(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, TTIRMatmulOp(op.lhs, op.rhs, op.result.type))


class LowerAdd(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, TTIRAddOp(op.lhs, op.rhs, op.result.type))


class LowerRelu(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReluOp, rewriter: PatternRewriter) -> None:
        rewriter.replace(op, TTIRReluOp(op.input, op.result.type))


@dataclass(frozen=True)
class ConvertToyToTTIRPass(ModulePass):
    """Replace toy.matmul, toy.add and toy.relu with ttir.matmul, ttir.add and ttir.relu."""

    name = "convert-toy-to-ttir"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = GreedyRewritePatternApplier([LowerMatmul(), LowerAdd(), LowerRelu()])
        PatternRewriteWalker(patterns).rewrite_module(op)
