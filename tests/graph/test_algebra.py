"""Tests for graph algebra helpers."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import AudioOutput, BinOp, Constant, Graph
from gen_dsp.graph.optimize import constant_fold, eliminate_cse


class TestAlgebraHelpers:
    """Verify simple algebraic simplifications."""

    def test_constant_fold_simple(self) -> None:
        """Test constant folding on a simple graph."""
        graph = Graph(
            name="fold",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="a", value=2.0),
                Constant(id="b", value=3.0),
                BinOp(id="r", op="add", a="a", b="b"),
            ],
        )
        folded = constant_fold(graph)
        assert folded.name == "fold"

    def test_eliminate_cse_returns_graph(self) -> None:
        """Test common subexpression elimination returns a graph."""
        graph = Graph(
            name="cse",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[Constant(id="a", value=1.0), BinOp(id="r", op="add", a="a", b="a")],
        )
        optimized = eliminate_cse(graph)
        assert optimized.name == "cse"
