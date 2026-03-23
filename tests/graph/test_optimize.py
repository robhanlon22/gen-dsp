"""Tests for optimization passes."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import (
    AudioOutput,
    BinOp,
    Constant,
    Graph,
    constant_fold,
    eliminate_dead_nodes,
    optimize_graph,
)


class TestConstantFold:
    """Verify constant folding."""

    def test_binop_div_folded(self) -> None:
        """Test division folds to a constant."""
        graph = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="r")],
            nodes=[
                Constant(id="sr", value=44100.0),
                Constant(id="k", value=1000.0),
                BinOp(id="r", op="div", a="sr", b="k"),
            ],
        )
        folded = constant_fold(graph)
        node = {n.id: n for n in folded.nodes}["r"]
        assert isinstance(node, Constant)

    def test_chain_folds(self) -> None:
        """Test chained constants collapse."""
        graph = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="result")],
            nodes=[
                Constant(id="a", value=2.0),
                Constant(id="b", value=3.0),
                BinOp(id="sum", op="add", a="a", b="b"),
                Constant(id="c", value=4.0),
                BinOp(id="result", op="mul", a="sum", b="c"),
            ],
        )
        folded = constant_fold(graph)
        node = {n.id: n for n in folded.nodes}["result"]
        assert isinstance(node, Constant)

class TestOptimizeHelpers:
    """Verify helper passes."""

    def test_eliminate_dead_nodes(self) -> None:
        """Test dead nodes are removed."""
        graph = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="out")],
            nodes=[Constant(id="dead", value=1.0), Constant(id="out", value=0.0)],
        )
        pruned = eliminate_dead_nodes(graph)
        assert any(node.id == "out" for node in pruned.nodes)

    def test_optimize_graph_returns_graph(self) -> None:
        """Test optimize_graph returns a graph object."""
        graph = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="out")],
            nodes=[Constant(id="out", value=0.0)],
        )
        optimized = optimize_graph(graph)
        assert optimized.name == "test"
