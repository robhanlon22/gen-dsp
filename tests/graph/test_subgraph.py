"""Tests for subgraph expansion."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import AudioInput, AudioOutput, BinOp, Graph, Subgraph
from gen_dsp.graph.subgraph import expand_subgraphs
from gen_dsp.graph.validate import validate_graph


class TestSubgraphExpansion:
    """Verify simple subgraph behavior."""

    def test_expand_simple_subgraph(self) -> None:
        """Test a nested graph expands cleanly."""
        inner = Graph(
            name="inner",
            inputs=[AudioInput(id="sig")],
            outputs=[AudioOutput(id="y", source="mul")],
            nodes=[BinOp(id="mul", op="mul", a="sig", b=0.5)],
        )
        graph = Graph(
            name="outer",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="filt")],
            nodes=[Subgraph(id="filt", graph=inner, inputs={"sig": "in1"})],
        )
        expanded = expand_subgraphs(graph)
        assert expanded.name == "outer"
        assert validate_graph(expanded) == []

    def test_subgraph_node_constructs(self) -> None:
        """Test Subgraph construction remains available."""
        inner = Graph(name="inner", outputs=[])
        node = Subgraph(id="sg", graph=inner, inputs={})
        assert node.id == "sg"
