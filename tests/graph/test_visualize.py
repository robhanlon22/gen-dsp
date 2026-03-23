"""Tests for graph visualization."""

from __future__ import annotations

pydantic = __import__("pytest").importorskip("pydantic")
from pathlib import Path

from gen_dsp.graph import (
    AudioOutput,
    Constant,
    Graph,
    graph_to_dot,
    graph_to_dot_file,
)


class TestGraphToDot:
    """Verify DOT output content."""

    def test_header(self, stereo_gain_graph: Graph) -> None:
        """Test DOT header."""
        dot = graph_to_dot(stereo_gain_graph)
        assert 'digraph "stereo_gain"' in dot
        assert "rankdir=LR" in dot
        assert "fontname" in dot

    def test_forward_edges(self, stereo_gain_graph: Graph) -> None:
        """Test forward edges are emitted."""
        dot = graph_to_dot(stereo_gain_graph)
        assert '"in1" -> "scaled1"' in dot
        assert '"gain" -> "scaled1"' in dot
        assert '"in2" -> "scaled2"' in dot
        assert '"gain" -> "scaled2"' in dot

    def test_feedback_edge_dashed(self, onepole_graph: Graph) -> None:
        """Test feedback edge is dashed."""
        dot = graph_to_dot(onepole_graph)
        assert '"result" -> "prev"' in dot
        assert "dashed" in dot

    def test_constant_node(self) -> None:
        """Test constant node styling."""
        graph = Graph(
            name="const_test",
            outputs=[AudioOutput(id="out1", source="c")],
            nodes=[Constant(id="c", value=3.14)],
        )
        dot = graph_to_dot(graph)
        assert "3.14" in dot
        assert "#e9ecef" in dot

    def test_dot_file_written(self, fbdelay_graph: Graph, tmp_path: Path) -> None:
        """Test DOT files are written to disk."""
        out = graph_to_dot_file(fbdelay_graph, tmp_path)
        assert out.exists()
        assert out.suffix == ".dot"
