"""Tests for graph validation."""

from __future__ import annotations

pydantic = __import__("pytest").importorskip("pydantic")

from gen_dsp.graph import AudioInput, AudioOutput, BinOp, Graph, History, validate_graph


class TestValidGraphs:
    """Validate basic well-formed graphs."""

    def test_stereo_gain_valid(self, stereo_gain_graph: Graph) -> None:
        """Test stereo gain is valid."""
        assert validate_graph(stereo_gain_graph) == []

    def test_onepole_valid(self, onepole_graph: Graph) -> None:
        """Test one-pole graph is valid."""
        assert validate_graph(onepole_graph) == []

    def test_fbdelay_valid(self, fbdelay_graph: Graph) -> None:
        """Test feedback delay graph is valid."""
        assert validate_graph(fbdelay_graph) == []

class TestInvalidGraphs:
    """Validate common error cases."""

    def test_duplicate_node_id(self) -> None:
        """Test duplicate node IDs are rejected."""
        g = Graph(
            name="test",
            nodes=[
                BinOp(id="x", op="add", a=1.0, b=2.0),
                BinOp(id="x", op="mul", a=3.0, b=4.0),
            ],
        )
        errors = validate_graph(g)
        assert any("Duplicate node ID" in e for e in errors)

    def test_node_references_unknown_id(self) -> None:
        """Test dangling references are rejected."""
        g = Graph(
            name="test",
            nodes=[BinOp(id="x", op="add", a="nonexistent", b=1.0)],
        )
        errors = validate_graph(g)
        assert any("unknown ID 'nonexistent'" in e for e in errors)

    def test_cycle_through_history_allowed(self) -> None:
        """Test feedback through history is valid."""
        g = Graph(
            name="test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="result")],
            nodes=[
                History(id="prev", input="result", init=0.0),
                BinOp(id="result", op="add", a="in1", b="prev"),
            ],
        )
        assert validate_graph(g) == []
