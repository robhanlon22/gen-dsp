"""Tests for topological sorting of DSP graphs."""

from __future__ import annotations

pydantic = __import__("pytest").importorskip("pydantic")
import pytest

from gen_dsp.graph import AudioOutput, BinOp, Graph, toposort


class TestToposortOrdering:
    """Verify topological ordering constraints."""

    def test_result_after_inputs(self, stereo_gain_graph: Graph) -> None:
        """Test result after inputs."""
        order = [n.id for n in toposort(stereo_gain_graph)]
        assert order == ["scaled1", "scaled2"]

    def test_history_before_dependents(self, onepole_graph: Graph) -> None:
        """Test history before dependents."""
        order = [n.id for n in toposort(onepole_graph)]
        assert order.index("prev") < order.index("wet")
        assert order.index("dry") < order.index("result")
        assert order.index("wet") < order.index("result")

    def test_delayline_before_read_write(self, fbdelay_graph: Graph) -> None:
        """Test delayline before read write."""
        order = [n.id for n in toposort(fbdelay_graph)]
        assert order.index("dline") < order.index("delayed")
        assert order.index("dline") < order.index("dwrite")
        assert order.index("delayed") < order.index("fb_scaled")
        assert order.index("delayed") < order.index("wet")

class TestToposortCompleteness:
    """Verify all nodes are present in the output."""

    def test_all_nodes_present(self, onepole_graph: Graph) -> None:
        """Test all nodes present."""
        result = toposort(onepole_graph)
        result_ids = {n.id for n in result}
        expected_ids = {n.id for n in onepole_graph.nodes}
        assert result_ids == expected_ids

    def test_returns_node_instances(self, onepole_graph: Graph) -> None:
        """Test returns node instances."""
        result = toposort(onepole_graph)
        node_map = {n.id: n for n in onepole_graph.nodes}
        for node in result:
            assert node is node_map[node.id]

class TestToposortEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_graph(self) -> None:
        """Test empty graph."""
        g = Graph(name="empty")
        assert toposort(g) == []

    def test_cycle_raises(self) -> None:
        """Test cycle raises."""
        g = Graph(
            name="bad",
            nodes=[
                BinOp(id="a", op="add", a="b", b=0.0),
                BinOp(id="b", op="add", a="a", b=0.0),
            ],
            outputs=[AudioOutput(id="out1", source="a")],
        )
        with pytest.raises(ValueError, match="cycle"):
            toposort(g)

    def test_deterministic_output(self, stereo_gain_graph: Graph) -> None:
        """Multiple calls produce identical ordering."""
        order1 = [n.id for n in toposort(stereo_gain_graph)]
        order2 = [n.id for n in toposort(stereo_gain_graph)]
        assert order1 == order2
