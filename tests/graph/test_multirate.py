"""Tests for multirate graph nodes."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import (
    AudioInput,
    AudioOutput,
    Graph,
    RateDiv,
    SampleHold,
    Slide,
    Splat,
    toposort,
    validate_graph,
)


class TestMultirateNodes:
    """Verify multirate node handling."""

    def test_rate_div_graph_valid(self) -> None:
        """Test RateDiv appears in a valid graph."""
        graph = Graph(
            name="rate_div",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="rate")],
            nodes=[RateDiv(id="rate", a="in1", rate=2)],
        )
        assert validate_graph(graph) == []
        assert [node.id for node in toposort(graph)] == ["rate"]

    def test_sample_hold_graph_valid(self) -> None:
        """Test SampleHold appears in a valid graph."""
        graph = Graph(
            name="sample_hold",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="hold")],
            nodes=[SampleHold(id="hold", a="in1", trigger=1.0)],
        )
        assert validate_graph(graph) == []

    def test_slide_and_splat_graph(self) -> None:
        """Test other multirate nodes construct cleanly."""
        graph = Graph(
            name="slide_splat",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="sum")],
            nodes=[
                Slide(id="sl", a="in1", up=10.0, down=20.0),
                Splat(id="sum", a="sl"),
            ],
        )
        assert validate_graph(graph) == []
