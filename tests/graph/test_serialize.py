"""Tests for graph serialization."""

from __future__ import annotations

import json

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import AudioInput, AudioOutput, BinOp, Constant, Graph


class TestSerialize:
    """Verify basic serialization helpers."""

    def test_graph_dump_json_round_trip(self) -> None:
        """Test Graph JSON round-trip."""
        graph = Graph(
            name="serialize",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="x")],
            nodes=[Constant(id="x", value=1.0)],
        )
        text = graph.model_dump_json()
        data = json.loads(text)
        assert data["name"] == "serialize"
        restored = Graph.model_validate_json(text)
        assert restored.name == "serialize"

    def test_graph_dump_contains_nodes(self) -> None:
        """Test model dump contains node names."""
        graph = Graph(
            name="serialize2",
            outputs=[AudioOutput(id="out1", source="x")],
            nodes=[BinOp(id="x", op="add", a=1.0, b=2.0)],
        )
        data = graph.model_dump()
        assert data["name"] == "serialize2"
