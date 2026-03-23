"""Tests for the Python DSP simulator."""

from __future__ import annotations

import numpy as np
import pytest

pydantic = pytest.importorskip("pydantic")
pytest.importorskip("numpy")

from gen_dsp.graph import AudioInput, AudioOutput, BinOp, Graph, History, Param
from gen_dsp.graph.simulate import SimResult, SimState, simulate

DEFAULT_SAMPLE_RATE = 48_000.0
DEFAULT_VOLUME = 0.5
UPDATED_VOLUME = 0.8


class TestSimStateAPI:
    """Verify state management APIs."""

    def test_create(self) -> None:
        """Test state creation."""
        graph = Graph(name="empty", outputs=[])
        state = SimState(graph, sample_rate=DEFAULT_SAMPLE_RATE)
        assert state.sr == DEFAULT_SAMPLE_RATE

    def test_param_get_set(self) -> None:
        """Test parameter get/set."""
        graph = Graph(
            name="p",
            params=[Param(name="vol", default=DEFAULT_VOLUME)],
            outputs=[],
        )
        state = SimState(graph)
        assert state.get_param("vol") == DEFAULT_VOLUME
        state.set_param("vol", UPDATED_VOLUME)
        assert state.get_param("vol") == UPDATED_VOLUME


class TestSimulateAPI:
    """Verify simulation results."""

    def test_result_shape(self) -> None:
        """Test output shape and dtype."""
        graph = Graph(
            name="pass",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="g1")],
            nodes=[BinOp(id="g1", op="mul", a="in1", b=1.0)],
        )
        inp = np.ones(16, dtype=np.float32)
        result = simulate(graph, inputs={"in1": inp})
        assert isinstance(result, SimResult)
        assert result.outputs["out1"].shape == (16,)
        assert result.outputs["out1"].dtype == np.float32

    def test_state_reuse(self) -> None:
        """Test state reuse across simulation calls."""
        graph = Graph(
            name="acc",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="a1")],
            nodes=[BinOp(id="a1", op="add", a="in1", b=1.0)],
        )
        inp = np.ones(5, dtype=np.float32)
        first = simulate(graph, inputs={"in1": inp})
        second = simulate(graph, inputs={"in1": inp}, state=first.state)
        assert second.outputs["out1"].shape == (5,)

    def test_reset_restores_history_output(self) -> None:
        """Test reset restores history output behavior."""
        graph = Graph(
            name="r",
            params=[Param(name="vol", default=DEFAULT_VOLUME)],
            nodes=[History(id="h1", init=1.0, input="h1")],
            outputs=[AudioOutput(id="out1", source="h1")],
        )
        state = SimState(graph)
        simulate(graph, n_samples=10, state=state)
        state.reset()
        assert state.get_param("vol") == DEFAULT_VOLUME
        result = simulate(graph, n_samples=1, state=state)
        assert result.outputs["out1"][0] == pytest.approx(1.0)
