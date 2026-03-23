"""Tests for graph model construction."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import (
    SVF,
    AudioInput,
    AudioOutput,
    BinOp,
    Constant,
    DelayLine,
    DelayRead,
    DelayWrite,
    Fold,
    Graph,
    History,
    Mix,
    OnePole,
    Wrap,
)

MAX_DELAY_SAMPLES = 96000
DEFAULT_ONEPOLE_COEFF = 0.5
DEFAULT_MIX = 0.5


class TestNodeConstruction:
    """Verify common node constructors."""

    def test_binop(self) -> None:
        """Test binary op construction."""
        node = BinOp(id="x", op="add", a="in1", b=0.5)
        assert node.id == "x"
        assert node.op == "add"

    def test_constant(self) -> None:
        """Test constant construction."""
        node = Constant(id="pi", value=3.14159)
        assert node.op == "constant"
        assert node.value == pytest.approx(3.14159)

    def test_delay_nodes(self) -> None:
        """Test delay node construction."""
        line = DelayLine(id="dl", max_samples=MAX_DELAY_SAMPLES)
        read = DelayRead(id="dr", delay="dl", tap=100.0)
        write = DelayWrite(id="dw", delay="dl", value="input_node")
        assert line.max_samples == MAX_DELAY_SAMPLES
        assert read.interp == "none"
        assert write.value == "input_node"

    def test_graph(self) -> None:
        """Test graph construction."""
        graph = Graph(
            name="gain",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="x")],
            nodes=[Constant(id="x", value=1.0)],
        )
        assert graph.name == "gain"
        assert len(graph.nodes) == 1

    def test_stateful_nodes(self) -> None:
        """Test stateful node construction."""
        history = History(id="h", input="result", init=0.0)
        onepole = OnePole(id="op", a="in1", coeff=DEFAULT_ONEPOLE_COEFF)
        svf = SVF(id="f", a="in1", freq=1000.0, q=0.707, mode="lp")
        assert history.op == "history"
        assert onepole.coeff == DEFAULT_ONEPOLE_COEFF
        assert svf.mode == "lp"

    def test_shape_nodes(self) -> None:
        """Test shape node construction."""
        wrap = Wrap(id="w", a="in1")
        fold = Fold(id="f", a="in1")
        mix = Mix(id="m", a="x", b="y", t=DEFAULT_MIX)
        assert wrap.hi == 1.0
        assert fold.lo == 0.0
        assert mix.t == DEFAULT_MIX
