"""
Build smoke tests for dsp-graph projects across all platforms.

These tests only check project generation structure; they do not invoke
native toolchains so Ruff linting stays fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Graph,
    Param,
    SinOsc,
)


@pytest.fixture
def gain_graph() -> Graph:
    """Minimal mono gain effect."""
    return Graph(
        name="gain",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="scaled")],
        params=[Param(name="volume", min=0.0, max=1.0, default=0.5)],
        nodes=[BinOp(id="scaled", op="mul", a="in1", b="volume")],
    )


@pytest.fixture
def generator_graph() -> Graph:
    """Minimal generator graph."""
    return Graph(
        name="sine_gen",
        inputs=[],
        outputs=[AudioOutput(id="out1", source="scaled")],
        params=[Param(name="freq", min=20.0, max=20000.0, default=440.0)],
        nodes=[
            SinOsc(id="osc", freq="freq"),
            BinOp(id="scaled", op="mul", a="osc", b=1.0),
        ],
    )


class TestProjectGeneration:
    """Verify generated project structure."""

    def test_generate_gain_project(self, gain_graph: Graph, tmp_path: Path) -> None:
        """Test a generated project contains core files."""
        config = ProjectConfig(name="gain", platform="clap")
        gen = ProjectGenerator.from_graph(gain_graph, config)
        out = gen.generate(output_dir=tmp_path / "gain_clap")
        assert (out / "gain.cpp").is_file()
        assert (out / "CMakeLists.txt").is_file()
        assert (out / "manifest.json").is_file()

    def test_generate_generator_project(
        self, generator_graph: Graph, tmp_path: Path
    ) -> None:
        """Test generator project generation."""
        config = ProjectConfig(name="sine_gen", platform="clap")
        gen = ProjectGenerator.from_graph(generator_graph, config)
        out = gen.generate(output_dir=tmp_path / "gen_clap")
        assert (out / "sine_gen.cpp").is_file()
        assert (out / "manifest.json").is_file()
