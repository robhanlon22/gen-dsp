"""Integration tests for dsp-graph project generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.graph import AudioInput, AudioOutput, BinOp, Graph, Param


@pytest.fixture
def simple_gain_graph() -> Graph:
    """Minimal mono gain graph for integration testing."""
    return Graph(
        name="gain",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="mul1")],
        params=[Param(name="volume", min=0.0, max=1.0, default=0.5)],
        nodes=[BinOp(id="mul1", op="mul", a="in1", b="volume")],
    )


class TestFromGraphProjectStructure:
    """Test project generation from graphs."""

    def test_generates_project_files(
        self, simple_gain_graph: Graph, tmp_path: Path
    ) -> None:
        """Test generated projects contain key files."""
        config = ProjectConfig(name="test_gain", platform="clap")
        gen = ProjectGenerator.from_graph(simple_gain_graph, config)
        out = gen.generate(output_dir=tmp_path / "proj_clap")
        assert (out / "gain.cpp").is_file()
        assert (out / "manifest.json").is_file()
        assert (out / "CMakeLists.txt").is_file()

    def test_manifest_content(self, simple_gain_graph: Graph, tmp_path: Path) -> None:
        """Test manifest content reflects graph metadata."""
        config = ProjectConfig(name="test_gain", platform="clap")
        gen = ProjectGenerator.from_graph(simple_gain_graph, config)
        out = gen.generate(output_dir=tmp_path / "proj")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["gen_name"] == "gain"
        assert manifest["num_inputs"] == 1
        assert manifest["num_outputs"] == 1
        assert manifest["source"] == "dsp-graph"
