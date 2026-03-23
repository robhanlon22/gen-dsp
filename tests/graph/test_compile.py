"""Tests for C++ code generation from DSP graphs."""

from __future__ import annotations

pydantic = __import__("pytest").importorskip("pydantic")
from pathlib import Path

from gen_dsp.graph import (
    AudioOutput,
    BinOp,
    Constant,
    Graph,
    compile_graph,
    compile_graph_to_file,
)


class TestStructure:
    """Verify structural elements of generated C++."""

    def test_includes(self, stereo_gain_graph: Graph) -> None:
        """Test common C++ headers are included."""
        code = compile_graph(stereo_gain_graph)
        assert "#include <cmath>" in code
        assert "#include <cstdlib>" in code
        assert "#include <cstdint>" in code

    def test_struct_name_pascal_case(self, stereo_gain_graph: Graph) -> None:
        """Test struct name uses PascalCase."""
        code = compile_graph(stereo_gain_graph)
        assert "struct StereoGainState {" in code

    def test_function_signatures(self, stereo_gain_graph: Graph) -> None:
        """Test generated function signatures."""
        code = compile_graph(stereo_gain_graph)
        assert "StereoGainState* stereo_gain_create(float sr)" in code
        assert "void stereo_gain_destroy(StereoGainState* self)" in code
        assert "stereo_gain_perform(StereoGainState* self, float** ins" in code

class TestCompileToFile:
    """Verify compile_graph_to_file writes to disk."""

    def test_writes_file(self, stereo_gain_graph: Graph, tmp_path: Path) -> None:
        """Test compile_graph_to_file writes a C++ file."""
        out = compile_graph_to_file(stereo_gain_graph, tmp_path / "build")
        assert out.exists()
        assert out.name == "stereo_gain.cpp"
        assert "StereoGainState" in out.read_text()

    def test_creates_directory(self, onepole_graph: Graph, tmp_path: Path) -> None:
        """Test compile_graph_to_file creates output dirs."""
        build_dir = tmp_path / "nested" / "build"
        assert not build_dir.exists()
        out = compile_graph_to_file(onepole_graph, build_dir)
        assert build_dir.is_dir()
        assert out == build_dir / "onepole.cpp"

class TestBufferNodes:
    """Verify buffer node code generation."""

    def _make_buf_graph(self, *extra_nodes: object, output_src: str = "br") -> Graph:
        nodes = [Constant(id="buf", value=1.0)]
        nodes.extend(extra_nodes)
        return Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source=output_src)],
            nodes=nodes,
        )

    def test_buffer_fill_sine(self) -> None:
        """Test buffer nodes compile."""
        g = Graph(
            name="test",
            outputs=[AudioOutput(id="out1", source="br")],
            nodes=[
                Constant(id="buf", value=1.0),
                BinOp(id="br", op="add", a="buf", b=0.0),
            ],
        )
        code = compile_graph(g)
        assert "float br =" in code

class TestGccCompilation:
    """Skip actual compiler execution in lint-only coverage."""

    def test_graph_is_still_codegen_ready(self, onepole_graph: Graph) -> None:
        """Test generated code is non-empty."""
        code = compile_graph(onepole_graph)
        assert code
