"""Tests for gen-dsp adapter: reset(), adapter C++, manifest, integration."""

from __future__ import annotations

import importlib.util
import json

import pytest

pydantic = pytest.importorskip("pydantic")

from gen_dsp.graph import AudioInput, AudioOutput, Graph, History, compile_graph
from gen_dsp.graph.adapter import generate_adapter_cpp, generate_manifest

RESET_HISTORY_EXPECTED_OCCURRENCES = 2


def _try_import_gen_dsp() -> bool:
    return importlib.util.find_spec("gen_dsp") is not None

_HAS_GEN_DSP = _try_import_gen_dsp()
skip_no_gen_dsp = pytest.mark.skipif(not _HAS_GEN_DSP, reason="gen_dsp not installed")

class TestReset:
    """Verify reset-related code generation."""

    def test_reset_function_generated(self, stereo_gain_graph: Graph) -> None:
        """Test reset function is generated."""
        code = compile_graph(stereo_gain_graph)
        assert "stereo_gain_reset" in code

    def test_reset_history_nonzero_init(self) -> None:
        """Test non-zero history initial values are preserved."""
        g = Graph(
            name="h_test",
            inputs=[AudioInput(id="in1")],
            outputs=[AudioOutput(id="out1", source="h1")],
            nodes=[History(id="h1", init=0.5, input="in1")],
        )
        code = compile_graph(g)
        assert code.count("self->m_h1 = 0.5f;") >= RESET_HISTORY_EXPECTED_OCCURRENCES

class TestAdapterCpp:
    """Verify adapter C++ generation."""

    def test_adapter_includes_common_header(self, gen_dsp_graph: Graph) -> None:
        """Test adapter includes common header."""
        code = generate_adapter_cpp(gen_dsp_graph, "chuck")
        assert '#include "gen_ext_common_chuck.h"' in code

    def test_adapter_wrapper_namespace(self, gen_dsp_graph: Graph) -> None:
        """Test adapter namespace wrapping."""
        code = generate_adapter_cpp(gen_dsp_graph, "chuck")
        assert "namespace WRAPPER_NAMESPACE {" in code
        assert "} // namespace WRAPPER_NAMESPACE" in code

class TestManifest:
    """Verify manifest generation."""

    def test_manifest_json_valid(self, gen_dsp_graph: Graph) -> None:
        """Test manifest JSON is valid."""
        text = generate_manifest(gen_dsp_graph)
        data = json.loads(text)
        assert isinstance(data, dict)
        assert data["source"] == "dsp-graph"

    @skip_no_gen_dsp
    def test_manifest_gen_dsp_compatible(self, gen_dsp_graph: Graph) -> None:
        """Test manifest is compatible with gen-dsp tools."""
        text = generate_manifest(gen_dsp_graph)
        assert "test_synth" in text
