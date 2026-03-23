"""Tests for the dsp-graph CLI."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")

from gen_dsp.graph.cli import main


@pytest.fixture
def graph_json(tmp_path: Path) -> Path:
    """Write a minimal valid graph JSON and return its path."""
    data = {
        "name": "test_graph",
        "inputs": [{"id": "in1"}],
        "outputs": [{"id": "out1", "source": "scaled"}],
        "params": [{"name": "gain", "min": 0.0, "max": 2.0, "default": 1.0}],
        "nodes": [{"id": "scaled", "op": "mul", "a": "in1", "b": "gain"}],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def invalid_graph_json(tmp_path: Path) -> Path:
    """Write a graph JSON with validation errors and return its path."""
    data = {
        "name": "bad",
        "nodes": [{"id": "a", "op": "add", "a": "missing", "b": 0.0}],
        "outputs": [{"id": "out1", "source": "a"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    return path


def _write_test_wav(path: Path, samples: list[float], sample_rate: int = 44100) -> None:
    """Write a mono float32 WAV file for testing."""
    raw = np.array(samples, dtype=np.float32).tobytes()
    n_channels = 1
    bits = 32
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    data_size = len(raw)

    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 3))
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw)


class TestCompile:
    """Tests for compile commands."""

    def test_compile_to_stdout(
        self, graph_json: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test compile to stdout."""
        rc = main(["compile", str(graph_json)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "TestGraphState" in out

    def test_compile_to_dir(self, graph_json: Path, tmp_path: Path) -> None:
        """Test compile to directory."""
        out_dir = tmp_path / "build"
        rc = main(["compile", str(graph_json), "-o", str(out_dir)])
        assert rc == 0
        cpp = out_dir / "test_graph.cpp"
        assert cpp.exists()
        assert "TestGraphState" in cpp.read_text()


class TestValidate:
    """Tests for validate commands."""

    def test_validate_valid(
        self, graph_json: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test validate on valid graph."""
        rc = main(["validate", str(graph_json)])
        assert rc == 0
        assert "valid" in capsys.readouterr().out

    def test_validate_invalid_exits_1(
        self, invalid_graph_json: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test validate on invalid graph."""
        rc = main(["validate", str(invalid_graph_json)])
        assert rc == 1
        assert "error:" in capsys.readouterr().err


class TestDot:
    """Tests for dot commands."""

    def test_dot_to_dir(self, graph_json: Path, tmp_path: Path) -> None:
        """Test dot output directory."""
        out_dir = tmp_path / "dot_out"
        rc = main(["dot", str(graph_json), "-o", str(out_dir)])
        assert rc == 0
        dot_file = out_dir / "test_graph.dot"
        assert dot_file.exists()
        assert "digraph" in dot_file.read_text()


class TestSimulate:
    """Tests for sim commands."""

    def test_generator_no_inputs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test generator simulation without audio inputs."""
        data = {
            "name": "gen",
            "outputs": [{"id": "out1", "source": "osc"}],
            "params": [{"name": "freq", "default": 440.0}],
            "nodes": [{"id": "osc", "op": "sinosc", "freq": "freq"}],
        }
        graph = tmp_path / "gen.json"
        graph.write_text(json.dumps(data))
        out_dir = tmp_path / "sim_out"

        rc = main(["sim", str(graph), "-n", "100", "-o", str(out_dir)])
        assert rc == 0
        assert (out_dir / "out1.wav").exists()
        assert "wrote" in capsys.readouterr().out

    def test_wav_round_trip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test WAV input and output file generation."""
        input_wav = tmp_path / "input.wav"
        _write_test_wav(input_wav, [0.5] * 64, sample_rate=48000)
        graph = tmp_path / "gain.json"
        graph.write_text(
            json.dumps(
                {
                    "name": "gain",
                    "inputs": [{"id": "in1"}],
                    "outputs": [{"id": "out1", "source": "scaled"}],
                    "params": [{"name": "gain", "default": 0.5}],
                    "nodes": [
                        {"id": "scaled", "op": "mul", "a": "in1", "b": "gain"}
                    ],
                }
            )
        )
        out_dir = tmp_path / "wav_out"
        rc = main(["sim", str(graph), "-i", f"in1={input_wav}", "-o", str(out_dir)])
        assert rc == 0
        assert (out_dir / "out1.wav").exists()
        assert capsys.readouterr().err == ""
