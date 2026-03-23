"""Tests for gen_dsp.core.parser module."""

from pathlib import Path

import pytest

from gen_dsp.core.parser import ExportInfo, GenExportParser
from gen_dsp.errors import ParseError

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_8 = 8


class TestGenExportParser:
    """Tests for GenExportParser class."""

    def test_parse_gigaverb_export(self, gigaverb_export: Path) -> object:
        """Gigaverb should parse into a populated ExportInfo."""
        parser = GenExportParser(gigaverb_export)
        info = parser.parse()

        assert info.name == "gen_exported"
        assert info.num_inputs == NUM_2  # Stereo input
        assert info.num_outputs == NUM_2
        assert info.num_params == NUM_8
        assert info.buffers == []  # No buffers
        assert info.cpp_path is not None
        assert info.cpp_path.exists()
        assert info.h_path is not None
        assert info.h_path.exists()

    def test_parse_rampleplayer_export(self, rampleplayer_export: Path) -> object:
        """RamplePlayer should parse with one input and buffers."""
        parser = GenExportParser(rampleplayer_export)
        info = parser.parse()

        assert info.name == "RamplePlayer"
        assert info.num_inputs == NUM_1
        assert info.num_outputs == NUM_2
        assert info.num_params == NUM_0
        assert "sample" in info.buffers
        assert len(info.buffers) >= NUM_1

    def test_parse_spectraldelayfb_export(self, spectraldelayfb_export: Path) -> object:
        """Spectraldelayfb should parse with nonzero I/O counts."""
        parser = GenExportParser(spectraldelayfb_export)
        info = parser.parse()

        assert info.name == "gen_exported"
        assert info.num_inputs > NUM_0
        assert info.num_outputs > NUM_0

    def test_parse_fm_bells_export(self, fm_bells_export: Path) -> object:
        """fm_bells should parse with the expected metadata."""
        parser = GenExportParser(fm_bells_export)
        info = parser.parse()

        assert info.name == "gen_exported"
        assert info.num_inputs == NUM_2
        assert info.num_outputs == NUM_2
        assert info.num_params == NUM_3
        assert info.buffers == []
        assert info.has_exp2f_issue is True

    def test_parse_slicer_export(self, slicer_export: Path) -> object:
        """Slicer should parse with its buffer metadata."""
        parser = GenExportParser(slicer_export)
        info = parser.parse()

        assert info.name == "gen_exported"
        assert info.num_inputs == NUM_1
        assert info.num_outputs == NUM_1
        assert info.num_params == NUM_3
        assert "storage" in info.buffers
        assert len(info.buffers) == NUM_1

    def test_parse_invalid_path_raises_error(self, tmp_path: Path) -> object:
        """Invalid paths should raise ParseError."""
        with pytest.raises(ParseError, match="not a directory"):
            GenExportParser(tmp_path / "nonexistent")

    def test_parse_empty_dir_raises_error(self, tmp_path: Path) -> object:
        """Empty directories should raise ParseError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ParseError, match="No gen~ export"):
            GenExportParser(empty_dir)

    def test_validate_buffer_names_valid(self, gigaverb_export: Path) -> object:
        """Valid buffer names should return no invalid names."""
        parser = GenExportParser(gigaverb_export)
        invalid = parser.validate_buffer_names(["sample", "buffer1", "_test"])
        assert invalid == []

    def test_validate_buffer_names_invalid(self, gigaverb_export: Path) -> object:
        """Invalid buffer names should be reported."""
        parser = GenExportParser(gigaverb_export)
        invalid = parser.validate_buffer_names(["123invalid", "has space", "has-dash"])
        assert len(invalid) == NUM_3
        assert "123invalid" in invalid
        assert "has space" in invalid
        assert "has-dash" in invalid


class TestExportInfo:
    """Tests for ExportInfo dataclass."""

    def test_export_info_defaults(self) -> object:
        """ExportInfo should default optional fields sensibly."""
        info = ExportInfo(name="test", path=Path())
        assert info.num_inputs == NUM_0
        assert info.num_outputs == NUM_0
        assert info.num_params == NUM_0
        assert info.buffers == []
        assert info.has_exp2f_issue is False
        assert info.cpp_path is None
        assert info.h_path is None
        assert info.genlib_ops_path is None
