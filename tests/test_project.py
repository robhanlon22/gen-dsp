"""Tests for gen_dsp.core.project module."""


from pathlib import Path

import pytest

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.errors import ValidationError

NUM_0 = 0
NUM_1 = 1


class TestProjectConfig:
    """Tests for ProjectConfig class."""

    def test_valid_config(self) -> object:
        """A valid config should produce no validation errors."""
        config = ProjectConfig(
            name="myeffect",
            platform="pd",
            buffers=["sample", "buffer1"],
        )
        errors = config.validate()
        assert errors == []

    def test_invalid_name_starts_with_number(self) -> object:
        """Names starting with a digit should be rejected."""
        config = ProjectConfig(name="123invalid", platform="pd")
        errors = config.validate()
        assert len(errors) == NUM_1
        assert "not a valid C identifier" in errors[NUM_0]

    def test_invalid_name_with_spaces(self) -> object:
        """Names with spaces should be rejected."""
        config = ProjectConfig(name="has space", platform="pd")
        errors = config.validate()
        assert len(errors) == NUM_1
        assert "not a valid C identifier" in errors[NUM_0]

    def test_invalid_platform(self) -> object:
        """Unknown platforms should be rejected."""
        config = ProjectConfig(name="valid", platform="invalid")
        errors = config.validate()
        assert len(errors) == NUM_1
        assert "Platform must be one of" in errors[NUM_0]
        assert "'invalid'" in errors[NUM_0]

    def test_too_many_buffers(self) -> object:
        """More than five buffers should be rejected."""
        config = ProjectConfig(
            name="valid",
            platform="pd",
            buffers=["b1", "b2", "b3", "b4", "b5", "b6"],
        )
        errors = config.validate()
        assert len(errors) == NUM_1
        assert "Maximum 5 buffers" in errors[NUM_0]

    def test_invalid_buffer_name(self) -> object:
        """Invalid buffer names should be rejected."""
        config = ProjectConfig(
            name="valid",
            platform="pd",
            buffers=["valid", "123invalid"],
        )
        errors = config.validate()
        assert len(errors) == NUM_1
        assert "not a valid C identifier" in errors[NUM_0]


class TestProjectGenerator:
    """Tests for ProjectGenerator class."""

    def test_generate_pd_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test generating PureData project without buffers."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="testverb", platform="pd")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        # Check directory was created
        assert project_dir.is_dir()

        # Check required files exist
        assert (project_dir / "Makefile").is_file()
        assert (project_dir / "gen_dsp.cpp").is_file()
        assert (project_dir / "gen_ext_common.h").is_file()
        assert (project_dir / "_ext.cpp").is_file()
        assert (project_dir / "_ext.h").is_file()
        assert (project_dir / "pd_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "pd-lib-builder").is_dir()
        assert (project_dir / "gen").is_dir()

        # Check Makefile content
        makefile = (project_dir / "Makefile").read_text()
        assert "gen.name = gen_exported" in makefile
        assert "lib.name = testverb" in makefile

        # Check gen_buffer.h has 0 buffers
        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_pd_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """Test generating PureData project with buffers."""
        parser = GenExportParser(rampleplayer_export)
        export_info = parser.parse()

        config = ProjectConfig(
            name="testsampler",
            platform="pd",
            buffers=["sample"],  # Override to ensure we have the expected buffer
        )
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        # Check gen_buffer.h has buffer configured
        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 sample" in buffer_h

    def test_generate_with_multiple_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test generating project with multiple buffers."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(
            name="multibuf",
            platform="pd",
            buffers=["buf1", "buf2", "buf3"],
        )
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 3" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 buf1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_1 buf2" in buffer_h
        assert "WRAPPER_BUFFER_NAME_2 buf3" in buffer_h

    def test_generate_invalid_config_raises_error(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test that invalid config raises ValidationError."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="123invalid", platform="pd")
        generator = ProjectGenerator(export_info, config)

        with pytest.raises(ValidationError):
            generator.generate(tmp_project)

    def test_generate_copies_gen_export(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test that gen~ export is copied to project."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="test", platform="pd")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        # Check gen directory and files
        gen_dir = project_dir / "gen"
        assert gen_dir.is_dir()
        assert (gen_dir / "gen_exported.cpp").is_file()
        assert (gen_dir / "gen_exported.h").is_file()
        assert (gen_dir / "gen_dsp").is_dir()
        assert (gen_dir / "gen_dsp" / "genlib.cpp").is_file()
