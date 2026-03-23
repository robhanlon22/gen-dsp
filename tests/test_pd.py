"""Tests for PureData external platform implementation."""

import platform as sys_platform
from pathlib import Path

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, PureDataPlatform, get_platform

NUM_0 = 0


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="pd", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestPdPlatform:
    """Test PureData platform registry and basic properties."""

    def test_registry_contains_pd(self) -> object:
        """The PD platform should be registered."""
        assert "pd" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["pd"] == PureDataPlatform

    def test_get_platform_pd(self) -> object:
        """`get_platform('pd')` should return the PD platform."""
        platform = get_platform("pd")
        assert isinstance(platform, PureDataPlatform)
        assert platform.name == "pd"

    def test_pd_extension(self) -> object:
        """The PD extension should match the host platform."""
        platform = PureDataPlatform()
        system = sys_platform.system().lower()
        if system == "darwin":
            assert platform.extension == ".pd_darwin"
        elif system == "linux":
            assert platform.extension == ".pd_linux"

    def test_pd_build_instructions(self) -> object:
        """PD build instructions should reference `make`."""
        platform = PureDataPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("make" in instr for instr in instructions)


class TestPdProjectGeneration:
    """Test PureData project generation."""

    def test_generate_pd_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate a PD project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "Makefile").is_file()
        assert (project_dir / "gen_dsp.cpp").is_file()
        assert (project_dir / "_ext.cpp").is_file()
        assert (project_dir / "_ext.h").is_file()
        assert (project_dir / "gen_ext_common.h").is_file()
        assert (project_dir / "pd_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "pd-include" / "m_pd.h").is_file()
        assert (project_dir / "pd-lib-builder").is_dir()
        assert (project_dir / "gen").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_pd_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate a PD project with one buffer."""
        project_dir = _generate_project(
            rampleplayer_export,
            tmp_project,
            "testsampler",
            buffers=["sample"],
        )

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 sample" in buffer_h

    def test_makefile_content(self, gigaverb_export: Path, tmp_project: Path) -> object:
        """The generated Makefile should reference the expected build pieces."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        makefile = (project_dir / "Makefile").read_text()
        assert "lib.name = testverb" in makefile
        assert "gen.name = gen_exported" in makefile
        assert "GENLIB_USE_FLOAT32" in makefile
        assert "pd-lib-builder" in makefile

    def test_generate_copies_gen_export(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """The gen~ export should be copied into the project."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "test")

        gen_dir = project_dir / "gen"
        assert gen_dir.is_dir()
        assert (gen_dir / "gen_exported.cpp").is_file()
        assert (gen_dir / "gen_exported.h").is_file()
        assert (gen_dir / "gen_dsp").is_dir()
        assert (gen_dir / "gen_dsp" / "genlib.cpp").is_file()
