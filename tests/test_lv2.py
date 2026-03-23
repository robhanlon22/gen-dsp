"""Tests for LV2 plugin platform implementation."""

from pathlib import Path

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, get_platform
from gen_dsp.platforms.lv2 import Lv2Platform

NUM_0 = 0


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="lv2", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestLv2Platform:
    """Test LV2 platform registry and basic properties."""

    def test_registry_contains_lv2(self) -> object:
        """The LV2 platform should be registered."""
        assert "lv2" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["lv2"] == Lv2Platform

    def test_get_platform_lv2(self) -> object:
        """`get_platform('lv2')` should return the LV2 platform."""
        platform = get_platform("lv2")
        assert isinstance(platform, Lv2Platform)
        assert platform.name == "lv2"

    def test_lv2_extension(self) -> object:
        """The LV2 extension should be `.lv2`."""
        platform = Lv2Platform()
        assert platform.extension == ".lv2"

    def test_lv2_build_instructions(self) -> object:
        """LV2 build instructions should reference `cmake`."""
        platform = Lv2Platform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("cmake" in instr for instr in instructions)


class TestLv2ProjectGeneration:
    """Test LV2 project generation."""

    def test_generate_lv2_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate an LV2 project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "CMakeLists.txt").is_file()
        assert (project_dir / "gen_ext_lv2.cpp").is_file()
        assert (project_dir / "_ext_lv2.cpp").is_file()
        assert (project_dir / "_ext_lv2.h").is_file()
        assert (project_dir / "gen_ext_common_lv2.h").is_file()
        assert (project_dir / "lv2_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "manifest.ttl").is_file()
        assert (project_dir / "testverb.ttl").is_file()
        assert (project_dir / "gen").is_dir()
        assert (project_dir / "build").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_lv2_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate an LV2 project with one buffer."""
        project_dir = _generate_project(
            rampleplayer_export,
            tmp_project,
            "testsampler",
            buffers=["sample"],
        )

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 sample" in buffer_h

    def test_cmakelists_content(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """The generated CMakeLists should reference the expected LV2 pieces."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "set(PROJECT_NAME testverb)" in cmake
        assert "LV2_EXT_NAME=testverb" in cmake
        assert "GEN_EXPORTED_NAME=gen_exported" in cmake
        assert "GENLIB_USE_FLOAT32" in cmake
        assert "FetchContent_Declare" in cmake
        assert "lv2/lv2" in cmake

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
