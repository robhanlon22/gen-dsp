"""Tests for SuperCollider UGen platform implementation."""

from pathlib import Path

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, SuperColliderPlatform, get_platform

NUM_0 = 0


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="sc", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestScPlatform:
    """Test SuperCollider platform registry and basic properties."""

    def test_registry_contains_sc(self) -> object:
        """The SC platform should be registered."""
        assert "sc" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["sc"] == SuperColliderPlatform

    def test_get_platform_sc(self) -> object:
        """`get_platform('sc')` should return the SC platform."""
        platform = get_platform("sc")
        assert isinstance(platform, SuperColliderPlatform)
        assert platform.name == "sc"

    def test_sc_extension(self) -> object:
        """The SC extension should be `.scx` or `.so`."""
        platform = SuperColliderPlatform()
        assert platform.extension in (".scx", ".so")

    def test_sc_build_instructions(self) -> object:
        """SC build instructions should reference `cmake`."""
        platform = SuperColliderPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("cmake" in instr for instr in instructions)


class TestScProjectGeneration:
    """Test SuperCollider project generation."""

    def test_generate_sc_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate an SC project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "CMakeLists.txt").is_file()
        assert (project_dir / "gen_ext_sc.cpp").is_file()
        assert (project_dir / "_ext_sc.cpp").is_file()
        assert (project_dir / "_ext_sc.h").is_file()
        assert (project_dir / "gen_ext_common_sc.h").is_file()
        assert (project_dir / "sc_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "Testverb.sc").is_file()
        assert (project_dir / "gen").is_dir()
        assert (project_dir / "build").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_sc_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate an SC project with one buffer."""
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
        """The generated CMakeLists should reference the expected SC pieces."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "set(PROJECT_NAME testverb)" in cmake
        assert "SC_EXT_NAME=testverb" in cmake
        assert "SC_UGEN_NAME=Testverb" in cmake
        assert "GEN_EXPORTED_NAME=gen_exported" in cmake
        assert "GENLIB_USE_FLOAT32" in cmake
        assert "FetchContent_Declare" in cmake
        assert "supercollider" in cmake

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
