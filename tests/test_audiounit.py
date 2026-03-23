"""Tests for AudioUnit (AUv2) platform implementation."""

import platform as sys_platform
from pathlib import Path

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, get_platform
from gen_dsp.platforms.audiounit import AudioUnitPlatform

NUM_0 = 0
NUM_1 = 1


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="au", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestAudioUnitPlatform:
    """Test AudioUnit platform registry and basic properties."""

    def test_registry_contains_au(self) -> object:
        """The AU platform should be registered."""
        assert "au" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["au"] == AudioUnitPlatform

    def test_get_platform_au(self) -> object:
        """`get_platform('au')` should return the AU platform."""
        platform = get_platform("au")
        assert isinstance(platform, AudioUnitPlatform)
        assert platform.name == "au"

    def test_au_extension(self) -> object:
        """The AU extension should be `.component`."""
        platform = AudioUnitPlatform()
        assert platform.extension == ".component"

    def test_au_build_instructions(self) -> object:
        """AU build instructions should reference `cmake`."""
        platform = AudioUnitPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("cmake" in instr for instr in instructions)

    def test_extension_is_consistent(self) -> object:
        """AU extensions should be the same across host platforms."""
        assert AudioUnitPlatform().extension == ".component"
        assert sys_platform.system()


class TestAudioUnitProjectGeneration:
    """Test AudioUnit project generation."""

    def test_generate_au_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate an AU project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "CMakeLists.txt").is_file()
        assert (project_dir / "Info.plist").is_file()
        assert (project_dir / "gen_ext_au.cpp").is_file()
        assert (project_dir / "_ext_au.cpp").is_file()
        assert (project_dir / "_ext_au.h").is_file()
        assert (project_dir / "gen_ext_common_au.h").is_file()
        assert (project_dir / "au_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "gen").is_dir()
        assert (project_dir / "build").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_au_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate an AU project with one buffer."""
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
        """The generated CMakeLists should reference the expected AU pieces."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "set(PROJECT_NAME testverb)" in cmake
        assert "AU_EXT_NAME=testverb" in cmake
        assert "GEN_EXPORTED_NAME=gen_exported" in cmake
        assert "GENLIB_USE_FLOAT32" in cmake
        assert "AudioToolbox" in cmake
        assert "CoreFoundation" in cmake
        assert ".component" in cmake

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
