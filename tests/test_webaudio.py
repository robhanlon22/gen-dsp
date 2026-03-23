"""Tests for Web Audio (AudioWorklet + WASM) platform implementation."""

from pathlib import Path

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, WebAudioPlatform, get_platform

NUM_0 = 0


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="webaudio", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestWebAudioPlatform:
    """Test Web Audio platform registry and basic properties."""

    def test_registry_contains_webaudio(self) -> object:
        """The Web Audio platform should be registered."""
        assert "webaudio" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["webaudio"] == WebAudioPlatform

    def test_get_platform_webaudio(self) -> object:
        """`get_platform('webaudio')` should return the Web Audio platform."""
        platform = get_platform("webaudio")
        assert isinstance(platform, WebAudioPlatform)
        assert platform.name == "webaudio"

    def test_webaudio_extension(self) -> object:
        """The Web Audio extension should be `.wasm`."""
        platform = WebAudioPlatform()
        assert platform.extension == ".wasm"

    def test_webaudio_build_instructions(self) -> object:
        """Web Audio build instructions should reference `make`."""
        platform = WebAudioPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("make" in instr for instr in instructions)


class TestWebAudioProjectGeneration:
    """Test Web Audio project generation."""

    def test_generate_project_gigaverb(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate a Web Audio project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "Makefile").is_file()
        assert (project_dir / "gen_ext_webaudio.cpp").is_file()
        assert (project_dir / "_ext_webaudio.cpp").is_file()
        assert (project_dir / "_ext_webaudio.h").is_file()
        assert (project_dir / "gen_ext_common_webaudio.h").is_file()
        assert (project_dir / "webaudio_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "_processor.js").is_file()
        assert (project_dir / "index.html").is_file()
        assert (project_dir / "gen").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate a Web Audio project with one buffer."""
        project_dir = _generate_project(
            rampleplayer_export,
            tmp_project,
            "testsampler",
            buffers=["sample"],
        )

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 sample" in buffer_h

    def test_processor_js_has_param_descriptors(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """The generated processor script should describe its parameters."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        processor_js = (project_dir / "_processor.js").read_text()
        assert "PARAM_DESCRIPTORS" in processor_js
        assert "registerProcessor" in processor_js
        assert "TestverbProcessor" in processor_js
        assert "testverb" in processor_js
        assert "roomsize" in processor_js

    def test_makefile_has_emcc(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """The generated Makefile should reference `emcc`."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        makefile = (project_dir / "Makefile").read_text()
        assert "emcc" in makefile.lower()
        assert "GENLIB_USE_FLOAT32" in makefile
        assert "WEBAUDIO_EXT_NAME=testverb" in makefile
        assert "WASM=1" in makefile
        assert "MODULARIZE=1" in makefile
        assert "wa_create" in makefile
        assert "wa_perform" in makefile
