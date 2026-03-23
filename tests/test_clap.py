"""Tests for CLAP plugin platform implementation."""

from pathlib import Path

from gen_dsp.core.manifest import Manifest, ParamInfo
from gen_dsp.core.midi import detect_midi_mapping
from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import PLATFORM_REGISTRY, get_platform
from gen_dsp.platforms.clap import ClapPlatform

NUM_0 = 0
NUM_1 = 1


def _generate_project(
    export_path: Path, tmp_project: Path, name: str, **kwargs: object
) -> Path:
    parser = GenExportParser(export_path)
    export_info = parser.parse()
    config = ProjectConfig(name=name, platform="clap", **kwargs)
    generator = ProjectGenerator(export_info, config)
    return generator.generate(tmp_project)


class TestClapPlatform:
    """Test CLAP platform registry and basic properties."""

    def test_registry_contains_clap(self) -> object:
        """The CLAP platform should be registered."""
        assert "clap" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["clap"] == ClapPlatform

    def test_get_platform_clap(self) -> object:
        """`get_platform('clap')` should return the CLAP platform."""
        platform = get_platform("clap")
        assert isinstance(platform, ClapPlatform)
        assert platform.name == "clap"

    def test_clap_extension(self) -> object:
        """The CLAP extension should be `.clap`."""
        platform = ClapPlatform()
        assert platform.extension == ".clap"

    def test_clap_build_instructions(self) -> object:
        """CLAP build instructions should reference `cmake`."""
        platform = ClapPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("cmake" in instr for instr in instructions)


class TestClapProjectGeneration:
    """Test CLAP project generation."""

    def test_generate_clap_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Gigaverb should generate a CLAP project without buffers."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        assert project_dir.is_dir()
        assert (project_dir / "CMakeLists.txt").is_file()
        assert (project_dir / "gen_ext_clap.cpp").is_file()
        assert (project_dir / "_ext_clap.cpp").is_file()
        assert (project_dir / "_ext_clap.h").is_file()
        assert (project_dir / "gen_ext_common_clap.h").is_file()
        assert (project_dir / "clap_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "gen").is_dir()
        assert (project_dir / "build").is_dir()

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_clap_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """RamplePlayer should generate a CLAP project with one buffer."""
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
        """The generated CMakeLists should reference the expected CLAP pieces."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")

        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "set(PROJECT_NAME testverb)" in cmake
        assert "CLAP_EXT_NAME=testverb" in cmake
        assert "GEN_EXPORTED_NAME=gen_exported" in cmake
        assert "GENLIB_USE_FLOAT32" in cmake
        assert "FetchContent_Declare" in cmake
        assert "free-audio/clap" in cmake
        assert ".clap" in cmake

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

    def test_cmakelists_shared_cache_on_by_default(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Shared cache should be enabled by default."""
        project_dir = _generate_project(gigaverb_export, tmp_project, "testverb")
        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "elseif(ON)" in cmake
        assert "GEN_DSP_CACHE_DIR" in cmake

    def test_cmakelists_shared_cache_off(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Shared cache should be disabled when requested."""
        project_dir = _generate_project(
            gigaverb_export,
            tmp_project,
            "testverb",
            shared_cache=False,
        )
        cmake = (project_dir / "CMakeLists.txt").read_text()
        assert "elseif(OFF)" in cmake
        assert "GEN_DSP_CACHE_DIR" in cmake


class TestClapMidiGeneration:
    """Test MIDI compile definitions in generated CLAP projects."""

    def test_cmakelists_midi_defines_with_explicit_mapping(
        self, tmp_path: Path
    ) -> object:
        """Explicit MIDI mapping should emit the expected CLAP defines."""
        output_dir = tmp_path / "midi_explicit"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
                ParamInfo(
                    index=1,
                    name="freq",
                    has_minmax=True,
                    min=20.0,
                    max=20000.0,
                    default=440.0,
                ),
                ParamInfo(
                    index=2, name="vel", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
            ],
        )

        config = ProjectConfig(
            name="testsynth",
            platform="clap",
            midi_gate="gate",
            midi_freq="freq",
            midi_vel="vel",
        )
        config.midi_mapping = detect_midi_mapping(
            manifest,
            midi_gate=config.midi_gate,
            midi_freq=config.midi_freq,
            midi_vel=config.midi_vel,
        )

        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "MIDI_ENABLED=1" in cmake
        assert "MIDI_GATE_IDX=0" in cmake
        assert "MIDI_FREQ_IDX=1" in cmake
        assert "MIDI_VEL_IDX=2" in cmake
        assert "MIDI_FREQ_UNIT_HZ=1" in cmake

    def test_cmakelists_midi_gate_only(self, tmp_path: Path) -> object:
        """Gate-only mapping should not emit FREQ or VEL defines."""
        output_dir = tmp_path / "midi_gate_only"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
                ParamInfo(
                    index=1,
                    name="cutoff",
                    has_minmax=True,
                    min=20.0,
                    max=20000.0,
                    default=1000.0,
                ),
            ],
        )

        config = ProjectConfig(name="testsynth", platform="clap")
        config.midi_mapping = detect_midi_mapping(manifest)
        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "MIDI_ENABLED=1" in cmake
        assert "MIDI_GATE_IDX=0" in cmake
        assert "MIDI_FREQ_IDX" not in cmake
        assert "MIDI_VEL_IDX" not in cmake

    def test_cmakelists_midi_freq_unit_midi(self, tmp_path: Path) -> object:
        """`midi_freq_unit='midi'` should produce `MIDI_FREQ_UNIT_HZ=0`."""
        output_dir = tmp_path / "midi_freq_unit"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
                ParamInfo(
                    index=1,
                    name="freq",
                    has_minmax=True,
                    min=0.0,
                    max=127.0,
                    default=60.0,
                ),
            ],
        )

        config = ProjectConfig(name="testsynth", platform="clap", midi_freq_unit="midi")
        config.midi_mapping = detect_midi_mapping(manifest, midi_freq_unit="midi")
        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "MIDI_FREQ_UNIT_HZ=0" in cmake

    def test_cmakelists_polyphony_defines(self, tmp_path: Path) -> object:
        """`num_voices=8` should emit the polyphony define."""
        output_dir = tmp_path / "poly_clap"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
                ParamInfo(
                    index=1,
                    name="freq",
                    has_minmax=True,
                    min=20.0,
                    max=20000.0,
                    default=440.0,
                ),
            ],
        )

        config = ProjectConfig(
            name="testsynth",
            platform="clap",
            midi_gate="gate",
            midi_freq="freq",
            num_voices=8,
        )
        config.midi_mapping = detect_midi_mapping(
            manifest,
            midi_gate=config.midi_gate,
            midi_freq=config.midi_freq,
        )
        config.midi_mapping.num_voices = config.num_voices

        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "NUM_VOICES=8" in cmake
        assert "MIDI_ENABLED=1" in cmake

    def test_voice_alloc_header_copied(self, tmp_path: Path) -> object:
        """`voice_alloc.h` is copied when `num_voices > 1`."""
        output_dir = tmp_path / "poly_header"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
            ],
        )

        config = ProjectConfig(
            name="testsynth", platform="clap", midi_gate="gate", num_voices=4
        )
        config.midi_mapping = detect_midi_mapping(manifest, midi_gate=config.midi_gate)
        config.midi_mapping.num_voices = config.num_voices
        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        assert (output_dir / "voice_alloc.h").is_file()

    def test_no_voice_alloc_header_mono(self, tmp_path: Path) -> object:
        """`voice_alloc.h` is not copied when `num_voices=1`."""
        output_dir = tmp_path / "mono_header"
        output_dir.mkdir()

        platform = ClapPlatform()
        manifest = Manifest(
            gen_name="test_synth",
            num_inputs=0,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0, name="gate", has_minmax=True, min=0.0, max=1.0, default=0.0
                ),
            ],
        )

        config = ProjectConfig(name="testsynth", platform="clap", midi_gate="gate")
        config.midi_mapping = detect_midi_mapping(manifest, midi_gate=config.midi_gate)
        platform.generate_project(manifest, output_dir, "testsynth", config=config)

        assert not (output_dir / "voice_alloc.h").exists()
