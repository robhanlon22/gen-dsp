"""Tests for ChucK chugin platform implementation."""

import platform as sys_platform
import shutil
import subprocess
from pathlib import Path

import pytest

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import (
    PLATFORM_REGISTRY,
    ChuckPlatform,
    get_platform,
)

NUM_0 = 0
NUM_1 = 1


def _resolve_executable(name: str) -> str:
    """Return the absolute path for an executable."""
    path = shutil.which(name)
    if path is None:
        message = f"{name} not found"
        raise RuntimeError(message)
    return path


_SUBPROCESS_RUN = subprocess.run


# Skip integration tests if no C++ compiler available
_has_cxx = shutil.which("g++") is not None or shutil.which("clang++") is not None
_has_make = shutil.which("make") is not None
_has_chuck = shutil.which("chuck") is not None
_can_build = _has_cxx and _has_make
_skip_no_toolchain = pytest.mark.skipif(
    not _can_build, reason="C++ compiler or make not found"
)
_skip_no_chuck = pytest.mark.skipif(
    not (_can_build and _has_chuck), reason="C++ toolchain or chuck not found"
)


def _build_chuck_script_lines(
    class_name: str,
    expected_params: int,
    buffers: dict[str, str] | None,
    *,
    expect_audio: bool,
    phasor_input: bool,
) -> list[str]:
    """Build the ChucK test script for a chugin."""
    lines = [f'@import "{class_name}"']
    if expect_audio:
        if phasor_input:
            lines += [
                f"Phasor src => {class_name} eff => Gain g => blackhole;",
                "10.0 => src.freq;",
            ]
        else:
            lines.append(f"Noise src => {class_name} eff => Gain g => blackhole;")
    else:
        lines.append(f"{class_name} eff => blackhole;")

    if buffers:
        for buf_name, wav_file in buffers.items():
            lines += [
                f'eff.loadBuffer("{buf_name}", "{wav_file}") => int frames;',
                f'<<< "LOADED_{buf_name}", frames >>>;',
            ]

    lines += [
        "eff.numParams() => int np;",
        '<<< "PARAMS", np >>>;',
    ]
    if expected_params > NUM_0:
        lines += [
            "eff.paramName(0) => string pname;",
            '<<< "PNAME", pname >>>;',
        ]
    if expect_audio:
        lines += [
            "50::ms => now;",
            "0.0 => float energy;",
            "repeat(2205) {",
            "    1::samp => now;",
            "    g.last() * g.last() +=> energy;",
            "}",
            'if (energy > 0.0) <<< "AUDIO_OK" >>>;',
            'else <<< "AUDIO_FAIL", energy >>>;',
        ]
    else:
        lines.append("100::ms => now;")

    lines.append('<<< "DONE" >>>;')
    return lines


def _assert_chuck_output(
    output: str,
    expected_params: int,
    buffers: dict[str, str] | None,
    *,
    expect_audio: bool,
) -> None:
    """Validate the ChucK runtime output."""
    assert "PARAMS" in output
    assert str(expected_params) in output
    if buffers:
        for buf_name in buffers:
            assert f"LOADED_{buf_name}" in output, (
                f"Buffer '{buf_name}' not loaded:\n{output}"
            )
    if expect_audio:
        assert "AUDIO_OK" in output, f"No audio output detected:\n{output}"
    assert "DONE" in output


def _validate_chugin(
    project_dir: Path,
    class_name: str,
    expected_params: int,
    **options: object,
) -> None:
    """Load a built chugin in ChucK and validate it works."""
    if not _has_chuck:
        return
    expect_audio = bool(options.get("expect_audio", False))
    buffers = options.get("buffers")
    phasor_input = bool(options.get("phasor_input", False))

    test_ck = project_dir / "test.ck"
    test_ck.write_text(
        "\n".join(
            _build_chuck_script_lines(
                class_name,
                expected_params,
                buffers,
                expect_audio=expect_audio,
                phasor_input=phasor_input,
            )
        )
        + "\n"
    )

    chuck = _resolve_executable("chuck")
    result = _SUBPROCESS_RUN(
        [chuck, "--chugin-path:.", "--silent", "test.ck"],
        check=False,
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == NUM_0, (
        f"chuck failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    _assert_chuck_output(
        result.stderr,
        expected_params,
        buffers,
        expect_audio=expect_audio,
    )


class TestChuckPlatform:
    """Test ChucK platform registry and basic properties."""

    def test_registry_contains_chuck(self) -> object:
        """Test test registry contains chuck."""
        assert "chuck" in PLATFORM_REGISTRY
        assert PLATFORM_REGISTRY["chuck"] == ChuckPlatform

    def test_get_platform_chuck(self) -> object:
        """Test test get platform chuck."""
        platform = get_platform("chuck")
        assert isinstance(platform, ChuckPlatform)
        assert platform.name == "chuck"

    def test_chuck_extension(self) -> object:
        """Test test chuck extension."""
        platform = ChuckPlatform()
        assert platform.extension == ".chug"

    def test_chuck_build_instructions(self) -> object:
        """Test test chuck build instructions."""
        platform = ChuckPlatform()
        instructions = platform.get_build_instructions()
        assert isinstance(instructions, list)
        assert len(instructions) > NUM_0
        assert any("make" in instr for instr in instructions)


class TestChuckProjectGeneration:
    """Test ChucK chugin project generation."""

    def test_generate_chuck_project_no_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test generating ChucK project without buffers."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="testverb", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        # Check directory was created
        assert project_dir.is_dir()

        # Check required files exist
        assert (project_dir / "makefile").is_file()
        assert (project_dir / "makefile.mac").is_file()
        assert (project_dir / "makefile.linux").is_file()
        assert (project_dir / "gen_ext_chuck.cpp").is_file()
        assert (project_dir / "_ext_chuck.cpp").is_file()
        assert (project_dir / "_ext_chuck.h").is_file()
        assert (project_dir / "gen_ext_common_chuck.h").is_file()
        assert (project_dir / "chuck_buffer.h").is_file()
        assert (project_dir / "gen_buffer.h").is_file()
        assert (project_dir / "chuck" / "include" / "chugin.h").is_file()
        assert (project_dir / "gen").is_dir()

        # Check gen_buffer.h has 0 buffers
        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 0" in buffer_h

    def test_generate_chuck_project_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """Test generating ChucK project with buffers."""
        parser = GenExportParser(rampleplayer_export)
        export_info = parser.parse()

        config = ProjectConfig(
            name="testsampler",
            platform="chuck",
            buffers=["sample"],
        )
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        # Check gen_buffer.h has buffer configured
        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 sample" in buffer_h

    def test_generate_chuck_project_multiple_buffers(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test generating ChucK project with multiple buffers."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(
            name="multibuf",
            platform="chuck",
            buffers=["buf1", "buf2", "buf3"],
        )
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        buffer_h = (project_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 3" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 buf1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_1 buf2" in buffer_h
        assert "WRAPPER_BUFFER_NAME_2 buf3" in buffer_h

    def test_makefile_content(self, gigaverb_export: Path, tmp_project: Path) -> object:
        """Test test makefile content."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="testverb", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        makefile = (project_dir / "makefile").read_text()
        assert "CHUGIN_NAME=Testverb" in makefile
        assert "CHUCK_EXT_NAME=Testverb" in makefile
        assert "GEN_EXPORTED_NAME=gen_exported" in makefile
        assert "GENLIB_USE_FLOAT32" in makefile

    def test_generate_copies_gen_export(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Test that gen~ export is copied to project."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="test", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        gen_dir = project_dir / "gen"
        assert gen_dir.is_dir()
        assert (gen_dir / "gen_exported.cpp").is_file()
        assert (gen_dir / "gen_exported.h").is_file()
        assert (gen_dir / "gen_dsp").is_dir()
        assert (gen_dir / "gen_dsp" / "genlib.cpp").is_file()


class TestChuckBuildIntegration:
    """
    Integration tests that generate and compile a chugin.

    Skipped when no C++ compiler or make is available.
    """

    @_skip_no_toolchain
    def test_build_chugin_no_buffers(
        self, gigaverb_export: Path, tmp_path: Path
    ) -> object:
        """Generate and compile a chugin from gigaverb (no buffers)."""
        project_dir = tmp_path / "gigaverb_chuck"
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="gigaverb", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        generator.generate(project_dir)

        # Determine build target
        target = "mac" if sys_platform.system().lower() == "darwin" else "linux"
        make = _resolve_executable("make")

        result = _SUBPROCESS_RUN(
            [make, target],
            check=False,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == NUM_0, (
            f"make {target} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Verify .chug was produced
        chug_files = list(project_dir.glob("*.chug"))
        assert len(chug_files) == NUM_1
        assert chug_files[NUM_0].name == "Gigaverb.chug"
        assert chug_files[NUM_0].stat().st_size > NUM_0

        _validate_chugin(project_dir, "Gigaverb", 8, expect_audio=True)

    @_skip_no_toolchain
    def test_build_chugin_with_buffers(
        self, rampleplayer_export: Path, tmp_path: Path
    ) -> object:
        """Generate and compile a chugin from RamplePlayer (has buffers)."""
        project_dir = tmp_path / "rampleplayer_chuck"
        parser = GenExportParser(rampleplayer_export)
        export_info = parser.parse()

        config = ProjectConfig(
            name="rampleplayer",
            platform="chuck",
            buffers=["sample"],
        )
        generator = ProjectGenerator(export_info, config)
        generator.generate(project_dir)

        target = "mac" if sys_platform.system().lower() == "darwin" else "linux"
        make = _resolve_executable("make")

        result = _SUBPROCESS_RUN(
            [make, target],
            check=False,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == NUM_0, (
            f"make {target} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        chug_files = list(project_dir.glob("*.chug"))
        assert len(chug_files) == NUM_1
        assert chug_files[NUM_0].name == "Rampleplayer.chug"

        # Copy test audio for buffer loading
        wav_src = Path(__file__).parent / "data" / "amen.wav"
        shutil.copy2(wav_src, project_dir / "amen.wav")

        _validate_chugin(
            project_dir,
            "Rampleplayer",
            0,
            expect_audio=True,
            buffers={"sample": "amen.wav"},
            phasor_input=True,
        )

    @_skip_no_toolchain
    def test_build_chugin_spectraldelayfb(
        self, spectraldelayfb_export: Path, tmp_path: Path
    ) -> object:
        """Generate and compile a chugin from spectraldelayfb (3in/2out, no buffers)."""
        project_dir = tmp_path / "spectraldelayfb_chuck"
        parser = GenExportParser(spectraldelayfb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="spectraldelayfb", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        generator.generate(project_dir)

        target = "mac" if sys_platform.system().lower() == "darwin" else "linux"
        make = _resolve_executable("make")

        result = _SUBPROCESS_RUN(
            [make, target],
            check=False,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == NUM_0, (
            f"make {target} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        chug_files = list(project_dir.glob("*.chug"))
        assert len(chug_files) == NUM_1
        assert chug_files[NUM_0].name == "Spectraldelayfb.chug"
        assert chug_files[NUM_0].stat().st_size > NUM_0

        _validate_chugin(project_dir, "Spectraldelayfb", 0, expect_audio=True)

    @_skip_no_toolchain
    @_skip_no_chuck
    def test_load_chugin_in_chuck(
        self, gigaverb_export: Path, tmp_path: Path
    ) -> object:
        """Build a chugin and verify ChucK can load and run it."""
        project_dir = tmp_path / "gigaverb_load"
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="gigaverb", platform="chuck")
        generator = ProjectGenerator(export_info, config)
        generator.generate(project_dir)

        target = "mac" if sys_platform.system().lower() == "darwin" else "linux"
        make = _resolve_executable("make")
        build = _SUBPROCESS_RUN(
            [make, target],
            check=False,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert build.returncode == NUM_0, (
            f"make {target} failed:\nstderr: {build.stderr}"
        )

        # Write a ChucK script that exercises the chugin
        test_ck = project_dir / "test.ck"
        test_ck.write_text(
            '@import "Gigaverb"\n'
            "Gigaverb eff => dac;\n"
            "eff.numParams() => int n;\n"
            '<<< "PARAMS", n >>>;\n'
            'eff.param("roomsize", 42.0);\n'
            'eff.param("roomsize") => float v;\n'
            '<<< "ROOMSIZE", v >>>;\n'
            "100::ms => now;\n"
            '<<< "DONE" >>>;\n'
        )

        chuck = _resolve_executable("chuck")
        result = _SUBPROCESS_RUN(
            [chuck, "--chugin-path:.", "--silent", "test.ck"],
            check=False,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == NUM_0, (
            f"chuck failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # ChucK <<< >>> output goes to stderr
        output = result.stderr
        assert "PARAMS" in output
        assert "8" in output  # gigaverb has 8 params
        assert "ROOMSIZE" in output
        assert "42.0" in output
        assert "DONE" in output
