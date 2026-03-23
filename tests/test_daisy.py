"""Tests for the Daisy embedded platform."""

from pathlib import Path

import pytest

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import (
    PLATFORM_REGISTRY,
    DaisyPlatform,
    get_platform,
)
from gen_dsp.platforms.daisy import (
    DAISY_BOARDS,
    LIBDAISY_VERSION,
    _generate_main_loop_body,
    _get_default_libdaisy_dir,
    _resolve_libdaisy_dir,
)

NUM_2 = 2
NUM_4 = 4


def test_registry_contains_daisy() -> None:
    """Daisy is registered as a platform."""
    assert PLATFORM_REGISTRY["daisy"] == DaisyPlatform


def test_get_platform_daisy() -> None:
    """The platform factory returns a DaisyPlatform instance."""
    platform = get_platform("daisy")
    assert isinstance(platform, DaisyPlatform)
    assert platform.name == "daisy"


def test_daisy_extension() -> None:
    """Daisy builds binary images."""
    assert DaisyPlatform().extension == ".bin"


def test_daisy_build_instructions() -> None:
    """Daisy exposes a non-empty build instruction list."""
    instructions = DaisyPlatform().get_build_instructions()
    assert isinstance(instructions, list)
    assert instructions
    assert any("make" in instruction for instruction in instructions)


def test_default_libdaisy_dir_in_cache() -> None:
    """The default libDaisy path is derived from the cache root."""
    sdk_dir = _get_default_libdaisy_dir()
    assert "gen-dsp" in str(sdk_dir)
    assert "libdaisy-src" in str(sdk_dir)
    assert "libDaisy" in str(sdk_dir)


def test_resolve_libdaisy_dir_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LIBDAISY_DIR and GEN_DSP_CACHE_DIR overrides are honored."""
    monkeypatch.setenv("LIBDAISY_DIR", "/custom/libdaisy")
    assert _resolve_libdaisy_dir() == Path("/custom/libdaisy")

    monkeypatch.delenv("LIBDAISY_DIR", raising=False)
    cache_dir = tmp_path / "mycache"
    monkeypatch.setenv("GEN_DSP_CACHE_DIR", str(cache_dir))
    assert str(_resolve_libdaisy_dir()) == f"{cache_dir}/libdaisy-src/libDaisy"


def test_libdaisy_version_is_string() -> None:
    """The libDaisy version constant is populated."""
    assert isinstance(LIBDAISY_VERSION, str)
    assert LIBDAISY_VERSION.startswith("v")


def test_generate_main_loop_body_comments_and_knobs() -> None:
    """Main-loop code switches between comment-only and automap modes."""
    seed_body = _generate_main_loop_body(DAISY_BOARDS["seed"], num_params=8)
    pod_body = _generate_main_loop_body(DAISY_BOARDS["pod"], num_params=5)

    assert "ProcessAllControls" not in seed_body
    assert "No hardware knobs" in seed_body
    assert "hw.ProcessAllControls();" in pod_body
    assert "wrapper_set_param(genState, 1" in pod_body


def test_daisy_board_registry() -> None:
    """The board registry exposes the expected core variants."""
    assert "seed" in DAISY_BOARDS
    assert "pod" in DAISY_BOARDS
    assert "patch" in DAISY_BOARDS
    assert DAISY_BOARDS["patch"].hw_channels == NUM_4
    assert DAISY_BOARDS["seed"].hw_channels == NUM_2


def test_daisy_project_generation_smoke(
    gigaverb_export: Path,
    tmp_project: Path,
) -> None:
    """A Daisy project can be generated from a real gen export."""
    parser = GenExportParser(gigaverb_export)
    export_info = parser.parse()

    config = ProjectConfig(name="testverb", platform="daisy")
    generator = ProjectGenerator(export_info, config)
    project_dir = generator.generate(tmp_project)

    assert project_dir.is_dir()
    assert (project_dir / "Makefile").is_file()
    assert (project_dir / "gen_ext_daisy.cpp").is_file()
    assert (project_dir / "gen_buffer.h").is_file()
    assert "WRAPPER_BUFFER_COUNT 0" in (project_dir / "gen_buffer.h").read_text()
