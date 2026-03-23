"""Tests for the Circle (Raspberry Pi bare metal) platform."""

from pathlib import Path

import pytest

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import (
    PLATFORM_REGISTRY,
    CirclePlatform,
    get_platform,
)
from gen_dsp.platforms.circle import (
    CIRCLE_BOARDS,
    CIRCLE_VERSION,
    _get_audio_base_class,
    _get_audio_include,
    _get_audio_label,
    _get_boot_config,
    _get_extra_libs,
    _resolve_circle_dir,
)

NUM_1 = 1
NUM_4 = 4


def test_registry_contains_circle() -> None:
    """Circle is registered as a platform."""
    assert PLATFORM_REGISTRY["circle"] == CirclePlatform


def test_get_platform_circle() -> None:
    """The platform factory returns a CirclePlatform instance."""
    platform = get_platform("circle")
    assert isinstance(platform, CirclePlatform)
    assert platform.name == "circle"


def test_circle_extension() -> None:
    """Circle builds image files."""
    assert CirclePlatform().extension == ".img"


def test_circle_build_instructions() -> None:
    """Circle exposes a non-empty build instruction list."""
    instructions = CirclePlatform().get_build_instructions()
    assert isinstance(instructions, list)
    assert instructions
    assert any("make" in instruction for instruction in instructions)


def test_resolve_circle_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CIRCLE_DIR takes priority over derived cache paths."""
    monkeypatch.setenv("CIRCLE_DIR", "/custom/circle")
    assert _resolve_circle_dir() == Path("/custom/circle")

    monkeypatch.delenv("CIRCLE_DIR", raising=False)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("GEN_DSP_CACHE_DIR", str(cache_dir))
    assert str(_resolve_circle_dir()) == f"{cache_dir}/circle-src/circle"


@pytest.mark.parametrize(
    ("device", "expected_header", "expected_class", "expected_label"),
    [
        ("i2s", "i2ssoundbasedevice.h", "CI2SSoundBaseDevice", "I2S"),
        ("pwm", "pwmsoundbasedevice.h", "CPWMSoundBaseDevice", "PWM"),
        ("hdmi", "hdmisoundbasedevice.h", "CHDMISoundBaseDevice", "HDMI"),
        ("usb", "usbsoundbasedevice.h", "CUSBSoundBaseDevice", "USB"),
    ],
)
def test_audio_device_helpers(
    device: str,
    expected_header: str,
    expected_class: str,
    expected_label: str,
) -> None:
    """Audio helper functions map device names consistently."""
    assert expected_header in _get_audio_include(device)
    assert _get_audio_base_class(device) == expected_class
    assert _get_audio_label(device) == expected_label


@pytest.mark.parametrize(
    ("device", "expected"),
    [("usb", "libusb.a"), ("i2s", ""), ("pwm", "")],
)
def test_extra_libs(device: str, expected: str) -> None:
    """Only USB audio needs libusb."""
    assert _get_extra_libs(device) == expected


@pytest.mark.parametrize(
    ("device", "expected"),
    [("i2s", "dtparam=i2s=on"), ("pwm", "PWM"), ("hdmi", "HDMI"), ("usb", "USB")],
)
def test_boot_config(device: str, expected: str) -> None:
    """Boot configuration reflects the selected audio device."""
    assert expected in _get_boot_config(device)


def test_circle_version_is_string() -> None:
    """Circle version metadata is populated."""
    assert isinstance(CIRCLE_VERSION, str)
    assert "Step" in CIRCLE_VERSION


def test_circle_project_generation_smoke(
    gigaverb_export: Path, tmp_project: Path
) -> None:
    """A Circle project can be generated from a real gen export."""
    parser = GenExportParser(gigaverb_export)
    export_info = parser.parse()

    config = ProjectConfig(name="testverb", platform="circle")
    generator = ProjectGenerator(export_info, config)
    project_dir = generator.generate(tmp_project)

    assert project_dir.is_dir()
    assert (project_dir / "Makefile").is_file()
    assert (project_dir / "gen_ext_circle.cpp").is_file()
    assert (project_dir / "gen_buffer.h").is_file()
    assert "WRAPPER_BUFFER_COUNT 0" in (project_dir / "gen_buffer.h").read_text()


def test_circle_board_registry() -> None:
    """The Circle board registry contains the expected keys."""
    assert "pi3-i2s" in CIRCLE_BOARDS
    assert "pi4-usb" in CIRCLE_BOARDS
    assert CIRCLE_BOARDS["pi3-i2s"].audio_device == "i2s"
    assert CIRCLE_BOARDS["pi4-usb"].audio_device == "usb"
    assert CIRCLE_BOARDS["pi3-i2s"].rasppi == NUM_1
    assert CIRCLE_BOARDS["pi4-usb"].rasppi == NUM_4
