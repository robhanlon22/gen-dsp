"""Tests for the VCV Rack module platform."""

import json
from pathlib import Path

import pytest

from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.platforms import (
    PLATFORM_REGISTRY,
    VcvRackPlatform,
    get_platform,
)
from gen_dsp.platforms.base import PluginCategory
from gen_dsp.platforms.vcvrack import (
    RACK_SDK_VERSION,
    _get_default_rack_sdk_dir,
    _get_rack_sdk_url,
    _resolve_rack_dir,
)

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2


def test_registry_contains_vcvrack() -> None:
    """VCV Rack is registered as a platform."""
    assert PLATFORM_REGISTRY["vcvrack"] == VcvRackPlatform


def test_get_platform_vcvrack() -> None:
    """The platform factory returns a VcvRackPlatform instance."""
    platform = get_platform("vcvrack")
    assert isinstance(platform, VcvRackPlatform)
    assert platform.name == "vcvrack"


def test_vcvrack_extension() -> None:
    """VCV Rack builds shared libraries."""
    assert VcvRackPlatform().extension in (".dylib", ".so", ".dll")


def test_vcvrack_build_instructions() -> None:
    """VCV Rack exposes a non-empty build instruction list."""
    instructions = VcvRackPlatform().get_build_instructions()
    assert isinstance(instructions, list)
    assert instructions
    assert any("make" in instruction for instruction in instructions)


def test_detect_plugin_type() -> None:
    """Plugin category detection matches the input count."""
    assert PluginCategory.from_num_inputs(NUM_0) == PluginCategory.GENERATOR
    assert PluginCategory.from_num_inputs(NUM_1) == PluginCategory.EFFECT
    assert PluginCategory.from_num_inputs(NUM_2) == PluginCategory.EFFECT


def test_sdk_url_returns_string() -> None:
    """The SDK URL is derived from the platform and version."""
    url = _get_rack_sdk_url()
    assert RACK_SDK_VERSION in url
    assert "Rack-SDK" in url


def test_default_rack_sdk_dir_in_cache() -> None:
    """The default SDK path is rooted in the local cache."""
    sdk_dir = _get_default_rack_sdk_dir()
    assert "rack-sdk-src" in str(sdk_dir)
    assert "Rack-SDK" in str(sdk_dir)


def test_resolve_rack_dir_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RACK_DIR and GEN_DSP_CACHE_DIR overrides are honored."""
    monkeypatch.setenv("RACK_DIR", "/custom/sdk")
    assert str(_resolve_rack_dir()) == "/custom/sdk"

    monkeypatch.delenv("RACK_DIR", raising=False)
    cache_dir = tmp_path / "mycache"
    monkeypatch.setenv("GEN_DSP_CACHE_DIR", str(cache_dir))
    assert str(_resolve_rack_dir()) == f"{cache_dir}/rack-sdk-src/Rack-SDK"


def test_vcvrack_project_generation_smoke(
    gigaverb_export: Path,
    tmp_project: Path,
) -> None:
    """A VCV Rack project can be generated from a real gen export."""
    parser = GenExportParser(gigaverb_export)
    export_info = parser.parse()

    config = ProjectConfig(name="testverb", platform="vcvrack")
    generator = ProjectGenerator(export_info, config)
    project_dir = generator.generate(tmp_project)

    assert project_dir.is_dir()
    assert (project_dir / "Makefile").is_file()
    assert (project_dir / "plugin.json").is_file()
    assert (project_dir / "gen_ext_vcvrack.cpp").is_file()
    assert (project_dir / "res" / "testverb.svg").is_file()


def test_vcvrack_plugin_json_content(
    gigaverb_export: Path,
    tmp_project: Path,
) -> None:
    """Plugin metadata reflects the project name."""
    parser = GenExportParser(gigaverb_export)
    export_info = parser.parse()

    config = ProjectConfig(name="testverb", platform="vcvrack")
    generator = ProjectGenerator(export_info, config)
    project_dir = generator.generate(tmp_project)

    plugin_json = json.loads((project_dir / "plugin.json").read_text())
    assert plugin_json["slug"] == "testverb"
    assert plugin_json["modules"][0]["slug"] == "testverb"
