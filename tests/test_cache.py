"""Tests for gen_dsp.core.cache module."""

import platform
from pathlib import Path

from gen_dsp.core.cache import get_cache_dir


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_returns_path(self) -> object:
        """get_cache_dir should return a Path."""
        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_contains_gen_dsp(self) -> object:
        """Cache paths should include the project name."""
        result = get_cache_dir()
        assert "gen-dsp" in result.parts

    def test_contains_fetchcontent(self) -> object:
        """Cache dir name should be fetchcontent."""
        result = get_cache_dir()
        assert result.name == "fetchcontent"

    def test_macos_path(self, monkeypatch: object) -> object:
        """MacOS should use the Library/Caches path layout."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        result = get_cache_dir()
        assert "Library" in result.parts
        assert "Caches" in result.parts

    def test_linux_default_path(self, monkeypatch: object) -> object:
        """Linux should default to ~/.cache when XDG is unset."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = get_cache_dir()
        assert ".cache" in result.parts

    def test_linux_xdg_override(self, monkeypatch: object, tmp_path: object) -> object:
        """Linux should honor XDG_CACHE_HOME overrides."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "custom_cache"))
        result = get_cache_dir()
        assert str(result).startswith(str(tmp_path / "custom_cache"))
        assert "gen-dsp" in result.parts

    def test_windows_path(self, monkeypatch: object, tmp_path: object) -> object:
        """Windows should use LOCALAPPDATA for the cache path."""
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
        result = get_cache_dir()
        assert str(result).startswith(str(tmp_path / "AppData" / "Local"))
        assert "gen-dsp" in result.parts
