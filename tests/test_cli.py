"""Tests for gen_dsp.cli module."""

import json
from pathlib import Path

import pytest

from gen_dsp.cli import main
from gen_dsp.platforms import list_platforms

NUM_0 = 0
NUM_1 = 1


class TestVersionAndHelp:
    """Tests for --version and --help."""

    def test_version_flag(self, capsys: object) -> object:
        """Test test version flag."""
        result = main(["--version"])
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "gen-dsp" in captured.out

    def test_short_version_flag(self, capsys: object) -> object:
        """Test test short version flag."""
        result = main(["-V"])
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "gen-dsp" in captured.out

    def test_help_flag(self, capsys: object) -> object:
        """Test test help flag."""
        result = main(["--help"])
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "gen-dsp" in captured.out

    def test_no_args_shows_help(self, capsys: object) -> object:
        """Test test no args shows help."""
        result = main([])
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "gen-dsp" in captured.out


class TestDefaultCommand:
    """Tests for the default command (positional source)."""

    def test_dry_run_export(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test dry run export."""
        output_dir = tmp_path / "output"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--dry-run",
            ]
        )

        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would create project" in captured.out
        assert not output_dir.exists()

    def test_creates_project(self, gigaverb_export: Path, tmp_path: Path) -> object:
        """Test test creates project."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
            ]
        )

        assert result == NUM_0
        assert output_dir.is_dir()
        assert (output_dir / "Makefile").is_file()
        assert (output_dir / "gen").is_dir()

    def test_with_buffers(self, gigaverb_export: Path, tmp_path: Path) -> object:
        """Test test with buffers."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--buffers",
                "buf1",
                "buf2",
            ]
        )

        assert result == NUM_0
        buffer_h = (output_dir / "gen_buffer.h").read_text()
        assert "WRAPPER_BUFFER_COUNT 2" in buffer_h
        assert "WRAPPER_BUFFER_NAME_0 buf1" in buffer_h
        assert "WRAPPER_BUFFER_NAME_1 buf2" in buffer_h

    def test_shared_cache_on_by_default(
        self, gigaverb_export: Path, tmp_path: Path
    ) -> object:
        """Test test shared cache on by default."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "clap",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
            ]
        )

        assert result == NUM_0
        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "elseif(ON)" in cmake

    def test_no_shared_cache_disables(
        self, gigaverb_export: Path, tmp_path: Path
    ) -> object:
        """Test test no shared cache disables."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "clap",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--no-shared-cache",
            ]
        )

        assert result == NUM_0
        cmake = (output_dir / "CMakeLists.txt").read_text()
        assert "elseif(OFF)" in cmake

    def test_board_rejects_non_daisy(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test board rejects non daisy."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--board",
                "pod",
            ]
        )

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "--board" in captured.err

    def test_invalid_name(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test invalid name."""
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "123invalid",
                "-o",
                str(tmp_path / "output"),
                "--no-build",
            ]
        )

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "not a valid C identifier" in captured.err

    def test_invalid_export_path(self, tmp_path: Path, capsys: object) -> object:
        """Test test invalid export path."""
        result = main(
            [
                str(tmp_path / "nonexistent"),
                "-p",
                "pd",
            ]
        )

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestAutoDetect:
    """Tests for source type auto-detection."""

    def test_detects_directory(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test detects directory."""
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-o",
                str(tmp_path / "out"),
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would create project" in captured.out
        assert "Export:" in captured.out

    def test_detects_gdsp_file(self, tmp_path: Path, capsys: object) -> object:
        """Test test detects gdsp file."""
        pytest.importorskip("pydantic")
        graph_file = tmp_path / "lowpass.gdsp"
        graph_file.write_text(
            """
            graph lowpass {
                in input
                out output = filt
                param freq 20..20000 = 1000
                filt = onepole(input, freq / 44100)
            }
            """
        )
        result = main(
            [
                str(graph_file),
                "-p",
                "chuck",
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "dsp-graph" in captured.out

    def test_detects_json_file(self, tmp_path: Path, capsys: object) -> object:
        """Test test detects json file."""
        pytest.importorskip("pydantic")
        graph_file = tmp_path / "test.json"
        data = {
            "name": "test_graph",
            "inputs": [{"id": "in1"}],
            "outputs": [{"id": "out1", "source": "scaled"}],
            "params": [{"name": "gain", "min": 0.0, "max": 2.0, "default": 1.0}],
            "nodes": [{"id": "scaled", "op": "mul", "a": "in1", "b": "gain"}],
        }
        graph_file.write_text(json.dumps(data))
        result = main(
            [
                str(graph_file),
                "-p",
                "pd",
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "dsp-graph" in captured.out

    def test_unrecognized_source(self, tmp_path: Path, capsys: object) -> object:
        """Test test unrecognized source."""
        bad_file = tmp_path / "something.txt"
        bad_file.write_text("hello")
        result = main(
            [
                str(bad_file),
                "-p",
                "pd",
            ]
        )
        assert result == NUM_1
        captured = capsys.readouterr()
        assert "unrecognized" in captured.err.lower() or "Error" in captured.err


class TestDetectCommand:
    """Tests for detect command."""

    def test_detect_text_output(self, gigaverb_export: Path, capsys: object) -> object:
        """Test test detect text output."""
        result = main(["detect", str(gigaverb_export)])

        assert result == NUM_0
        captured = capsys.readouterr()
        assert "gen_exported" in captured.out
        assert "Signal inputs:" in captured.out
        assert "Signal outputs:" in captured.out

    def test_detect_json_output(self, gigaverb_export: Path, capsys: object) -> object:
        """Test test detect json output."""
        result = main(["detect", str(gigaverb_export), "--json"])

        assert result == NUM_0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "gen_exported"
        assert "num_inputs" in data
        assert "num_outputs" in data
        assert "buffers" in data

    def test_detect_with_buffers(
        self, rampleplayer_export: Path, capsys: object
    ) -> object:
        """Test test detect with buffers."""
        result = main(["detect", str(rampleplayer_export), "--json"])

        assert result == NUM_0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "sample" in data["buffers"]

    def test_detect_invalid_path(self, tmp_path: Path, capsys: object) -> object:
        """Test test detect invalid path."""
        result = main(["detect", str(tmp_path / "nonexistent")])

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestPatchCommand:
    """Tests for patch command."""

    def test_patch_dry_run(self, gigaverb_export: Path) -> object:
        """Test test patch dry run."""
        result = main(["patch", str(gigaverb_export), "--dry-run"])

        assert result == NUM_0

    def test_patch_invalid_path(self, tmp_path: Path, capsys: object) -> object:
        """Test test patch invalid path."""
        result = main(["patch", str(tmp_path / "nonexistent")])

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "not found" in captured.err


class TestBuildCommand:
    """Tests for build command."""

    def test_build_invalid_path(self, tmp_path: Path, capsys: object) -> object:
        """Test test build invalid path."""
        result = main(["build", str(tmp_path / "nonexistent")])

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_build_no_makefile(self, tmp_path: Path, capsys: object) -> object:
        """Test test build no makefile."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = main(["build", str(empty_dir)])

        assert result == NUM_1
        captured = capsys.readouterr()
        assert "Makefile" in captured.err or "Error" in captured.err


class TestListCommand:
    """Tests for list command."""

    def test_list_shows_all_platforms(self, capsys: object) -> object:
        """Test test list shows all platforms."""
        result = main(["list"])
        assert result == NUM_0
        captured = capsys.readouterr()
        for platform_name in list_platforms():
            assert platform_name in captured.out

    def test_list_output_one_per_line(self, capsys: object) -> object:
        """Test test list output one per line."""
        result = main(["list"])
        assert result == NUM_0
        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        assert len(lines) == len(list_platforms())


class TestCacheCommand:
    """Tests for cache command."""

    def test_cache_shows_cache_dir(self, capsys: object) -> object:
        """Test test cache shows cache dir."""
        main(["cache"])
        captured = capsys.readouterr()
        assert "Cache directory:" in captured.out

    def test_cache_shows_fetchcontent(self, capsys: object) -> object:
        """Test test cache shows fetchcontent."""
        main(["cache"])
        captured = capsys.readouterr()
        assert "FetchContent" in captured.out
        assert "clap" in captured.out
        assert "vst3" in captured.out

    def test_cache_shows_rack_sdk(self, capsys: object) -> object:
        """Test test cache shows rack sdk."""
        main(["cache"])
        captured = capsys.readouterr()
        assert "Rack SDK" in captured.out

    def test_cache_shows_libdaisy(self, capsys: object) -> object:
        """Test test cache shows libdaisy."""
        main(["cache"])
        captured = capsys.readouterr()
        assert "libDaisy" in captured.out


class TestNameInference:
    """Tests for name inference when -n is not provided."""

    def test_infers_name_from_export_dir(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test infers name from export dir."""
        output_dir = tmp_path / "output"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-o",
                str(output_dir),
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would create project" in captured.out

    def test_infers_name_from_graph_file(
        self, tmp_path: Path, capsys: object
    ) -> object:
        """Test test infers name from graph file."""
        graph_file = tmp_path / "lowpass.gdsp"
        graph_file.write_text(
            """
            graph lowpass {
                in input
                out output = filt
                param freq 20..20000 = 1000
                filt = onepole(input, freq / 44100)
            }
            """
        )
        result = main(
            [
                str(graph_file),
                "-p",
                "chuck",
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "lowpass" in captured.out

    def test_explicit_name_overrides_inference(
        self, gigaverb_export: Path, tmp_path: Path
    ) -> object:
        """Test test explicit name overrides inference."""
        output_dir = tmp_path / "output"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "myverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0


class TestOutputDirInference:
    """Tests for default output directory {name}_{platform}."""

    def test_output_dir_includes_platform(
        self, gigaverb_export: Path, capsys: object
    ) -> object:
        """Test test output dir includes platform."""
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "chuck",
                "-n",
                "myverb",
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "myverb_chuck" in captured.out

    def test_graph_output_dir_includes_platform(
        self, tmp_path: Path, capsys: object
    ) -> object:
        """Test test graph output dir includes platform."""
        graph_file = tmp_path / "foo.gdsp"
        graph_file.write_text(
            """
            graph foo {
                in input
                out output = scaled
                param gain 0..2 = 1
                scaled = input * gain
            }
            """
        )
        result = main(
            [
                str(graph_file),
                "-p",
                "au",
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "foo_au" in captured.out


class TestNoBuildFlag:
    """Tests for --no-build flag (reversed polarity from old --build)."""

    def test_dry_run_shows_build_intent(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test dry run shows build intent."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would build after creating" in captured.out
        assert not output_dir.exists()

    def test_no_build_dry_run_omits_build_intent(
        self, gigaverb_export: Path, tmp_path: Path, capsys: object
    ) -> object:
        """Test test no build dry run omits build intent."""
        output_dir = tmp_path / "testverb"
        result = main(
            [
                str(gigaverb_export),
                "-p",
                "pd",
                "-n",
                "testverb",
                "-o",
                str(output_dir),
                "--no-build",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would build after creating" not in captured.out

    def test_graph_dry_run_shows_build_intent(
        self, tmp_path: Path, capsys: object
    ) -> object:
        """Test test graph dry run shows build intent."""
        graph_file = tmp_path / "gain.gdsp"
        graph_file.write_text(
            """
            graph gain {
                in input
                out output = scaled
                param vol 0..2 = 1
                scaled = input * vol
            }
            """
        )
        result = main(
            [
                str(graph_file),
                "-p",
                "chuck",
                "--dry-run",
            ]
        )
        assert result == NUM_0
        captured = capsys.readouterr()
        assert "Would build after creating" in captured.out
