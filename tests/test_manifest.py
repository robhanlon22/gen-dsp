"""Tests for the Manifest IR and parameter parsing."""

import json
from pathlib import Path

from gen_dsp.core.manifest import (
    Manifest,
    ParamInfo,
    manifest_from_export_info,
    parse_params_from_export,
)
from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.project import ProjectConfig, ProjectGenerator

NUM_0 = 0
NUM_0_1 = 0.1
NUM_0_25 = 0.25
NUM_0_5 = 0.5
NUM_0_7 = 0.7
NUM_1 = 1
NUM_2 = 2
NUM_23_0 = 23.0
NUM_3 = 3
NUM_7 = 7
NUM_75_0 = 75.0
NUM_8 = 8


class TestParamInfoSerialization:
    """Test ParamInfo to_dict / from_dict round-trips."""

    def test_round_trip(self) -> object:
        """Test test round trip."""
        p = ParamInfo(
            index=3,
            name="bandwidth",
            has_minmax=True,
            min=0.0,
            max=1.0,
            default=0.0,
        )
        d = p.to_dict()
        p2 = ParamInfo.from_dict(d)
        assert p2.index == p.index
        assert p2.name == p.name
        assert p2.has_minmax == p.has_minmax
        assert p2.min == p.min
        assert p2.max == p.max
        assert p2.default == p.default

    def test_dict_keys(self) -> object:
        """Test test dict keys."""
        p = ParamInfo(
            index=0, name="x", has_minmax=False, min=0.0, max=1.0, default=0.0
        )
        d = p.to_dict()
        assert set(d.keys()) == {"index", "name", "has_minmax", "min", "max", "default"}


class TestManifestSerialization:
    """Test Manifest to_dict / from_dict / to_json / from_json round-trips."""

    def _make_manifest(self) -> Manifest:
        return Manifest(
            gen_name="gen_exported",
            num_inputs=2,
            num_outputs=2,
            params=[
                ParamInfo(
                    index=0,
                    name="bandwidth",
                    has_minmax=True,
                    min=0.0,
                    max=1.0,
                    default=0.0,
                ),
                ParamInfo(
                    index=1,
                    name="revtime",
                    has_minmax=True,
                    min=0.1,
                    max=1.0,
                    default=0.1,
                ),
            ],
            buffers=["sample"],
            source="gen~",
            version="0.8.0",
        )

    def test_num_params_property(self) -> object:
        """Test test num params property."""
        m = self._make_manifest()
        assert m.num_params == NUM_2

    def test_num_params_empty(self) -> object:
        """Test test num params empty."""
        m = Manifest(gen_name="x", num_inputs=0, num_outputs=1)
        assert m.num_params == NUM_0

    def test_dict_round_trip(self) -> object:
        """Test test dict round trip."""
        m = self._make_manifest()
        d = m.to_dict()
        m2 = Manifest.from_dict(d)
        assert m2.gen_name == m.gen_name
        assert m2.num_inputs == m.num_inputs
        assert m2.num_outputs == m.num_outputs
        assert m2.num_params == m.num_params
        assert len(m2.params) == len(m.params)
        assert m2.buffers == m.buffers
        assert m2.source == m.source
        assert m2.version == m.version

    def test_json_round_trip(self) -> object:
        """Test test json round trip."""
        m = self._make_manifest()
        j = m.to_json()
        m2 = Manifest.from_json(j)
        assert m2.gen_name == m.gen_name
        assert m2.num_params == m.num_params
        assert m2.params[NUM_1].name == "revtime"
        assert m2.params[NUM_1].min == NUM_0_1

    def test_json_is_valid(self) -> object:
        """Test test json is valid."""
        m = self._make_manifest()
        j = m.to_json()
        parsed = json.loads(j)
        assert parsed["gen_name"] == "gen_exported"
        assert len(parsed["params"]) == NUM_2
        # num_params should NOT be in the JSON (derived property)
        assert "num_params" not in parsed

    def test_from_dict_defaults(self) -> object:
        """Test test from dict defaults."""
        d = {"gen_name": "test", "num_inputs": 1, "num_outputs": 1}
        m = Manifest.from_dict(d)
        assert m.params == []
        assert m.buffers == []
        assert m.source == "gen~"
        assert m.version == "0.8.0"


class TestParamParsing:
    """
    Test parameter metadata extraction from gen~ exports.

    Migrated from TestScParamParsing and TestLv2ParamParsing.
    """

    def test_parse_gigaverb_params(self, gigaverb_export: Path) -> object:
        """Test test parse gigaverb params."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)

        assert len(params) == NUM_8
        # Check first param
        assert params[NUM_0].index == NUM_0
        assert params[NUM_0].name == "bandwidth"
        # Check last param
        assert params[NUM_7].index == NUM_7
        assert params[NUM_7].name == "tail"
        # All gigaverb params have hasminmax=true
        for p in params:
            assert p.has_minmax is True
            assert p.max >= p.min
        # Spot-check: revtime has min=0.1, others have min=0
        assert params[NUM_0].min == NUM_0  # bandwidth
        assert params[NUM_0].max == NUM_1
        revtime = next(p for p in params if p.name == "revtime")
        assert revtime.min == NUM_0_1
        assert revtime.max == NUM_1

    def test_parse_spectraldelayfb_params(self, spectraldelayfb_export: Path) -> object:
        """Test test parse spectraldelayfb params."""
        parser = GenExportParser(spectraldelayfb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)

        assert len(params) == NUM_0

    def test_parse_params_sorted_by_index(self, gigaverb_export: Path) -> object:
        """Test test parse params sorted by index."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)

        indices = [p.index for p in params]
        assert indices == sorted(indices)

    def test_param_names_are_valid_identifiers(self, gigaverb_export: Path) -> object:
        """Test test param names are valid identifiers."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)

        for p in params:
            assert p.name.isidentifier(), f"{p.name} is not a valid identifier"

    def test_defaults_clamped_to_range(self, gigaverb_export: Path) -> object:
        """Test test defaults clamped to range."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)

        for p in params:
            assert p.min <= p.default <= p.max, (
                f"{p.name}: default {p.default} outside [{p.min}, {p.max}]"
            )

    def test_defaults_from_gen_export(self, gigaverb_export: Path) -> object:
        """Test test defaults from gen export."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        params = parse_params_from_export(export_info)
        by_name = {p.name: p for p in params}

        # Known gigaverb initial values (from reset() in gen_exported.cpp)
        # Values that exceed range are clamped: revtime init=11 -> max=1,
        # roomsize init=75 -> max=300, spread init=23 -> max=100
        assert by_name["bandwidth"].default == NUM_0_5
        assert by_name["damping"].default == NUM_0_7
        assert by_name["dry"].default == NUM_1
        assert by_name["early"].default == NUM_0_25
        assert by_name["revtime"].default == NUM_1  # init=11, clamped to max=1
        assert by_name["roomsize"].default == NUM_75_0  # init=75, within [0.1, 300]
        assert by_name["spread"].default == NUM_23_0  # init=23, within [0, 100]
        assert by_name["tail"].default == NUM_0_25


class TestManifestFromExportInfo:
    """Test manifest_from_export_info() integration."""

    def test_gigaverb_manifest(self, gigaverb_export: Path) -> object:
        """Test test gigaverb manifest."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        manifest = manifest_from_export_info(export_info, [], "0.8.0")

        assert manifest.gen_name == "gen_exported"
        assert manifest.num_inputs == NUM_2
        assert manifest.num_outputs == NUM_2
        assert manifest.num_params == NUM_8
        assert len(manifest.params) == NUM_8
        assert manifest.buffers == []
        assert manifest.source == "gen~"
        assert manifest.version == "0.8.0"

    def test_rampleplayer_manifest_with_buffers(
        self, rampleplayer_export: Path
    ) -> object:
        """Test test rampleplayer manifest with buffers."""
        parser = GenExportParser(rampleplayer_export)
        export_info = parser.parse()

        manifest = manifest_from_export_info(export_info, ["sample"], "0.8.0")

        assert manifest.num_inputs == NUM_1
        assert manifest.num_outputs == NUM_2
        assert manifest.buffers == ["sample"]

    def test_spectraldelayfb_manifest(self, spectraldelayfb_export: Path) -> object:
        """Test spectraldelayfb manifest."""
        parser = GenExportParser(spectraldelayfb_export)
        export_info = parser.parse()

        manifest = manifest_from_export_info(export_info, [], "0.8.0")

        assert manifest.num_inputs == NUM_3
        assert manifest.num_outputs == NUM_2
        assert manifest.num_params == NUM_0
        assert manifest.params == []

    def test_manifest_json_round_trip(self, gigaverb_export: Path) -> object:
        """Test manifest json round trip."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        m1 = manifest_from_export_info(export_info, [], "0.8.0")
        j = m1.to_json()
        m2 = Manifest.from_json(j)

        assert m2.gen_name == m1.gen_name
        assert m2.num_params == m1.num_params
        assert m2.params[NUM_0].name == m1.params[NUM_0].name


class TestManifestJsonEmission:
    """Test that manifest.json is emitted during project generation."""

    def test_manifest_json_emitted(
        self, gigaverb_export: Path, tmp_project: Path
    ) -> object:
        """Verify manifest.json appears in generated project."""
        parser = GenExportParser(gigaverb_export)
        export_info = parser.parse()

        config = ProjectConfig(name="testverb", platform="clap")
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        manifest_path = project_dir / "manifest.json"
        assert manifest_path.is_file()

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["gen_name"] == "gen_exported"
        assert data["num_inputs"] == NUM_2
        assert data["num_outputs"] == NUM_2
        assert len(data["params"]) == NUM_8
        assert data["buffers"] == []
        assert data["source"] == "gen~"

    def test_manifest_json_with_buffers(
        self, rampleplayer_export: Path, tmp_project: Path
    ) -> object:
        """Verify manifest.json includes buffers when present."""
        parser = GenExportParser(rampleplayer_export)
        export_info = parser.parse()

        config = ProjectConfig(name="testsampler", platform="clap", buffers=["sample"])
        generator = ProjectGenerator(export_info, config)
        project_dir = generator.generate(tmp_project)

        data = json.loads((project_dir / "manifest.json").read_text(encoding="utf-8"))
        assert data["buffers"] == ["sample"]
