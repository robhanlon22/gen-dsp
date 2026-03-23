"""Tests for the graph data model (multi-plugin chain configurations)."""

import json
from pathlib import Path

import pytest

from gen_dsp.core.graph import (
    ChainNodeConfig,
    Connection,
    GraphConfig,
    ResolvedChainNode,
    allocate_edge_buffers,
    extract_chain_order,
    parse_graph,
    resolve_chain,
    resolve_dag,
    topological_sort,
    validate_dag,
    validate_linear_chain,
)
from gen_dsp.core.manifest import Manifest
from gen_dsp.core.parser import ExportInfo
from gen_dsp.errors import ValidationError

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_21 = 21
NUM_22 = 22
NUM_3 = 3
NUM_5 = 5
NUM_NEG_1 = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_graph(tmp_path: Path, data: dict) -> Path:
    """Write a graph JSON file and return its path."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(data), encoding="utf-8")
    return graph_path


def _simple_graph_data() -> dict:
    """Return a minimal valid linear chain graph."""
    return {
        "nodes": {
            "reverb": {"export": "gigaverb"},
            "delay": {"export": "spectraldelayfb"},
        },
        "connections": [
            ["audio_in", "reverb"],
            ["reverb", "delay"],
            ["delay", "audio_out"],
        ],
    }


def _single_node_graph_data() -> dict:
    """Return a single-node chain graph."""
    return {
        "nodes": {
            "reverb": {"export": "gigaverb"},
        },
        "connections": [
            ["audio_in", "reverb"],
            ["reverb", "audio_out"],
        ],
    }

    # ---------------------------------------------------------------------------
    # TestGraphParsing
    # ---------------------------------------------------------------------------


class TestGraphParsing:
    """Test parse_graph() JSON loading and validation."""

    def test_parse_minimal_graph(self, tmp_path: object) -> object:
        """Test test parse minimal graph."""
        path = _write_graph(tmp_path, _simple_graph_data())
        graph = parse_graph(path)

        assert len(graph.nodes) == NUM_2
        assert "reverb" in graph.nodes
        assert "delay" in graph.nodes
        assert graph.nodes["reverb"].export == "gigaverb"
        assert graph.nodes["delay"].export == "spectraldelayfb"
        assert len(graph.connections) == NUM_3

    def test_parse_graph_with_midi_channel(self, tmp_path: object) -> object:
        """Test test parse graph with midi channel."""
        data = _simple_graph_data()
        data["nodes"]["reverb"]["midi_channel"] = 3
        data["nodes"]["delay"]["midi_channel"] = 5

        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)

        assert graph.nodes["reverb"].midi_channel == NUM_3
        assert graph.nodes["delay"].midi_channel == NUM_5

    def test_parse_graph_with_cc_map(self, tmp_path: object) -> object:
        """Test test parse graph with cc map."""
        data = _simple_graph_data()
        data["nodes"]["reverb"]["cc"] = {"21": "revtime", "22": "damping"}

        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)

        assert graph.nodes["reverb"].cc_map == {NUM_21: "revtime", NUM_22: "damping"}

    def test_parse_graph_missing_file(self, tmp_path: object) -> object:
        """Test test parse graph missing file."""
        with pytest.raises(ValidationError, match="not found"):
            parse_graph(tmp_path / "nonexistent.json")

    def test_parse_graph_invalid_json(self, tmp_path: object) -> object:
        """Test test parse graph invalid json."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValidationError, match="Invalid JSON"):
            parse_graph(bad_path)

    def test_parse_graph_not_object(self, tmp_path: object) -> object:
        """Test test parse graph not object."""
        path = _write_graph(tmp_path, [1, 2, 3])
        with pytest.raises(ValidationError, match="must be an object"):
            parse_graph(path)

    def test_parse_graph_missing_nodes(self, tmp_path: object) -> object:
        """Test test parse graph missing nodes."""
        path = _write_graph(tmp_path, {"connections": []})
        with pytest.raises(ValidationError, match="'nodes'"):
            parse_graph(path)

    def test_parse_graph_missing_connections(self, tmp_path: object) -> object:
        """Test test parse graph missing connections."""
        path = _write_graph(tmp_path, {"nodes": {}})
        with pytest.raises(ValidationError, match="'connections'"):
            parse_graph(path)

    def test_parse_graph_node_missing_export(self, tmp_path: object) -> object:
        """Test test parse graph node missing export."""
        data = {
            "nodes": {"bad_node": {"midi_channel": 1}},
            "connections": [["audio_in", "bad_node"], ["bad_node", "audio_out"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="'export'"):
            parse_graph(path)

    def test_parse_graph_invalid_cc_key(self, tmp_path: object) -> object:
        """Test test parse graph invalid cc key."""
        data = _simple_graph_data()
        data["nodes"]["reverb"]["cc"] = {"abc": "revtime"}
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="CC key"):
            parse_graph(path)

    def test_parse_graph_invalid_midi_channel_type(self, tmp_path: object) -> object:
        """Test test parse graph invalid midi channel type."""
        data = _simple_graph_data()
        data["nodes"]["reverb"]["midi_channel"] = "three"
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="midi_channel"):
            parse_graph(path)

    def test_parse_graph_bad_connection_format(self, tmp_path: object) -> object:
        """Test test parse graph bad connection format."""
        data = {
            "nodes": {"reverb": {"export": "gigaverb"}},
            "connections": [["audio_in", "reverb", "extra"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="\\[from, to\\] pair"):
            parse_graph(path)

    def test_parse_graph_default_midi_channel_is_none(self, tmp_path: object) -> object:
        """Test test parse graph default midi channel is none."""
        path = _write_graph(tmp_path, _simple_graph_data())
        graph = parse_graph(path)

        assert graph.nodes["reverb"].midi_channel is None
        assert graph.nodes["delay"].midi_channel is None

    # ---------------------------------------------------------------------------
    # TestChainValidation
    # ---------------------------------------------------------------------------


class TestChainValidation:
    """Test validate_linear_chain() error detection."""

    def test_valid_chain_no_errors(self) -> object:
        """Test test valid chain no errors."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "audio_out"),
            ],
        )
        assert validate_linear_chain(graph) == []

    def test_missing_audio_in(self) -> object:
        """Test test missing audio in."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[Connection("a", "audio_out")],
        )
        errors = validate_linear_chain(graph)
        assert any("audio_in" in e for e in errors)

    def test_missing_audio_out(self) -> object:
        """Test test missing audio out."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[Connection("audio_in", "a")],
        )
        errors = validate_linear_chain(graph)
        assert any("audio_out" in e for e in errors)

    def test_fan_out_detected(self) -> object:
        """Test test fan out detected."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("Fan-out" in e for e in errors)

    def test_fan_in_detected(self) -> object:
        """Test test fan in detected."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("audio_in", "b"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("Fan-in" in e or "Fan-out" in e for e in errors)

    def test_unknown_node_reference(self) -> object:
        """Test test unknown node reference."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "unknown_node"),
                Connection("unknown_node", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("unknown_node" in e for e in errors)

    def test_unconnected_node(self) -> object:
        """Test test unconnected node."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "orphan": ChainNodeConfig(id="orphan", export="ex_orphan"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("orphan" in e for e in errors)

    def test_invalid_midi_channel_too_high(self) -> object:
        """Test test invalid midi channel too high."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a", midi_channel=17),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("midi_channel" in e for e in errors)

    def test_invalid_midi_channel_zero(self) -> object:
        """Test test invalid midi channel zero."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a", midi_channel=0),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("midi_channel" in e for e in errors)

    def test_invalid_cc_number(self) -> object:
        """Test test invalid cc number."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a", cc_map={128: "param"}),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("CC number" in e for e in errors)

    def test_reserved_name_as_node_id(self) -> object:
        """Test test reserved name as node id."""
        graph = GraphConfig(
            nodes={
                "audio_in": ChainNodeConfig(id="audio_in", export="ex_a"),
            },
            connections=[
                Connection("audio_in", "audio_out"),
            ],
        )
        errors = validate_linear_chain(graph)
        assert any("reserved" in e for e in errors)

    # ---------------------------------------------------------------------------
    # TestChainOrdering
    # ---------------------------------------------------------------------------


class TestChainOrdering:
    """Test extract_chain_order() path walking."""

    def test_two_node_chain(self) -> object:
        """Test test two node chain."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="gigaverb"),
                "delay": ChainNodeConfig(id="delay", export="spectraldelayfb"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "delay"),
                Connection("delay", "audio_out"),
            ],
        )
        order = extract_chain_order(graph)
        assert order == ["reverb", "delay"]

    def test_single_node_chain(self) -> object:
        """Test test single node chain."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="gigaverb"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "audio_out"),
            ],
        )
        order = extract_chain_order(graph)
        assert order == ["reverb"]

    def test_three_node_chain(self) -> object:
        """Test test three node chain."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
                "c": ChainNodeConfig(id="c", export="ex_c"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "c"),
                Connection("c", "audio_out"),
            ],
        )
        order = extract_chain_order(graph)
        assert order == ["a", "b", "c"]

    def test_broken_chain_raises(self) -> object:
        """Test test broken chain raises."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
            ],
        )
        with pytest.raises(ValidationError, match="audio_out"):
            extract_chain_order(graph)

    def test_no_audio_in_raises(self) -> object:
        """Test test no audio in raises."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[Connection("a", "audio_out")],
        )
        with pytest.raises(ValidationError, match="audio_in"):
            extract_chain_order(graph)

    def test_cycle_detected(self) -> object:
        """Test test cycle detected."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "a"),  # cycle
            ],
        )
        # validate_linear_chain would catch this as fan-out/fan-in,
        # but extract_chain_order should also detect cycles
        with pytest.raises(ValidationError):
            extract_chain_order(graph)

    # ---------------------------------------------------------------------------
    # TestChainResolution
    # ---------------------------------------------------------------------------


class TestChainResolution:
    """Test resolve_chain() with real fixture exports."""

    def test_resolve_single_node(self, gigaverb_export: object) -> object:
        """Test test resolve single node."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="gigaverb"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "audio_out"),
            ],
        )
        export_dirs = {"gigaverb": gigaverb_export}
        chain = resolve_chain(graph, export_dirs, "0.8.0")

        assert len(chain) == NUM_1
        assert chain[NUM_0].config.id == "reverb"
        assert chain[NUM_0].index == NUM_0
        assert chain[NUM_0].manifest.num_inputs == NUM_2
        assert chain[NUM_0].manifest.num_outputs == NUM_2
        assert chain[NUM_0].config.midi_channel == NUM_1  # auto-assigned

    def test_resolve_two_node_chain(
        self, gigaverb_export: object, spectraldelayfb_export: object
    ) -> object:
        """Test test resolve two node chain."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="gigaverb"),
                "delay": ChainNodeConfig(id="delay", export="spectraldelayfb"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "delay"),
                Connection("delay", "audio_out"),
            ],
        )
        export_dirs = {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        }
        chain = resolve_chain(graph, export_dirs, "0.8.0")

        assert len(chain) == NUM_2
        assert chain[NUM_0].config.id == "reverb"
        assert chain[NUM_0].index == NUM_0
        assert chain[NUM_0].config.midi_channel == NUM_1
        assert chain[NUM_1].config.id == "delay"
        assert chain[NUM_1].index == NUM_1
        assert chain[NUM_1].config.midi_channel == NUM_2

    def test_resolve_preserves_explicit_midi_channel(
        self, gigaverb_export: object
    ) -> object:
        """Test test resolve preserves explicit midi channel."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(
                    id="reverb", export="gigaverb", midi_channel=5
                ),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "audio_out"),
            ],
        )
        export_dirs = {"gigaverb": gigaverb_export}
        chain = resolve_chain(graph, export_dirs, "0.8.0")

        assert chain[NUM_0].config.midi_channel == NUM_5

    def test_resolve_missing_export(self, gigaverb_export: object) -> object:
        """Test test resolve missing export."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="nonexistent"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "audio_out"),
            ],
        )
        export_dirs = {"gigaverb": gigaverb_export}
        with pytest.raises(ValidationError, match="nonexistent"):
            resolve_chain(graph, export_dirs, "0.8.0")

    def test_resolve_bad_export_path(self, tmp_path: object) -> object:
        """Test test resolve bad export path."""
        graph = GraphConfig(
            nodes={
                "bad": ChainNodeConfig(id="bad", export="empty"),
            },
            connections=[
                Connection("audio_in", "bad"),
                Connection("bad", "audio_out"),
            ],
        )
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        export_dirs = {"empty": empty_dir}
        with pytest.raises(ValidationError, match="failed to parse"):
            resolve_chain(graph, export_dirs, "0.8.0")


# ---------------------------------------------------------------------------
# Phase 2: DAG tests
# ---------------------------------------------------------------------------


def _diamond_dag_data() -> dict:
    """Return a diamond DAG: audio_in -> reverb + delay -> mix -> audio_out."""
    return {
        "nodes": {
            "reverb": {"export": "gigaverb"},
            "delay": {"export": "spectraldelayfb"},
            "mix": {"type": "mixer", "inputs": 2},
        },
        "connections": [
            ["audio_in", "reverb"],
            ["audio_in", "delay"],
            ["reverb", "mix:0"],
            ["delay", "mix:1"],
            ["mix", "audio_out"],
        ],
    }


def _diamond_graph() -> GraphConfig:
    """Construct a diamond DAG GraphConfig directly."""
    return GraphConfig(
        nodes={
            "reverb": ChainNodeConfig(id="reverb", export="gigaverb", node_type="gen"),
            "delay": ChainNodeConfig(
                id="delay", export="spectraldelayfb", node_type="gen"
            ),
            "mix": ChainNodeConfig(id="mix", node_type="mixer", mixer_inputs=2),
        },
        connections=[
            Connection("audio_in", "reverb"),
            Connection("audio_in", "delay"),
            Connection("reverb", "mix", dst_input_index=0),
            Connection("delay", "mix", dst_input_index=1),
            Connection("mix", "audio_out"),
        ],
    )


class TestConnectionParsing:
    """Test Connection dataclass and :index parsing."""

    def test_parse_connection_without_index(self, tmp_path: object) -> object:
        """Test test parse connection without index."""
        data = _simple_graph_data()
        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)
        for c in graph.connections:
            assert c.dst_input_index is None

    def test_parse_connection_with_index(self, tmp_path: object) -> object:
        """Test test parse connection with index."""
        data = _diamond_dag_data()
        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)
        indexed = [c for c in graph.connections if c.dst_input_index is not None]
        assert len(indexed) == NUM_2
        indices = sorted(c.dst_input_index for c in indexed)
        assert indices == [NUM_0, NUM_1]

    def test_parse_connection_invalid_index(self, tmp_path: object) -> object:
        """Test test parse connection invalid index."""
        data = {
            "nodes": {"a": {"export": "gigaverb"}},
            "connections": [["audio_in", "a:abc"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="invalid input index"):
            parse_graph(path)

    def test_connection_backward_compat(self, tmp_path: object) -> object:
        """Test test connection backward compat."""
        data = _simple_graph_data()
        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)
        assert len(graph.connections) == NUM_3
        assert graph.connections[NUM_0].src_node == "audio_in"
        assert graph.connections[NUM_0].dst_node == "reverb"


class TestMixerNodeParsing:
    """Test mixer node parsing and validation."""

    def test_parse_mixer_node(self, tmp_path: object) -> object:
        """Test test parse mixer node."""
        data = _diamond_dag_data()
        path = _write_graph(tmp_path, data)
        graph = parse_graph(path)
        mix = graph.nodes["mix"]
        assert mix.node_type == "mixer"
        assert mix.mixer_inputs == NUM_2
        assert mix.export is None

    def test_mixer_node_missing_inputs(self, tmp_path: object) -> object:
        """Test test mixer node missing inputs."""
        data = {
            "nodes": {"mix": {"type": "mixer"}},
            "connections": [["audio_in", "mix"], ["mix", "audio_out"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="inputs"):
            parse_graph(path)

    def test_mixer_node_invalid_inputs(self, tmp_path: object) -> object:
        """Test test mixer node invalid inputs."""
        data = {
            "nodes": {"mix": {"type": "mixer", "inputs": 0}},
            "connections": [["audio_in", "mix"], ["mix", "audio_out"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="positive integer"):
            parse_graph(path)

    def test_unknown_node_type(self, tmp_path: object) -> object:
        """Test test unknown node type."""
        data = {
            "nodes": {"a": {"type": "splitter"}},
            "connections": [["audio_in", "a"], ["a", "audio_out"]],
        }
        path = _write_graph(tmp_path, data)
        with pytest.raises(ValidationError, match="unknown node type"):
            parse_graph(path)

    def test_linear_validation_rejects_mixer(self) -> object:
        """Test test linear validation rejects mixer."""
        graph = _diamond_graph()
        errors = validate_linear_chain(graph)
        assert any("mixer" in e for e in errors)


class TestValidateDAG:
    """Test validate_dag() error detection."""

    def test_valid_diamond_dag(self) -> object:
        """Test test valid diamond dag."""
        graph = _diamond_graph()
        assert validate_dag(graph) == []

    def test_dag_missing_audio_in(self) -> object:
        """Test test dag missing audio in."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[Connection("a", "audio_out")],
        )
        errors = validate_dag(graph)
        assert any("audio_in" in e for e in errors)

    def test_dag_missing_audio_out(self) -> object:
        """Test test dag missing audio out."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[Connection("audio_in", "a")],
        )
        errors = validate_dag(graph)
        assert any("audio_out" in e for e in errors)

    def test_dag_cycle_detected(self) -> object:
        """Test test dag cycle detected."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "a"),
                Connection("b", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("Cycle" in e for e in errors)

    def test_dag_disconnected_node(self) -> object:
        """Test test dag disconnected node."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "orphan": ChainNodeConfig(id="orphan", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("orphan" in e for e in errors)

    def test_dag_unreachable_from_audio_in(self) -> object:
        """Test test dag unreachable from audio in."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
                Connection("b", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("not reachable from audio_in" in e for e in errors)

    def test_dag_cannot_reach_audio_out(self) -> object:
        """Test test dag cannot reach audio out."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("audio_in", "b"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("cannot reach audio_out" in e for e in errors)

    def test_dag_mixer_input_count_mismatch(self) -> object:
        """Test test dag mixer input count mismatch."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "mix": ChainNodeConfig(id="mix", node_type="mixer", mixer_inputs=3),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "mix"),
                Connection("mix", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("expects 3 inputs but has 1" in e for e in errors)

    def test_dag_reserved_name(self) -> object:
        """Test test dag reserved name."""
        graph = GraphConfig(
            nodes={
                "audio_in": ChainNodeConfig(id="audio_in", export="ex_a"),
            },
            connections=[Connection("audio_in", "audio_out")],
        )
        errors = validate_dag(graph)
        assert any("reserved" in e for e in errors)

    def test_dag_invalid_midi_channel(self) -> object:
        """Test test dag invalid midi channel."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a", midi_channel=17),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        errors = validate_dag(graph)
        assert any("midi_channel" in e for e in errors)

    def test_valid_linear_dag(self) -> object:
        """Test test valid linear dag."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "audio_out"),
            ],
        )
        assert validate_dag(graph) == []


class TestTopologicalSort:
    """Test topological_sort() ordering."""

    def test_diamond_dag(self) -> object:
        """Test test diamond dag."""
        graph = _diamond_graph()
        order = topological_sort(graph)
        assert len(order) == NUM_3
        assert order.index("mix") > order.index("reverb")
        assert order.index("mix") > order.index("delay")

    def test_fan_out_dag(self) -> object:
        """Test test fan out dag."""
        graph = GraphConfig(
            nodes={
                "src": ChainNodeConfig(id="src", export="ex_a"),
                "a": ChainNodeConfig(id="a", export="ex_b"),
                "b": ChainNodeConfig(id="b", export="ex_c"),
                "mix": ChainNodeConfig(id="mix", node_type="mixer", mixer_inputs=2),
            },
            connections=[
                Connection("audio_in", "src"),
                Connection("src", "a"),
                Connection("src", "b"),
                Connection("a", "mix", dst_input_index=0),
                Connection("b", "mix", dst_input_index=1),
                Connection("mix", "audio_out"),
            ],
        )
        order = topological_sort(graph)
        assert order.index("src") < order.index("a")
        assert order.index("src") < order.index("b")
        assert order.index("a") < order.index("mix")
        assert order.index("b") < order.index("mix")

    def test_single_node(self) -> object:
        """Test test single node."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        assert topological_sort(graph) == ["a"]

    def test_linear_chain(self) -> object:
        """Test test linear chain."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
                "c": ChainNodeConfig(id="c", export="ex_c"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "c"),
                Connection("c", "audio_out"),
            ],
        )
        assert topological_sort(graph) == ["a", "b", "c"]


class TestAllocateEdgeBuffers:
    """Test allocate_edge_buffers() buffer assignment."""

    def _make_resolved_map(
        self, nodes: dict[str, tuple[int, int]]
    ) -> dict[str, ResolvedChainNode]:
        """Create a minimal resolved node map for buffer allocation tests."""
        result = {}
        for i, (nid, (n_in, n_out)) in enumerate(nodes.items()):
            config = ChainNodeConfig(id=nid, export=f"ex_{nid}")
            info = ExportInfo(
                name=f"gen_{nid}",
                path=Path(f"/fake/{nid}"),
                num_inputs=n_in,
                num_outputs=n_out,
            )
            manifest = Manifest(
                gen_name=f"gen_{nid}",
                num_inputs=n_in,
                num_outputs=n_out,
            )
            result[nid] = ResolvedChainNode(
                config=config, index=i, export_info=info, manifest=manifest
            )
        return result

    def test_fan_out_shares_buffer(self) -> object:
        """Fan-out edges from same source share one buffer_id."""
        graph = GraphConfig(
            nodes={
                "src": ChainNodeConfig(id="src", export="ex_a"),
                "a": ChainNodeConfig(id="a", export="ex_b"),
                "b": ChainNodeConfig(id="b", export="ex_c"),
            },
            connections=[
                Connection("audio_in", "src"),
                Connection("src", "a"),
                Connection("src", "b"),
                Connection("a", "audio_out"),
                Connection("b", "audio_out"),
            ],
        )
        resolved = self._make_resolved_map({"src": (2, 2), "a": (2, 2), "b": (2, 2)})
        topo = ["src", "a", "b"]
        edges, _total = allocate_edge_buffers(graph, resolved, topo)

        # Edges from src should share the same buffer_id
        src_edges = [e for e in edges if e.src_node == "src"]
        assert len(src_edges) == NUM_2
        assert src_edges[NUM_0].buffer_id == src_edges[NUM_1].buffer_id

    def test_audio_in_edges_not_allocated(self) -> object:
        """Edges from audio_in have buffer_id = -1."""
        graph = GraphConfig(
            nodes={"a": ChainNodeConfig(id="a", export="ex_a")},
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "audio_out"),
            ],
        )
        resolved = self._make_resolved_map({"a": (2, 2)})
        edges, _total = allocate_edge_buffers(graph, resolved, ["a"])

        audio_in_edges = [e for e in edges if e.src_node == "audio_in"]
        assert len(audio_in_edges) == NUM_1
        assert audio_in_edges[NUM_0].buffer_id == NUM_NEG_1

    def test_buffer_count(self) -> object:
        """Total buffer count matches unique source allocations."""
        graph = GraphConfig(
            nodes={
                "reverb": ChainNodeConfig(id="reverb", export="ex_a"),
                "delay": ChainNodeConfig(id="delay", export="ex_b"),
                "mix": ChainNodeConfig(id="mix", export="ex_c"),
            },
            connections=[
                Connection("audio_in", "reverb"),
                Connection("reverb", "delay"),
                Connection("delay", "mix"),
                Connection("mix", "audio_out"),
            ],
        )
        resolved = self._make_resolved_map(
            {
                "reverb": (2, 2),
                "delay": (3, 2),
                "mix": (2, 2),
            }
        )
        topo = ["reverb", "delay", "mix"]
        _edges, total = allocate_edge_buffers(graph, resolved, topo)

        # Sources needing buffers: reverb, delay, mix = 3 allocated buffers
        assert total == NUM_3

    def test_channel_count_from_source(self) -> object:
        """Edge channel count comes from source node's output count."""
        graph = GraphConfig(
            nodes={
                "a": ChainNodeConfig(id="a", export="ex_a"),
                "b": ChainNodeConfig(id="b", export="ex_b"),
            },
            connections=[
                Connection("audio_in", "a"),
                Connection("a", "b"),
                Connection("b", "audio_out"),
            ],
        )
        resolved = self._make_resolved_map({"a": (2, 3), "b": (3, 2)})
        edges, _ = allocate_edge_buffers(graph, resolved, ["a", "b"])
        a_to_b = [e for e in edges if e.src_node == "a" and e.dst_node == "b"]
        assert len(a_to_b) == NUM_1
        assert a_to_b[NUM_0].num_channels == NUM_3  # a outputs 3 channels


class TestResolveDAG:
    """Test resolve_dag() with real fixtures."""

    def test_resolve_diamond_dag(
        self, gigaverb_export: Path, spectraldelayfb_export: Path
    ) -> object:
        """Test test resolve diamond dag."""
        export_dirs = {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        }
        graph = _diamond_graph()
        resolved = resolve_dag(graph, export_dirs, "0.8.0")

        assert len(resolved) == NUM_3
        node_ids = [n.config.id for n in resolved]
        # mix must come after reverb and delay
        assert node_ids.index("mix") > node_ids.index("reverb")
        assert node_ids.index("mix") > node_ids.index("delay")

        # Mixer node has synthetic manifest
        mix_node = next(n for n in resolved if n.config.id == "mix")
        assert mix_node.export_info is None
        assert mix_node.manifest.num_params == NUM_2
        assert mix_node.manifest.params[NUM_0].name == "gain_0"
        assert mix_node.manifest.params[NUM_1].name == "gain_1"

    def test_resolve_dag_assigns_midi_channels(
        self, gigaverb_export: Path, spectraldelayfb_export: Path
    ) -> object:
        """Test test resolve dag assigns midi channels."""
        export_dirs = {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        }
        graph = _diamond_graph()
        resolved = resolve_dag(graph, export_dirs, "0.8.0")
        channels = [n.config.midi_channel for n in resolved]
        assert channels == [NUM_1, NUM_2, NUM_3]

    def test_resolve_dag_mixer_output_channels(
        self, gigaverb_export: Path, spectraldelayfb_export: Path
    ) -> object:
        """Test test resolve dag mixer output channels."""
        export_dirs = {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        }
        graph = _diamond_graph()
        resolved = resolve_dag(graph, export_dirs, "0.8.0")
        mix_node = next(n for n in resolved if n.config.id == "mix")
        # gigaverb outputs 2, spectraldelayfb outputs 2
        assert mix_node.manifest.num_outputs == NUM_2

    def test_resolve_dag_missing_export(self, gigaverb_export: Path) -> object:
        """Test resolve dag missing export."""
        export_dirs = {"gigaverb": gigaverb_export}
        graph = _diamond_graph()
        with pytest.raises(ValidationError, match="spectraldelayfb"):
            resolve_dag(graph, export_dirs, "0.8.0")
