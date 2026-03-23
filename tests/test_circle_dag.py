"""Tests for Circle DAG project generation."""

import json
from pathlib import Path

from gen_dsp.core.graph import (
    ChainNodeConfig,
    Connection,
    EdgeBuffer,
    GraphConfig,
    ResolvedChainNode,
    allocate_edge_buffers,
    parse_graph,
    resolve_dag,
    validate_dag,
    validate_linear_chain,
)
from gen_dsp.core.manifest import Manifest, ParamInfo
from gen_dsp.core.parser import ExportInfo
from gen_dsp.core.project import ProjectConfig
from gen_dsp.platforms.circle import (
    CirclePlatform,
    _build_dag_buffer_decls,
    _build_dag_buffer_init,
    _build_dag_create,
    _build_dag_destroy,
    _build_dag_includes,
    _build_dag_io_defines,
    _build_dag_midi_dispatch,
    _build_dag_mixer_gain_decls,
    _build_dag_per_node_flags,
    _build_dag_perform,
    _build_dag_set_param,
)

NUM_0 = 0


def _make_resolved_node(
    node_id: str,
    export_name: str,
    index: int,
    **kwargs: object,
) -> ResolvedChainNode:
    """Build a resolved DAG node without reading export files."""
    num_inputs = int(kwargs.get("num_inputs", 2))
    num_outputs = int(kwargs.get("num_outputs", 2))
    num_params = int(kwargs.get("num_params", 0))
    params = kwargs.get("params")
    midi_channel = kwargs.get("midi_channel")
    cc_map = kwargs.get("cc_map")
    node_type = str(kwargs.get("node_type", "gen"))
    mixer_inputs = int(kwargs.get("mixer_inputs", 0))
    config = ChainNodeConfig(
        id=node_id,
        export=export_name if node_type == "gen" else None,
        node_type=node_type,
        mixer_inputs=mixer_inputs,
        midi_channel=
            index + 1 if midi_channel is None else int(midi_channel),
        cc_map=cc_map or {},
    )
    export_info = (
        ExportInfo(
            name=f"gen_{export_name}",
            path=Path(f"/fake/{export_name}"),
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_params=num_params,
        )
        if node_type == "gen"
        else None
    )
    manifest = Manifest(
        gen_name=f"gen_{export_name}" if node_type == "gen" else f"mixer_{node_id}",
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        params=params or [],
    )
    return ResolvedChainNode(
        config=config,
        index=index,
        export_info=export_info,
        manifest=manifest,
    )


def _diamond_nodes() -> list[ResolvedChainNode]:
    """Build a simple diamond graph: reverb + delay -> mix."""
    return [
        _make_resolved_node(
            "reverb",
            "gigaverb",
            0,
            num_params=5,
            params=[
                ParamInfo(
                    index=i,
                    name=f"param{i}",
                    has_minmax=True,
                    min=0.0,
                    max=1.0,
                    default=0.5,
                )
                for i in range(5)
            ],
        ),
        _make_resolved_node(
            "delay",
            "spectraldelayfb",
            1,
            num_inputs=3,
            num_params=3,
            params=[
                ParamInfo(
                    index=i,
                    name=f"param{i}",
                    has_minmax=True,
                    min=0.0,
                    max=1.0,
                    default=0.5,
                )
                for i in range(3)
            ],
        ),
        _make_resolved_node(
            "mix",
            "mix",
            2,
            num_params=2,
            params=[
                ParamInfo(
                    index=0,
                    name="gain_0",
                    has_minmax=True,
                    min=0.0,
                    max=2.0,
                    default=1.0,
                ),
                ParamInfo(
                    index=1,
                    name="gain_1",
                    has_minmax=True,
                    min=0.0,
                    max=2.0,
                    default=1.0,
                ),
            ],
            node_type="mixer",
            mixer_inputs=2,
        ),
    ]


def _diamond_graph() -> GraphConfig:
    """Build a graph matching the diamond node layout."""
    return GraphConfig(
        nodes={
            "reverb": ChainNodeConfig(id="reverb", export="gigaverb", midi_channel=1),
            "delay": ChainNodeConfig(
                id="delay",
                export="spectraldelayfb",
                midi_channel=2,
            ),
            "mix": ChainNodeConfig(
                id="mix",
                node_type="mixer",
                mixer_inputs=2,
                midi_channel=3,
            ),
        },
        connections=[
            Connection("audio_in", "reverb"),
            Connection("audio_in", "delay"),
            Connection("reverb", "mix", dst_input_index=0),
            Connection("delay", "mix", dst_input_index=1),
            Connection("mix", "audio_out"),
        ],
    )


def _diamond_edge_buffers() -> tuple[list[EdgeBuffer], int]:
    """Build edge buffers for the diamond DAG."""
    edges = [
        EdgeBuffer(-1, "audio_in", "reverb", None, 2),
        EdgeBuffer(-1, "audio_in", "delay", None, 2),
        EdgeBuffer(0, "reverb", "mix", 0, 2),
        EdgeBuffer(1, "delay", "mix", 1, 2),
        EdgeBuffer(2, "mix", "audio_out", None, 2),
    ]
    return edges, 3


def test_dag_helper_functions() -> None:
    """The DAG helper code emits expected fragments."""
    nodes = _diamond_nodes()
    edges, _ = _diamond_edge_buffers()
    graph = _diamond_graph()

    assert "DAG_NUM_BUFFERS    3" in _build_dag_buffer_decls(3, 2)
    assert "m_pDagBuf_0[ch] = m_DagBufStorage_0[ch]" in _build_dag_buffer_init(2)
    assert '#include "_ext_circle_0.h"' in _build_dag_includes(nodes)
    assert "m_mix_gain_0" in _build_dag_mixer_gain_decls(nodes)
    assert "reverb_circle::wrapper_create" in _build_dag_create(nodes)
    assert "mix_circle" not in _build_dag_destroy(nodes)
    assert "reverb_circle::wrapper_set_param" in _build_dag_set_param(nodes)
    assert "m_pDagBuf_0[ch][s]" in _build_dag_perform(
        nodes,
        edges,
        graph,
        max_channels=3,
    )
    assert "channel == 3" in _build_dag_midi_dispatch(nodes, graph)
    assert "_ext_circle_0.o: CPPFLAGS +=" in _build_dag_per_node_flags(nodes)
    assert "NODE_2" not in _build_dag_io_defines(nodes)


def test_dag_project_generation_smoke(
    gigaverb_export: Path,
    spectraldelayfb_export: Path,
    tmp_path: Path,
) -> None:
    """A DAG project can be generated from real exports."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    graph = parse_graph(graph_path)
    dag_nodes = resolve_dag(
        graph,
        {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        },
        "0.8.0",
    )

    resolved_map = {node.config.id: node for node in dag_nodes}
    topo_order = [node.config.id for node in dag_nodes]
    edge_buffers, num_buffers = allocate_edge_buffers(
        graph,
        resolved_map,
        topo_order,
    )

    output_dir = tmp_path / "dag_project"
    output_dir.mkdir()

    CirclePlatform().generate_dag_project(
        dag_nodes,
        graph,
        edge_buffers,
        num_buffers,
        output_dir,
        "mydag",
        ProjectConfig(name="mydag", platform="circle"),
    )

    assert (output_dir / "gen_ext_circle.cpp").is_file()
    assert (output_dir / "Makefile").is_file()
    assert (output_dir / "gen_buffer.h").is_file()


def test_linear_and_dag_validation(tmp_path: Path) -> None:
    """Linear graphs stay linear and diamond graphs validate as DAGs."""
    linear_graph_path = tmp_path / "linear.json"
    linear_graph_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    linear_graph = parse_graph(linear_graph_path)
    assert validate_linear_chain(linear_graph) == []

    dag_graph_path = tmp_path / "dag.json"
    dag_graph_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    dag_graph = parse_graph(dag_graph_path)
    assert validate_dag(dag_graph) == []
