"""Tests for Circle chain project generation."""

import json
from pathlib import Path

import pytest

from gen_dsp.core.graph import (
    ChainNodeConfig,
    Connection,
    GraphConfig,
    ResolvedChainNode,
    parse_graph,
    resolve_chain,
)
from gen_dsp.core.manifest import Manifest, ParamInfo
from gen_dsp.core.parser import ExportInfo
from gen_dsp.core.project import ProjectConfig
from gen_dsp.errors import ProjectError
from gen_dsp.platforms.circle import (
    CirclePlatform,
    _build_chain_create,
    _build_chain_destroy,
    _build_chain_includes,
    _build_chain_io_defines,
    _build_chain_midi_dispatch,
    _build_chain_per_node_flags,
    _build_chain_perform,
    _build_chain_set_param,
)


def _make_resolved_node(
    node_id: str,
    export_name: str,
    index: int,
    **kwargs: object,
) -> ResolvedChainNode:
    """Build a resolved gen node without reading export files."""
    num_inputs = int(kwargs.get("num_inputs", 2))
    num_outputs = int(kwargs.get("num_outputs", 2))
    num_params = int(kwargs.get("num_params", 0))
    params = kwargs.get("params")
    midi_channel = kwargs.get("midi_channel")
    cc_map = kwargs.get("cc_map")
    config = ChainNodeConfig(
        id=node_id,
        export=export_name,
        midi_channel=
            index + 1 if midi_channel is None else int(midi_channel),
        cc_map=cc_map or {},
    )
    export_info = ExportInfo(
        name=f"gen_{export_name}",
        path=Path(f"/fake/{export_name}"),
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_params=num_params,
    )
    manifest = Manifest(
        gen_name=f"gen_{export_name}",
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


def _two_node_chain() -> list[ResolvedChainNode]:
    """Build a simple two-node chain."""
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
    ]


def _two_node_graph() -> GraphConfig:
    """Build a graph that matches the two-node chain."""
    return GraphConfig(
        nodes={
            "reverb": ChainNodeConfig(id="reverb", export="gigaverb", midi_channel=1),
            "delay": ChainNodeConfig(
                id="delay",
                export="spectraldelayfb",
                midi_channel=2,
            ),
        },
        connections=[
            Connection("audio_in", "reverb"),
            Connection("reverb", "delay"),
            Connection("delay", "audio_out"),
        ],
    )


def test_chain_helper_functions() -> None:
    """The chain code helpers produce expected fragments."""
    chain = _two_node_chain()
    graph = _two_node_graph()

    assert '#include "_ext_circle_0.h"' in _build_chain_includes(chain)
    assert "NODE_0_NUM_INPUTS  2" in _build_chain_io_defines(chain)
    assert "reverb_circle::wrapper_create" in _build_chain_create(chain)
    assert "delay_circle::wrapper_destroy" in _build_chain_destroy(chain)
    assert "reverb_circle::wrapper_set_param" in _build_chain_set_param(chain)
    assert "m_pScratchA" in _build_chain_perform(chain, max_channels=3)
    assert "channel == 1" in _build_chain_midi_dispatch(chain, graph)
    assert "_ext_circle_0.o: CPPFLAGS +=" in _build_chain_per_node_flags(chain)


def test_chain_project_generation_smoke(
    gigaverb_export: Path,
    spectraldelayfb_export: Path,
    tmp_path: Path,
) -> None:
    """A chain project can be generated from real exports."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
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
    graph = parse_graph(graph_path)
    chain = resolve_chain(
        graph,
        {
            "gigaverb": gigaverb_export,
            "spectraldelayfb": spectraldelayfb_export,
        },
        "0.8.0",
    )

    output_dir = tmp_path / "chain_project"
    output_dir.mkdir()

    CirclePlatform().generate_chain_project(chain, graph, output_dir, "mychain")

    assert (output_dir / "gen_ext_circle.cpp").is_file()
    assert (output_dir / "_ext_circle_0.cpp").is_file()
    assert "libusb.a" in (output_dir / "Makefile").read_text(encoding="utf-8")


def test_chain_invalid_board_raises(
    gigaverb_export: Path,
    tmp_path: Path,
) -> None:
    """Invalid board names are rejected."""
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": {"reverb": {"export": "gigaverb"}},
                "connections": [
                    ["audio_in", "reverb"],
                    ["reverb", "audio_out"],
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = parse_graph(graph_path)
    chain = resolve_chain(graph, {"gigaverb": gigaverb_export}, "0.8.0")

    output_dir = tmp_path / "bad_board"
    output_dir.mkdir()

    config = ProjectConfig(name="test", platform="circle", board="invalid-board")
    with pytest.raises(ProjectError):
        CirclePlatform().generate_chain_project(
            chain,
            graph,
            output_dir,
            "test",
            config,
        )
