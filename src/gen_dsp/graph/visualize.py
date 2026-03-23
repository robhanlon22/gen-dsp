"""Graphviz DOT visualization for DSP graphs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from gen_dsp.graph._deps import is_feedback_edge
from gen_dsp.graph.models import (
    ADSR,
    SVF,
    Accum,
    Allpass,
    BinOp,
    Biquad,
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    Change,
    Clamp,
    Compare,
    Constant,
    Counter,
    Cycle,
    DCBlock,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Elapsed,
    Fold,
    GateOut,
    GateRoute,
    Graph,
    History,
    Latch,
    Lookup,
    Mix,
    MulAccum,
    NamedConstant,
    Noise,
    OnePole,
    Pass,
    Peek,
    Phasor,
    PulseOsc,
    RateDiv,
    SampleHold,
    SampleRate,
    SawOsc,
    Scale,
    Select,
    Selector,
    SinOsc,
    Slide,
    SmoothParam,
    Smoothstep,
    Splat,
    Subgraph,
    TriOsc,
    UnaryOp,
    Wave,
    Wrap,
)

try:
    import graphviz
except ModuleNotFoundError:
    graphviz = None


def _box(label: str, color: str = "#fde0c8") -> tuple[str, str, str]:
    return "box", color, label


def _diamond(label: str) -> tuple[str, str, str]:
    return "diamond", "#fff3cd", label


def _box3d(label: str, color: str = "#fde0c8") -> tuple[str, str, str]:
    return "box3d", color, label


_NODE_ATTR_BUILDERS = {
    BinOp: lambda node: _box(f"{node.id}\\n{node.op}", "#fff3cd"),
    UnaryOp: lambda node: _box(f"{node.id}\\n{node.op}", "#fff3cd"),
    Clamp: lambda node: _box(f"{node.id}\\nclamp", "#fff3cd"),
    Constant: lambda node: _box(f"{node.id}\\n{node.value}", "#e9ecef"),
    History: lambda node: _box(f"{node.id}\\nz^-1"),
    DelayLine: lambda node: _box3d(f"{node.id}\\ndelay[{node.max_samples}]"),
    DelayRead: lambda node: _box(f"{node.id}\\nread"),
    DelayWrite: lambda node: _box(f"{node.id}\\nwrite"),
    Phasor: lambda node: _box(f"{node.id}\\nphasor", "#e2d5f1"),
    Noise: lambda node: _box(f"{node.id}\\nnoise", "#e2d5f1"),
    Compare: lambda node: _diamond(f"{node.id}\\n{node.op}"),
    Select: lambda node: _diamond(f"{node.id}\\nselect"),
    Wrap: lambda node: _box(f"{node.id}\\nwrap", "#fff3cd"),
    Fold: lambda node: _box(f"{node.id}\\nfold", "#fff3cd"),
    Mix: lambda node: _box(f"{node.id}\\nmix", "#fff3cd"),
    Delta: lambda node: _box(f"{node.id}\\ndelta"),
    Change: lambda node: _box(f"{node.id}\\nchange"),
    Biquad: lambda node: _box(f"{node.id}\\nbiquad"),
    SVF: lambda node: _box(f"{node.id}\\nsvf({node.mode})"),
    OnePole: lambda node: _box(f"{node.id}\\nonepole"),
    DCBlock: lambda node: _box(f"{node.id}\\ndcblock"),
    Allpass: lambda node: _box(f"{node.id}\\nallpass"),
    SinOsc: lambda node: _box(f"{node.id}\\nsinosc", "#e2d5f1"),
    TriOsc: lambda node: _box(f"{node.id}\\ntriosc", "#e2d5f1"),
    SawOsc: lambda node: _box(f"{node.id}\\nsawosc", "#e2d5f1"),
    PulseOsc: lambda node: _box(f"{node.id}\\npulseosc", "#e2d5f1"),
    SampleHold: lambda node: _box(f"{node.id}\\nsample_hold"),
    Latch: lambda node: _box(f"{node.id}\\nlatch"),
    Accum: lambda node: _box(f"{node.id}\\naccum"),
    Counter: lambda node: _box(f"{node.id}\\ncounter"),
    Buffer: lambda node: _box3d(f"{node.id}\\nbuffer[{node.size}]"),
    BufRead: lambda node: _box(f"{node.id}\\nbuf_read"),
    BufWrite: lambda node: _box(f"{node.id}\\nbuf_write"),
    Splat: lambda node: _box(f"{node.id}\\nsplat"),
    BufSize: lambda node: _box(f"{node.id}\\nbuf_size"),
    Cycle: lambda node: _box(f"{node.id}\\ncycle"),
    Wave: lambda node: _box(f"{node.id}\\nwave"),
    Lookup: lambda node: _box(f"{node.id}\\nlookup"),
    Elapsed: lambda node: _box(f"{node.id}\\nelapsed"),
    MulAccum: lambda node: _box(f"{node.id}\\nmulaccum"),
    RateDiv: lambda node: _box(f"{node.id}\\nrate_div"),
    Scale: lambda node: _box(f"{node.id}\\nscale", "#fff3cd"),
    SmoothParam: lambda node: _box(f"{node.id}\\nsmooth"),
    Slide: lambda node: _box(f"{node.id}\\nslide"),
    ADSR: lambda node: _box(f"{node.id}\\nadsr"),
    Peek: lambda node: _box(f"{node.id}\\npeek", "#d4edda"),
    Pass: lambda node: _box(f"{node.id}\\npass", "#fff3cd"),
    NamedConstant: lambda node: _box(f"{node.id}\\n{node.op}", "#e9ecef"),
    SampleRate: lambda node: _box(f"{node.id}\\nsamplerate", "#e9ecef"),
    Smoothstep: lambda node: _box(f"{node.id}\\nsmoothstep", "#fff3cd"),
    GateRoute: lambda node: _box(f"{node.id}\\ngate {node.count}", "#fff3cd"),
    GateOut: lambda node: _box(f"{node.id}\\ngate_out {node.channel}", "#fff3cd"),
    Selector: lambda node: _box(f"{node.id}\\nselector", "#fff3cd"),
    Subgraph: lambda node: _box3d(
        (
            f"{node.id}\\nsubgraph "
            f"({len(node.graph.inputs)}in/{len(node.graph.outputs)}out)"
        ),
        "#cce5ff",
    ),
}


def _node_attrs(node: object) -> tuple[str, str, str]:
    """Return (shape, fillcolor, label) for a graph node."""
    builder = _NODE_ATTR_BUILDERS.get(type(node))
    if builder is not None:
        return builder(node)
    return "box", "#ffffff", str(getattr(node, "id", "?"))


def _iter_refs(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        refs: list[str] = []
        for item in value.values():
            refs.extend(_iter_refs(item))
        return refs
    if isinstance(value, list):
        refs: list[str] = []
        for item in value:
            refs.extend(_iter_refs(item))
        return refs
    return []


def _emit_node_edges(node: object, all_ids: set[str], w: Callable[[str], None]) -> None:
    for field_name, value in node.__dict__.items():
        if field_name in ("id", "op"):
            continue
        if not isinstance(value, (str, list, dict)):
            continue
        for ref in _iter_refs(value):
            if ref not in all_ids:
                continue
            if is_feedback_edge(node, field_name):
                w(f'    "{ref}" -> "{node.id}" [style=dashed label="z^-1"];')
            else:
                w(f'    "{ref}" -> "{node.id}";')


def _emit_graph_nodes(graph: Graph, w: Callable[[str], None]) -> None:
    for inp in graph.inputs:
        w(
            f'    "{inp.id}" [shape=box style="rounded,filled"'
            f' fillcolor="#d4edda" label="{inp.id}"];'
        )
    for out in graph.outputs:
        w(
            f'    "{out.id}" [shape=box style="rounded,filled"'
            f' fillcolor="#f8d7da"'
            f' label="{out.id}"];'
        )
    for p in graph.params:
        label = f"{p.name}\\n[{p.min}, {p.max}]\\ndefault={p.default}"
        w(
            f'    "{p.name}" [shape=ellipse style=filled'
            f' fillcolor="#cce5ff"'
            f' label="{label}"];'
        )
    for node in graph.nodes:
        shape, color, label = _node_attrs(node)
        w(
            f'    "{node.id}" [shape={shape} style=filled'
            f' fillcolor="{color}"'
            f' label="{label}"];'
        )


def _emit_graph_edges(
    graph: Graph,
    all_ids: set[str],
    w: Callable[[str], None],
) -> None:
    for node in graph.nodes:
        _emit_node_edges(node, all_ids, w)
    for out in graph.outputs:
        w(f'    "{out.source}" -> "{out.id}";')


def graph_to_dot(graph: Graph) -> str:
    """Convert a DSP graph to a Graphviz DOT string."""
    lines: list[str] = []
    w = lines.append

    w(f'digraph "{graph.name}" {{')
    w("    rankdir=LR;")
    w('    node [fontname="Helvetica" fontsize=10];')
    w("")

    all_ids: set[str] = {inp.id for inp in graph.inputs}
    all_ids.update(out.id for out in graph.outputs)
    all_ids.update(p.name for p in graph.params)
    all_ids.update(node.id for node in graph.nodes)

    _emit_graph_nodes(graph, w)

    w("")
    _emit_graph_edges(graph, all_ids, w)

    w("}")
    return "\n".join(lines) + "\n"


def graph_to_dot_file(graph: Graph, output_dir: str | Path) -> Path:
    """
    Write a DOT file for the graph to output_dir/{name}.dot.

    If the ``dot`` binary is on PATH, also renders a PDF to
    ``output_dir/{name}.pdf``.

    Returns the path to the written ``.dot`` file.
    """
    dot_src = graph_to_dot(graph)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dot_path = out / f"{graph.name}.dot"
    dot_path.write_text(dot_src)

    dot_bin = shutil.which("dot")
    if dot_bin is not None and graphviz is not None:
        graphviz.Source(dot_src).render(
            filename=graph.name,
            directory=str(out),
            format="pdf",
            cleanup=True,
        )

    return dot_path
