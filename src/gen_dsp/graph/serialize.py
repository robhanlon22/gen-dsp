"""Serialize a Graph to .gdsp DSL source."""

from __future__ import annotations

from dataclasses import dataclass

from gen_dsp.graph.models import (
    SVF,
    BinOp,
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    Compare,
    Constant,
    Cycle,
    DelayLine,
    DelayRead,
    DelayWrite,
    GateOut,
    GateRoute,
    Graph,
    History,
    Lookup,
    NamedConstant,
    Node,
    SampleRate,
    Selector,
    Splat,
    UnaryOp,
    Wave,
)
from gen_dsp.graph.toposort import toposort

Ref = str | float

DEFAULT_SAMPLE_RATE = 44100
_NUM_LARGE_INT_LIMIT = 1e15

# BinOp ops that use infix syntax
_INFIX_OPS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "pow": "**",
}

# Compare ops -> infix symbols
_CMP_OPS: dict[str, str] = {
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
    "eq": "==",
    "neq": "!=",
}

# Builtin functions: op -> list of positional field names
_BUILTIN_FIELDS: dict[str, list[str]] = {
    "phasor": ["freq"],
    "sinosc": ["freq"],
    "triosc": ["freq"],
    "sawosc": ["freq"],
    "pulseosc": ["freq", "width"],
    "noise": [],
    "onepole": ["a", "coeff"],
    "dcblock": ["a"],
    "allpass": ["a", "coeff"],
    "biquad": ["a", "b0", "b1", "b2", "a1", "a2"],
    "svf": ["a", "freq", "q"],
    "clamp": ["a", "lo", "hi"],
    "wrap": ["a", "lo", "hi"],
    "fold": ["a", "lo", "hi"],
    "scale": ["a", "in_lo", "in_hi", "out_lo", "out_hi"],
    "smoothstep": ["a", "edge0", "edge1"],
    "mix": ["a", "b", "t"],
    "select": ["cond", "a", "b"],
    "delta": ["a"],
    "change": ["a"],
    "sample_hold": ["a", "trig"],
    "latch": ["a", "trig"],
    "accum": ["incr", "reset"],
    "mulaccum": ["incr", "reset"],
    "counter": ["trig", "max"],
    "elapsed": [],
    "rate_div": ["a", "divisor"],
    "smooth": ["a", "coeff"],
    "slide": ["a", "up", "down"],
    "adsr": ["gate", "attack", "decay", "sustain", "release"],
    "pass": ["a"],
    "peek": ["a"],
    "samplerate": [],
    "cycle": ["buffer", "phase"],
    "wave": ["buffer", "phase"],
    "lookup": ["buffer", "index"],
    "buf_read": ["buffer", "index"],
    "buf_size": ["buffer"],
    "gate_route": ["a", "index"],
    "gate_out": ["gate", "channel"],
    "selector": ["index"],
}


@dataclass(frozen=True)
class _SerializationContext:
    graph: Graph
    indent: str
    control_set: set[str]
    const_map: dict[str, float]
    named_const_map: dict[str, str]


def _format_num(v: float) -> str:
    """Format a float as a clean numeric literal."""
    if v == int(v) and abs(v) < _NUM_LARGE_INT_LIMIT:
        return str(int(v))
    s = f"{v:.15g}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
        if "." not in s:
            s += ".0"
    return s


def _format_ref(
    ref: Ref,
    const_map: dict[str, float],
    named_const_map: dict[str, str] | None = None,
) -> str:
    """Format a Ref (str|float) for .gdsp output, inlining constants."""
    if isinstance(ref, (int, float)):
        return _format_num(float(ref))
    if ref in const_map:
        return _format_num(const_map[ref])
    if named_const_map and ref in named_const_map:
        return named_const_map[ref]
    return ref


def _append_header(lines: list[str], graph: Graph, indent: str) -> None:
    opts: list[str] = []
    if graph.sample_rate != DEFAULT_SAMPLE_RATE:
        opts.append(f"sr={_format_num(float(graph.sample_rate))}")
    if graph.control_interval != 0:
        opts.append(f"control={graph.control_interval}")
    opt_str = f" ({', '.join(opts)})" if opts else ""
    lines.append(f"graph {graph.name}{opt_str} {{")
    if graph.inputs:
        names = ", ".join(inp.id for inp in graph.inputs)
        lines.append(f"{indent}in {names}")


def _append_params(lines: list[str], graph: Graph, indent: str) -> None:
    lines.extend(
        f"{indent}param {p.name} {_format_num(p.min)}..{_format_num(p.max)}"
        f" = {_format_num(p.default)}"
        for p in graph.params
    )


def _append_memory_declarations(
    lines: list[str],
    graph: Graph,
    indent: str,
) -> None:
    for node in graph.nodes:
        if isinstance(node, Buffer):
            fill_part = f" fill={node.fill}" if node.fill != "zeros" else ""
            lines.append(f"{indent}buffer {node.id} {node.size}{fill_part}")
        elif isinstance(node, DelayLine):
            lines.append(f"{indent}delay {node.id} {node.max_samples}")
        elif isinstance(node, History):
            lines.append(f"{indent}history {node.id} = {_format_num(node.init)}")


def _append_processing_lines(
    lines: list[str],
    ctx: _SerializationContext,
) -> list[tuple[str, str]]:
    history_writes: list[tuple[str, str]] = []
    for node in toposort(ctx.graph):
        if isinstance(node, Constant):
            continue
        if isinstance(node, (Buffer, DelayLine)):
            continue
        if isinstance(node, History):
            if node.input:
                history_writes.append((node.id, node.input))
            continue

        expr = _node_to_expr(node, ctx.const_map, ctx.named_const_map)
        if expr is None:
            continue

        control_prefix = "@control " if node.id in ctx.control_set else ""
        if isinstance(node, (DelayWrite, BufWrite, Splat)):
            lines.append(f"{ctx.indent}{control_prefix}{expr}")
        else:
            lines.append(f"{ctx.indent}{control_prefix}{node.id} = {expr}")
    return history_writes


def _append_history_writes(
    lines: list[str],
    indent: str,
    history_writes: list[tuple[str, str]],
    const_map: dict[str, float],
    named_const_map: dict[str, str],
) -> None:
    for hist_id, input_ref in history_writes:
        ref_str = _format_ref(input_ref, const_map, named_const_map)
        lines.append(f"{indent}{hist_id} <- {ref_str}")


def _append_outputs(
    lines: list[str],
    graph: Graph,
    indent: str,
    const_map: dict[str, float],
    named_const_map: dict[str, str],
) -> None:
    for out in graph.outputs:
        src = _format_ref(out.source, const_map, named_const_map) if out.source else ""
        lines.append(f"{indent}out {out.id} = {src}")


def graph_to_gdsp(graph: Graph) -> str:
    """
    Serialize a gen-dsp Graph to .gdsp DSL source.

    Produces a human-readable .gdsp string that, when parsed back with
    ``gen_dsp.graph.dsl.parse()``, yields an equivalent Graph.

    Constant nodes are inlined as numeric literals.  NamedConstant nodes
    are inlined as bare identifiers (``pi``, ``e``, etc.).  History
    feedback writes use the ``<-`` operator.  Processing nodes are
    emitted in topological order.
    """
    lines: list[str] = []
    indent = "    "
    control_set = set(graph.control_nodes)

    # Build constant inlining map
    const_map: dict[str, float] = {}
    named_const_map: dict[str, str] = {}
    for node in graph.nodes:
        if isinstance(node, Constant):
            const_map[node.id] = node.value
        elif isinstance(node, NamedConstant):
            named_const_map[node.id] = node.op

    try:
        toposort(graph)
    except ValueError:
        list(graph.nodes)

    _append_header(lines, graph, indent)
    _append_params(lines, graph, indent)
    _append_memory_declarations(lines, graph, indent)

    # Blank line before processing
    if (
        graph.inputs
        or graph.params
        or any(isinstance(n, (Buffer, DelayLine, History)) for n in graph.nodes)
    ):
        lines.append("")

    history_writes = _append_processing_lines(
        lines,
        _SerializationContext(
            graph=graph,
            indent=indent,
            control_set=control_set,
            const_map=const_map,
            named_const_map=named_const_map,
        ),
    )
    _append_history_writes(lines, indent, history_writes, const_map, named_const_map)

    # Blank line before outputs
    if graph.outputs:
        lines.append("")

    _append_outputs(lines, graph, indent, const_map, named_const_map)

    lines.append("}")
    return "\n".join(lines) + "\n"


def _node_to_expr_binop(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if not isinstance(node, BinOp):
        return None
    left = _format_ref(node.a, const_map, named_const_map)
    right = _format_ref(node.b, const_map, named_const_map)
    if node.op in _INFIX_OPS:
        return f"{left} {_INFIX_OPS[node.op]} {right}"
    return f"{node.op}({left}, {right})"


def _node_to_expr_compare(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if not isinstance(node, Compare):
        return None
    return (
        f"{_format_ref(node.a, const_map, named_const_map)} "
        f"{_CMP_OPS[node.op]} "
        f"{_format_ref(node.b, const_map, named_const_map)}"
    )


def _node_to_expr_unary(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if not isinstance(node, UnaryOp):
        return None
    return f"{node.op}({_format_ref(node.a, const_map, named_const_map)})"


def _node_to_expr_simple(node: Node) -> str | None:
    if isinstance(node, NamedConstant):
        return node.op
    if isinstance(node, SampleRate):
        return "samplerate()"
    return None


def _node_to_expr_statement(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if isinstance(node, DelayWrite):
        return (
            f"delay_write {node.delay} "
            f"({_format_ref(node.value, const_map, named_const_map)})"
        )
    if isinstance(node, DelayRead):
        interp_part = f", interp={node.interp}" if node.interp != "none" else ""
        return (
            f"delay_read {node.delay} "
            f"({_format_ref(node.tap, const_map, named_const_map)}{interp_part})"
        )
    if isinstance(node, BufWrite):
        return (
            f"buf_write({node.buffer}, "
            f"{_format_ref(node.index, const_map, named_const_map)}, "
            f"{_format_ref(node.value, const_map, named_const_map)})"
        )
    if isinstance(node, Splat):
        return (
            f"splat({node.buffer}, "
            f"{_format_ref(node.index, const_map, named_const_map)}, "
            f"{_format_ref(node.value, const_map, named_const_map)})"
        )
    return None


def _node_to_expr_buffer(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if isinstance(node, BufRead):
        interp_part = f", interp={node.interp}" if node.interp != "none" else ""
        return (
            f"buf_read({node.buffer}, "
            f"{_format_ref(node.index, const_map, named_const_map)}{interp_part})"
        )
    if isinstance(node, BufSize):
        return f"buf_size({node.buffer})"
    if isinstance(node, Cycle):
        return (
            f"cycle({node.buffer}, "
            f"{_format_ref(node.phase, const_map, named_const_map)})"
        )
    if isinstance(node, Wave):
        return (
            f"wave({node.buffer}, "
            f"{_format_ref(node.phase, const_map, named_const_map)})"
        )
    if isinstance(node, Lookup):
        return (
            f"lookup({node.buffer}, "
            f"{_format_ref(node.index, const_map, named_const_map)})"
        )
    return None


def _node_to_expr_control(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    if isinstance(node, SVF):
        mode_part = f", mode={node.mode}" if node.mode != "lp" else ""
        return (
            f"svf({_format_ref(node.a, const_map, named_const_map)}, "
            f"{_format_ref(node.freq, const_map, named_const_map)}, "
            f"{_format_ref(node.q, const_map, named_const_map)}{mode_part})"
        )
    if isinstance(node, Selector):
        inputs_str = ", ".join(
            _format_ref(item, const_map, named_const_map) for item in node.inputs
        )
        return (
            f"selector({_format_ref(node.index, const_map, named_const_map)}, "
            f"{inputs_str})"
        )
    if isinstance(node, GateRoute):
        return (
            f"gate_route({_format_ref(node.a, const_map, named_const_map)}, "
            f"{_format_ref(node.index, const_map, named_const_map)}, {node.count})"
        )
    if isinstance(node, GateOut):
        return f"gate_out({node.gate}, {node.channel})"
    return None


def _node_to_expr_builtin(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str | None:
    op = getattr(node, "op", None)
    if op not in _BUILTIN_FIELDS:
        return None
    args: list[str] = []
    for fname in _BUILTIN_FIELDS[op]:
        val = getattr(node, fname, None)
        if val is None:
            continue
        if isinstance(val, list):
            args.extend(
                _format_ref(item, const_map, named_const_map)
                for item in val
                if isinstance(item, (str, float))
            )
        elif isinstance(val, (str, float)):
            args.append(_format_ref(val, const_map, named_const_map))
        else:
            args.append(str(val))
    return f"{op}({', '.join(args)})"


def _node_to_expr_fallback(
    node: Node, const_map: dict[str, float], named_const_map: dict[str, str] | None
) -> str:
    fields_data = {k: v for k, v in node.__dict__.items() if k not in ("id", "op")}
    args_list: list[str] = []
    for v in fields_data.values():
        if isinstance(v, (str, float)):
            args_list.append(_format_ref(v, const_map, named_const_map))
        elif isinstance(v, int):
            args_list.append(str(v))
        elif isinstance(v, list):
            args_list.extend(
                _format_ref(item, const_map, named_const_map)
                for item in v
                if isinstance(item, (str, float))
            )
    return f"{node.op}({', '.join(args_list)})"


def _node_to_expr(
    node: Node,
    const_map: dict[str, float],
    named_const_map: dict[str, str] | None = None,
) -> str | None:
    """Convert a single node to its .gdsp expression string."""
    for formatter in (
        _node_to_expr_binop,
        _node_to_expr_compare,
        _node_to_expr_unary,
        _node_to_expr_simple,
        _node_to_expr_statement,
        _node_to_expr_buffer,
        _node_to_expr_control,
        _node_to_expr_builtin,
    ):
        expr = formatter(node, const_map, named_const_map)
        if expr is not None:
            return expr
    return _node_to_expr_fallback(node, const_map, named_const_map)
