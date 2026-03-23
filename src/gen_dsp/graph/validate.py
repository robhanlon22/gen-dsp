"""Validation helpers for DSP graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Self

from gen_dsp.graph._deps import build_forward_deps
from gen_dsp.graph.models import (
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    Cycle,
    DelayLine,
    DelayRead,
    DelayWrite,
    GateOut,
    GateRoute,
    Graph,
    History,
    Lookup,
    Splat,
    Subgraph,
    Wave,
)
from gen_dsp.graph.optimize import _STATEFUL_TYPES
from gen_dsp.graph.subgraph import expand_subgraphs

_NON_REF_FIELDS = {
    "id",
    "op",
    "interp",
    "mode",
    "output",
    "count",
    "channel",
    "fill",
}


class GraphValidationError(str):
    """
    A structured validation error that behaves as a plain string.

    Subclasses ``str`` so all existing call sites (``== []``, ``in``,
    ``"; ".join(errors)``, ``print(f"error: {err}")``) work unchanged.

    Attributes:
    ----------
    kind : str
        Machine-readable error category.  Stable values:

        ``"duplicate_id"``
            Two nodes share the same ID.
        ``"id_collision"``
            A node ID equals an audio input ID or param name.
        ``"dangling_ref"``
            A field references an ID that does not exist.
        ``"bad_output_source"``
            ``AudioOutput.source`` does not reference a node.
        ``"missing_delay_line"``
            ``DelayRead``/``DelayWrite`` references a non-existent ``DelayLine``.
        ``"missing_buffer"``
            A buffer consumer (``BufRead``, ``BufWrite``, ``BufSize``, ``Splat``,
            ``Cycle``, ``Wave``, ``Lookup``) references a non-existent ``Buffer``.
        ``"missing_gate_route"``
            ``GateOut.gate`` references a non-existent ``GateRoute``.
        ``"gate_channel_range"``
            ``GateOut.channel`` is outside ``[1, gate_route.count]``.
        ``"invalid_control_node"``
            An ID in ``Graph.control_nodes`` is not a node ID.
        ``"control_audio_dep"``
            A control-rate node depends on an audio input.
        ``"control_rate_dep"``
            A control-rate node depends on an audio-rate node.
        ``"cycle"``
            Graph contains a pure cycle (not through ``History`` or delay feedback).
        ``"expansion_error"``
            ``expand_subgraphs()`` raised a ``ValueError`` (malformed ``Subgraph``).
        ``"unmapped_param"`` *(warning)*
            A subgraph param uses its default because it was not mapped at the
            call site.  Only emitted when ``warn_unmapped_params=True``.

    node_id : str | None
        ID of the offending node, if applicable.
    field_name : str | None
        Name of the offending field, if applicable.
    severity : str
        ``"error"`` or ``"warning"``.

    """

    __slots__ = ("field_name", "kind", "node_id", "severity")

    kind: str
    node_id: str | None
    field_name: str | None
    severity: str  # "error" | "warning"

    def __new__(
        cls,
        kind: str,
        message: str,
        *,
        node_id: str | None = None,
        field_name: str | None = None,
        severity: str = "error",
    ) -> Self:
        """Create a validation error with attached structured metadata."""
        obj = str.__new__(cls, message)
        obj.kind = kind
        obj.node_id = node_id
        obj.field_name = field_name
        obj.severity = severity
        return obj


def _collect_refs(node: object) -> list[str]:
    """Return string refs from input fields, excluding `id` and `op`."""
    refs: list[str] = []
    for field_name, value in node.__dict__.items():
        if field_name in ("id", "op"):
            continue
        if isinstance(value, list):
            refs.extend(item for item in value if isinstance(item, str))
        elif isinstance(value, str):
            refs.append(value)
    return refs


def _is_invariant_value(value: object, param_names: set[str]) -> bool:
    """Return True when a field value cannot vary at audio rate."""
    if isinstance(value, float):
        return True
    if isinstance(value, str):
        return value in param_names
    if isinstance(value, list):
        return all(_is_invariant_value(item, param_names) for item in value)
    return False


def _check_unmapped_params(
    nodes: list[object],
    prefix: str,
    warnings: list[GraphValidationError],
) -> None:
    """Recursively check for unmapped subgraph params, emitting warnings."""
    for node in nodes:
        if not isinstance(node, Subgraph):
            continue
        inner_param_names = {p.name for p in node.graph.params}
        mapped = set(node.params.keys())
        unmapped = inner_param_names - mapped
        sg_id = prefix + node.id if prefix else node.id
        for pname in sorted(unmapped):
            default = next(p.default for p in node.graph.params if p.name == pname)
            message = (
                f"Subgraph '{sg_id}': param '{pname}' not mapped, using default "
                f"{default}"
            )
            warnings.append(
                GraphValidationError(
                    "unmapped_param",
                    message,
                    node_id=sg_id,
                    field_name=pname,
                    severity="warning",
                )
            )
        # Recurse into nested subgraphs
        inner_prefix = (prefix + node.id + "__") if prefix else (node.id + "__")
        _check_unmapped_params(list(node.graph.nodes), inner_prefix, warnings)


def _collect_duplicate_and_collision_errors(
    graph: Graph,
) -> tuple[dict[str, int], set[str], set[str], list[GraphValidationError]]:
    """Check for duplicate node IDs and collisions with inputs or params."""
    errors: list[GraphValidationError] = []
    node_ids: dict[str, int] = {}
    input_ids = {inp.id for inp in graph.inputs}
    param_names = {p.name for p in graph.params}

    for node in graph.nodes:
        nid = node.id
        if nid in node_ids:
            errors.append(
                GraphValidationError(
                    "duplicate_id",
                    f"Duplicate node ID: '{nid}'",
                    node_id=nid,
                )
            )
        node_ids[nid] = 0

        if nid in input_ids:
            errors.append(
                GraphValidationError(
                    "id_collision",
                    f"Node ID '{nid}' collides with audio input ID",
                    node_id=nid,
                )
            )
        if nid in param_names:
            errors.append(
                GraphValidationError(
                    "id_collision",
                    f"Node ID '{nid}' collides with param name",
                    node_id=nid,
                )
            )

    return node_ids, input_ids, param_names, errors


def _collect_reference_errors(
    graph: Graph,
    all_ids: set[str],
) -> list[GraphValidationError]:
    """Check that every string reference resolves to a known ID."""
    errors: list[GraphValidationError] = []
    for node in graph.nodes:
        for field_name, value in node.__dict__.items():
            if field_name in _NON_REF_FIELDS:
                continue
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, str) and item not in all_ids:
                        errors.append(
                            GraphValidationError(
                                "dangling_ref",
                                f"Node '{node.id}' field '{field_name}[{idx}]' "
                                f"references unknown ID '{item}'",
                                node_id=node.id,
                                field_name=field_name,
                            )
                        )
            elif isinstance(value, str) and value not in all_ids:
                errors.append(
                    GraphValidationError(
                        "dangling_ref",
                        f"Node '{node.id}' field '{field_name}' references unknown ID "
                        f"'{value}'",
                        node_id=node.id,
                        field_name=field_name,
                    )
                )
    return errors


def _collect_output_errors(
    graph: Graph,
    node_ids: set[str],
) -> list[GraphValidationError]:
    """Check that every output source references a node."""
    errors: list[GraphValidationError] = []
    for out in graph.outputs:
        if out.source in node_ids:
            continue
        errors.append(
            GraphValidationError(
                "bad_output_source",
                f"Output '{out.id}' source '{out.source}' does not reference a node",
                node_id=out.id,
                field_name="source",
            )
        )
    return errors


def _collect_delay_errors(graph: Graph) -> list[GraphValidationError]:
    """Check that delay nodes reference existing delay lines."""
    errors: list[GraphValidationError] = []
    delay_line_ids = {node.id for node in graph.nodes if isinstance(node, DelayLine)}
    for node in graph.nodes:
        if isinstance(node, DelayRead) and node.delay not in delay_line_ids:
            errors.append(
                GraphValidationError(
                    "missing_delay_line",
                    f"DelayRead '{node.id}' references non-existent delay line "
                    f"'{node.delay}'",
                    node_id=node.id,
                    field_name="delay",
                )
            )
        if isinstance(node, DelayWrite) and node.delay not in delay_line_ids:
            errors.append(
                GraphValidationError(
                    "missing_delay_line",
                    f"DelayWrite '{node.id}' references non-existent delay line "
                    f"'{node.delay}'",
                    node_id=node.id,
                    field_name="delay",
                )
            )
    return errors


def _collect_buffer_errors(graph: Graph) -> list[GraphValidationError]:
    """Check that buffer nodes reference existing buffers."""
    errors: list[GraphValidationError] = []
    buffer_ids = {node.id for node in graph.nodes if isinstance(node, Buffer)}
    for node in graph.nodes:
        if isinstance(node, BufRead) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"BufRead '{node.id}' references non-existent buffer "
                    f"'{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, BufWrite) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"BufWrite '{node.id}' references non-existent buffer "
                    f"'{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, Splat) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"Splat '{node.id}' references non-existent buffer '{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, BufSize) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"BufSize '{node.id}' references non-existent buffer "
                    f"'{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, Cycle) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"Cycle '{node.id}' references non-existent buffer '{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, Wave) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"Wave '{node.id}' references non-existent buffer '{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
        if isinstance(node, Lookup) and node.buffer not in buffer_ids:
            errors.append(
                GraphValidationError(
                    "missing_buffer",
                    f"Lookup '{node.id}' references non-existent buffer "
                    f"'{node.buffer}'",
                    node_id=node.id,
                    field_name="buffer",
                )
            )
    return errors


def _collect_gate_errors(graph: Graph) -> list[GraphValidationError]:
    """Check that gate outputs reference a gate route and valid channels."""
    errors: list[GraphValidationError] = []
    gate_route_map = {
        node.id: node for node in graph.nodes if isinstance(node, GateRoute)
    }
    for node in graph.nodes:
        if not isinstance(node, GateOut):
            continue
        gate_route = gate_route_map.get(node.gate)
        if gate_route is None:
            errors.append(
                GraphValidationError(
                    "missing_gate_route",
                    f"GateOut '{node.id}' references non-existent gate route "
                    f"'{node.gate}'",
                    node_id=node.id,
                    field_name="gate",
                )
            )
            continue
        if node.channel < 1 or node.channel > gate_route.count:
            errors.append(
                GraphValidationError(
                    "gate_channel_range",
                    f"GateOut '{node.id}' channel {node.channel} out of range "
                    f"[1, {gate_route.count}]",
                    node_id=node.id,
                    field_name="channel",
                )
            )
    return errors


def _collect_control_invariant_ids(
    graph: Graph,
    param_names: set[str],
) -> set[str]:
    """Return node IDs that are safe to treat as invariant."""
    invariant_ids: set[str] = set()
    for node in graph.nodes:
        if isinstance(node, _STATEFUL_TYPES):
            continue
        is_inv = True
        for fn, val in node.__dict__.items():
            if fn in _NON_REF_FIELDS:
                continue
            if not _is_invariant_value(val, param_names):
                is_inv = False
                break
        if is_inv:
            invariant_ids.add(node.id)
    return invariant_ids


def _collect_control_node_dependency_errors(
    node: object,
    cid: str,
    input_ids: set[str],
    node_id_set: set[str],
    allowed: set[str],
) -> list[GraphValidationError]:
    """Check one control-rate node for audio/control dependencies."""
    errors: list[GraphValidationError] = []
    if isinstance(node, History):
        return errors
    for field_name, value in node.__dict__.items():
        if field_name in _NON_REF_FIELDS:
            continue
        if isinstance(value, list):
            str_refs = [v for v in value if isinstance(v, str)]
        elif isinstance(value, str):
            str_refs = [value]
        else:
            continue
        for ref_val in str_refs:
            if ref_val in input_ids:
                errors.append(
                    GraphValidationError(
                        "control_audio_dep",
                        f"Control-rate node '{cid}' depends on audio input '{ref_val}'",
                        node_id=cid,
                        field_name=field_name,
                    )
                )
            elif ref_val in node_id_set and ref_val not in allowed:
                errors.append(
                    GraphValidationError(
                        "control_rate_dep",
                        f"Control-rate node '{cid}' depends on audio-rate node "
                        f"'{ref_val}'",
                        node_id=cid,
                        field_name=field_name,
                    )
                )
    return errors


def _collect_control_errors(
    graph: Graph,
    node_ids: dict[str, int],
    input_ids: set[str],
    param_names: set[str],
) -> list[GraphValidationError]:
    """Check control-rate node dependencies."""
    if graph.control_interval <= 0 or not graph.control_nodes:
        return []

    errors: list[GraphValidationError] = []
    node_id_set = set(node_ids)
    node_by_id = {n.id: n for n in graph.nodes}

    errors.extend(
        [
            GraphValidationError(
                "invalid_control_node",
                f"control_nodes: '{cid}' is not a node ID",
                node_id=cid,
            )
            for cid in graph.control_nodes
            if cid not in node_id_set
        ]
    )

    invariant_ids = _collect_control_invariant_ids(graph, param_names)
    allowed = set(graph.control_nodes) | param_names | invariant_ids
    for cid in graph.control_nodes:
        node = node_by_id.get(cid)
        if node is None:
            continue
        errors.extend(
            _collect_control_node_dependency_errors(
                node, cid, input_ids, node_id_set, allowed
            )
        )
    return errors


def _collect_cycle_errors(
    graph: Graph,
    node_ids: dict[str, int],
) -> list[GraphValidationError]:
    """Check for pure cycles in the graph."""
    deps = build_forward_deps(graph)
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    reverse: dict[str, list[str]] = defaultdict(list)
    for nid, dep_set in deps.items():
        for dep in dep_set:
            if dep in in_degree:
                in_degree[nid] += 1
                reverse[dep].append(nid)

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for dependent in reverse[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if visited < len(node_ids):
        cycle_nodes = sorted(nid for nid, deg in in_degree.items() if deg > 0)
        message = f"Graph contains a cycle through nodes: {', '.join(cycle_nodes)}"
        return [GraphValidationError("cycle", message)]
    return []


def validate_graph(
    graph: Graph, *, warn_unmapped_params: bool = False
) -> list[GraphValidationError]:
    """
    Validate a DSP graph and return a list of errors (empty = valid).

    When *warn_unmapped_params* is ``True``, warnings for subgraph params
    that silently fall back to defaults are appended after all errors.
    """
    # Collect unmapped-param warnings *before* expansion (needs Subgraph nodes)
    warnings: list[GraphValidationError] = []
    if warn_unmapped_params:
        _check_unmapped_params(list(graph.nodes), "", warnings)

    try:
        graph = expand_subgraphs(graph)
    except ValueError as exc:
        return [GraphValidationError("expansion_error", str(exc))]

    node_ids, input_ids, param_names, errors = _collect_duplicate_and_collision_errors(
        graph
    )
    all_ids = set(node_ids) | input_ids | param_names
    errors.extend(_collect_reference_errors(graph, all_ids))
    errors.extend(_collect_output_errors(graph, set(node_ids)))
    errors.extend(_collect_delay_errors(graph))
    errors.extend(_collect_buffer_errors(graph))
    errors.extend(_collect_gate_errors(graph))
    errors.extend(_collect_control_errors(graph, node_ids, input_ids, param_names))
    errors.extend(_collect_cycle_errors(graph, node_ids))

    errors.extend(warnings)
    return errors
