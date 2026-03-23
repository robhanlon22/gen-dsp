"""Subgraph expansion -- inline Subgraph nodes into flat graphs."""

from __future__ import annotations

from gen_dsp.graph.models import Graph, Node, Subgraph

_NON_REF_FIELDS = frozenset(
    {"id", "op", "interp", "mode", "output", "count", "channel"}
)


def expand_subgraphs(graph: Graph) -> Graph:
    """
    Recursively expand all Subgraph nodes into a flat graph.

    Returns the graph unchanged if it contains no Subgraph nodes.
    Raises ValueError on invalid subgraph wiring.
    """
    if not any(isinstance(n, Subgraph) for n in graph.nodes):
        return graph

    out_nodes: list[Node] = []
    # Maps subgraph ID (and compound IDs) to the expanded output node ID
    output_map: dict[str, str] = {}
    # Collect prefixed control_nodes from inner subgraphs
    new_control_nodes: list[str] = list(graph.control_nodes)

    # Parent namespace sets for collision detection
    parent_param_names = {p.name for p in graph.params}
    parent_input_ids = {inp.id for inp in graph.inputs}

    for node in graph.nodes:
        if isinstance(node, Subgraph):
            expanded_nodes, node_outputs, control_nodes = _expand_one(
                node, parent_param_names, parent_input_ids
            )
            out_nodes.extend(expanded_nodes)
            output_map.update(node_outputs)
            new_control_nodes.extend(control_nodes)
            # Check for namespace collisions with parent params/inputs
        else:
            out_nodes.append(node)

    # Rewrite parent-level refs that point to subgraph IDs
    out_nodes = [_rewrite_refs(n, output_map) for n in out_nodes]

    new_outputs = []
    for out in graph.outputs:
        source = output_map.get(out.source, out.source)
        if source != out.source:
            new_outputs.append(out.model_copy(update={"source": source}))
        else:
            new_outputs.append(out)

    updates: dict[str, object] = {"nodes": out_nodes, "outputs": new_outputs}
    if new_control_nodes != list(graph.control_nodes):
        updates["control_nodes"] = new_control_nodes
    return graph.model_copy(update=updates)


def _expand_one(
    sg: Subgraph,
    parent_param_names: set[str],
    parent_input_ids: set[str],
) -> tuple[list[Node], dict[str, str], list[str]]:
    """Expand a single Subgraph node into nodes, outputs, and control IDs."""
    inner = expand_subgraphs(sg.graph)
    prefix = sg.id + "__"
    _validate_inner_subgraph(sg, inner)
    rewrite_map = _build_rewrite_map(sg, inner, prefix)

    expanded_nodes: list[Node] = []
    for node in inner.nodes:
        new_node = _rewrite_node(node, prefix, rewrite_map)
        if new_node.id in parent_param_names:
            msg = (
                f"Subgraph '{sg.id}': expanded node '{new_node.id}' "
                "collides with parent param"
            )
            raise ValueError(msg)
        if new_node.id in parent_input_ids:
            msg = (
                f"Subgraph '{sg.id}': expanded node '{new_node.id}' "
                "collides with parent input"
            )
            raise ValueError(msg)
        expanded_nodes.append(new_node)

    # Build output_map entries
    node_outputs: dict[str, str] = {}
    # Selected output (or first)
    selected = sg.output or inner.outputs[0].id
    for out in inner.outputs:
        prefixed_source = prefix + out.source
        if out.id == selected:
            node_outputs[sg.id] = prefixed_source
        # Compound ID for all outputs
        node_outputs[sg.id + "__" + out.id] = prefixed_source

    control_nodes = [prefix + cn_id for cn_id in inner.control_nodes]
    return expanded_nodes, node_outputs, control_nodes


def _validate_inner_subgraph(
    sg: Subgraph,
    inner: Graph,
) -> None:
    """Validate that a subgraph expansion is well formed."""
    if not inner.outputs:
        msg = f"Subgraph '{sg.id}': inner graph has no outputs"
        raise ValueError(msg)

    inner_output_ids = {o.id for o in inner.outputs}
    inner_input_ids = {inp.id for inp in inner.inputs}
    inner_param_names = {p.name for p in inner.params}

    if sg.output and sg.output not in inner_output_ids:
        msg = (
            f"Subgraph '{sg.id}': output '{sg.output}' not found in inner graph "
            f"(available: {sorted(inner_output_ids)})"
        )
        raise ValueError(msg)

    missing_inputs = sorted(key for key in sg.inputs if key not in inner_input_ids)
    if missing_inputs:
        msg = (
            f"Subgraph '{sg.id}': input key '{missing_inputs[0]}' not found "
            f"in inner graph (available: {sorted(inner_input_ids)})"
        )
        raise ValueError(msg)

    missing_input_mappings = sorted(
        iid for iid in inner_input_ids if iid not in sg.inputs
    )
    if missing_input_mappings:
        msg = (
            f"Subgraph '{sg.id}': missing input mapping for "
            f"'{missing_input_mappings[0]}'"
        )
        raise ValueError(msg)

    missing_params = sorted(key for key in sg.params if key not in inner_param_names)
    if missing_params:
        msg = (
            f"Subgraph '{sg.id}': param key '{missing_params[0]}' not found "
            f"in inner graph (available: {sorted(inner_param_names)})"
        )
        raise ValueError(msg)


def _build_rewrite_map(
    sg: Subgraph,
    inner: Graph,
    prefix: str,
) -> dict[str, str | float]:
    """Build rewrite map from inner IDs and params to outer references."""
    inner_param_names = {p.name for p in inner.params}
    param_defaults = {p.name: p.default for p in inner.params}
    rewrite_map: dict[str, str | float] = {
        nid: prefix + nid for nid in (n.id for n in inner.nodes)
    }
    rewrite_map.update(sg.inputs)
    rewrite_map.update(
        {
            pname: sg.params[pname] if pname in sg.params else param_defaults[pname]
            for pname in inner_param_names
        }
    )
    return rewrite_map


def _rewrite_node(
    node: Node,
    prefix: str,
    rewrite_map: dict[str, str | float],
) -> Node:
    """Clone a node with prefixed ID and rewritten ref fields."""
    updates: dict[str, object] = {"id": prefix + node.id}
    for field_name, value in node.__dict__.items():
        if field_name in _NON_REF_FIELDS:
            continue
        if isinstance(value, list):
            new_list = [
                rewrite_map.get(v, v) if isinstance(v, str) else v for v in value
            ]
            if new_list != value:
                updates[field_name] = new_list
        elif isinstance(value, str) and value in rewrite_map:
            updates[field_name] = rewrite_map[value]
        elif isinstance(value, (float, int)):
            continue
    return node.model_copy(update=updates)


def _rewrite_refs(node: Node, output_map: dict[str, str]) -> Node:
    """Rewrite parent-level refs pointing to subgraph IDs."""
    updates: dict[str, object] = {}
    for field_name, value in node.__dict__.items():
        if field_name in _NON_REF_FIELDS:
            continue
        if isinstance(value, list):
            new_list = [
                output_map.get(v, v) if isinstance(v, str) else v for v in value
            ]
            if new_list != value:
                updates[field_name] = new_list
        elif isinstance(value, str) and value in output_map:
            updates[field_name] = output_map[value]
    if not updates:
        return node
    return node.model_copy(update=updates)
