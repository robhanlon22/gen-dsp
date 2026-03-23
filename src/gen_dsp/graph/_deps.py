"""Shared dependency helpers for graph analysis."""

from __future__ import annotations

from collections import defaultdict

from gen_dsp.graph.models import Graph, History


def is_feedback_edge(node: object, field_name: str) -> bool:
    """Return True if a field is a feedback edge rather than a data dependency."""
    return isinstance(node, History) and field_name == "input"


def build_forward_deps(graph: Graph) -> dict[str, set[str]]:
    """
    Build forward dependency map: {node_id: set of node_ids it depends on}.

    Excludes feedback edges (History.input) and non-node references such as
    audio inputs and parameter names.
    """
    node_ids = {node.id for node in graph.nodes}
    deps: dict[str, set[str]] = defaultdict(set)
    for node in graph.nodes:
        nid = node.id
        for field_name, value in node.__dict__.items():
            if field_name in ("id", "op"):
                continue
            if isinstance(value, list):
                deps[nid].update(
                    item for item in value if isinstance(item, str) and item in node_ids
                )
                continue
            if (
                isinstance(value, str)
                and value in node_ids
                and not is_feedback_edge(node, field_name)
            ):
                deps[nid].add(value)
    return deps
