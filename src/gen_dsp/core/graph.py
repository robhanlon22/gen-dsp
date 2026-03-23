"""
Graph data model for multi-plugin chain and DAG configurations.

Defines the JSON graph format and validation logic for serial chains
and arbitrary DAGs of gen~ plugins (including built-in mixer nodes).

Typical flow:
    graph.json -> parse_graph() -> GraphConfig
              -> validate_linear_chain() -> errors (if any)     [Phase 1]
              -> extract_chain_order() -> ordered node IDs       [Phase 1]
              -> resolve_chain() -> list[ResolvedChainNode]      [Phase 1]

    Or for DAGs:
              -> validate_dag() -> errors (if any)               [Phase 2]
              -> topological_sort() -> ordered node IDs          [Phase 2]
              -> allocate_edge_buffers() -> EdgeBuffer list       [Phase 2]
              -> resolve_dag() -> list[ResolvedChainNode]        [Phase 2]
"""

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from gen_dsp.core.manifest import Manifest, ParamInfo, manifest_from_export_info
from gen_dsp.core.parser import ExportInfo, GenExportParser
from gen_dsp.errors import ValidationError

_AUDIO_IN = "audio_in"
_AUDIO_OUT = "audio_out"
_MIDI_CHANNEL_MIN = 1
_MIDI_CHANNEL_MAX = 16
_CC_MIN = 0
_CC_MAX = 127
_CONNECTION_PAIR_LENGTH = 2


@dataclass
class Connection:
    """
    A directed edge in the plugin graph.

    Attributes:
        src_node: Source node ID (or "audio_in").
        dst_node: Destination node ID (or "audio_out").
        dst_input_index: Optional input index on the destination node.
            None means sequential from 0. Used for mixer input selection
            (e.g. "mix:1" -> dst_input_index=1).

    """

    src_node: str
    dst_node: str
    dst_input_index: int | None = None


@dataclass
class ChainNodeConfig:
    """Configuration for a single node in the chain graph."""

    id: str
    export: str | None = None
    node_type: str = "gen"  # "gen" or "mixer"
    mixer_inputs: int = 0  # only for mixer nodes
    midi_channel: int | None = None
    cc_map: dict[int, str] = field(default_factory=dict)


@dataclass
class GraphConfig:
    """Parsed graph configuration from JSON."""

    nodes: dict[str, ChainNodeConfig]
    connections: list[Connection]


@dataclass
class EdgeBuffer:
    """
    Buffer allocation for a single edge in the DAG.

    Attributes:
        buffer_id: Unique buffer identifier. Edges from the same source
            share one buffer_id (fan-out = zero-copy).
        src_node: Source node ID.
        dst_node: Destination node ID.
        dst_input_index: Optional input index on the destination.
        num_channels: Channel count (from source output count).

    """

    buffer_id: int
    src_node: str
    dst_node: str
    dst_input_index: int | None
    num_channels: int


@dataclass
class ResolvedChainNode:
    """A chain node with fully resolved export info and manifest."""

    config: ChainNodeConfig
    index: int
    export_info: ExportInfo | None
    manifest: Manifest


def _parse_connection_config(conn: object, index: int) -> Connection:
    """Parse a single connection entry from graph JSON."""
    if not isinstance(conn, (list, tuple)) or len(conn) != _CONNECTION_PAIR_LENGTH:
        msg = f"Connection {index} must be a [from, to] pair, got: {conn}"
        raise ValidationError(msg)
    src = str(conn[0])
    dst_raw = str(conn[1])

    dst_input_index: int | None = None
    if ":" in dst_raw:
        parts = dst_raw.rsplit(":", 1)
        try:
            dst_input_index = int(parts[1])
            dst_raw = parts[0]
        except ValueError as e:
            msg = f"Connection {index}: invalid input index in '{conn[1]}'"
            raise ValidationError(msg) from e

    return Connection(src, dst_raw, dst_input_index)


def _parse_cc_map(node_id: str, raw_cc: object) -> dict[int, str]:
    """Parse the optional CC map for a node."""
    if not isinstance(raw_cc, dict):
        msg = (
            f"Node '{node_id}': 'cc' must be an object mapping CC numbers "
            "to param names"
        )
        raise ValidationError(msg)

    cc_map: dict[int, str] = {}
    for cc_str, param_name in raw_cc.items():
        try:
            cc_num = int(cc_str)
        except ValueError as e:
            msg = f"Node '{node_id}': CC key '{cc_str}' must be an integer"
            raise ValidationError(msg) from e
        cc_map[cc_num] = param_name
    return cc_map


def _parse_mixer_node_config(node_id: str, node_data: dict[str, object]) -> int:
    """Parse mixer-specific node fields."""
    if "inputs" not in node_data:
        msg = f"Node '{node_id}': mixer nodes must have an 'inputs' field"
        raise ValidationError(msg)
    mixer_inputs = node_data["inputs"]
    if not isinstance(mixer_inputs, int) or mixer_inputs < 1:
        msg = f"Node '{node_id}': mixer 'inputs' must be a positive integer"
        raise ValidationError(msg)
    return mixer_inputs


def _parse_gen_node_config(node_id: str, node_data: dict[str, object]) -> str:
    """Parse gen-specific node fields."""
    if "export" not in node_data:
        msg = f"Node '{node_id}' must have an 'export' field"
        raise ValidationError(msg)
    return str(node_data["export"])


def _parse_node_config(node_id: str, node_data: object) -> ChainNodeConfig:
    """Parse a single node entry from graph JSON."""
    if not isinstance(node_data, dict):
        msg = f"Node '{node_id}' must be an object"
        raise ValidationError(msg)

    node_type = node_data.get("type", "gen")
    if node_type == "mixer":
        export = None
        mixer_inputs = _parse_mixer_node_config(node_id, node_data)
    elif node_type == "gen":
        export = _parse_gen_node_config(node_id, node_data)
        mixer_inputs = 0
    else:
        msg = f"Node '{node_id}': unknown node type '{node_type}'"
        raise ValidationError(msg)

    cc_map = _parse_cc_map(node_id, node_data["cc"]) if "cc" in node_data else {}

    midi_channel = node_data.get("midi_channel")
    if midi_channel is not None and not isinstance(midi_channel, int):
        msg = f"Node '{node_id}': 'midi_channel' must be an integer"
        raise ValidationError(msg)

    return ChainNodeConfig(
        id=node_id,
        export=export,
        node_type=node_type,
        mixer_inputs=mixer_inputs,
        midi_channel=midi_channel,
        cc_map=cc_map,
    )


def parse_graph(json_path: Path) -> GraphConfig:
    """
    Parse a chain graph from a JSON file.

    Args:
        json_path: Path to the JSON graph file.

    Returns:
        Parsed GraphConfig.

    Raises:
        ValidationError: If the JSON is malformed or missing required fields.

    """
    if not json_path.is_file():
        msg = f"Graph file not found: {json_path}"
        raise ValidationError(msg)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in graph file: {e}"
        raise ValidationError(msg) from e

    if not isinstance(data, dict):
        msg = "Graph JSON must be an object"
        raise ValidationError(msg)

    nodes_data = data.get("nodes")
    if not isinstance(nodes_data, dict):
        msg = "'nodes' must be an object"
        raise ValidationError(msg)

    connections_data = data.get("connections")
    if not isinstance(connections_data, list):
        msg = "'connections' must be an array"
        raise ValidationError(msg)

    nodes = {
        node_id: _parse_node_config(node_id, node_data)
        for node_id, node_data in nodes_data.items()
    }
    connections = [
        _parse_connection_config(conn, i) for i, conn in enumerate(connections_data)
    ]
    return GraphConfig(nodes=nodes, connections=connections)


def _graph_sources_targets(graph: GraphConfig) -> tuple[list[str], list[str]]:
    """Return connection source and target lists."""
    return (
        [c.src_node for c in graph.connections],
        [c.dst_node for c in graph.connections],
    )


def _reserved_name_errors(graph: GraphConfig, reserved: set[str]) -> list[str]:
    """Validate that reserved names are not used as node IDs."""
    return [
        f"'{name}' is a reserved name and cannot be used as a node ID"
        for name in reserved
        if name in graph.nodes
    ]


def _reference_errors(
    graph: GraphConfig,
    sources: list[str],
    targets: list[str],
    reserved: set[str],
) -> list[str]:
    """Validate references and connectivity for node IDs."""
    all_refs = set(sources) | set(targets)
    errors = [
        f"Connection references unknown node '{ref}'"
        for ref in all_refs
        if ref not in reserved and ref not in graph.nodes
    ]
    errors.extend(
        f"Node '{node_id}' is not connected in the graph"
        for node_id in graph.nodes
        if node_id not in all_refs
    )
    return errors


def _midi_cc_errors(graph: GraphConfig) -> list[str]:
    """Validate MIDI channel and CC number ranges."""
    errors: list[str] = []
    for node_id, node in graph.nodes.items():
        if node.midi_channel is not None and not (
            _MIDI_CHANNEL_MIN <= node.midi_channel <= _MIDI_CHANNEL_MAX
        ):
            errors.append(
                f"Node '{node_id}': midi_channel must be 1-16, got {node.midi_channel}"
            )
        errors.extend(
            f"Node '{node_id}': CC number must be 0-127, got {cc_num}"
            for cc_num in node.cc_map
            if not (_CC_MIN <= cc_num <= _CC_MAX)
        )
    return errors


def _linear_structure_errors(graph: GraphConfig) -> list[str]:
    """Validate linear-chain fan-in/fan-out constraints."""
    source_counts = Counter(c.src_node for c in graph.connections)
    target_counts = Counter(c.dst_node for c in graph.connections)
    errors = [
        f"Fan-out detected: '{src}' connects to {count} targets"
        for src, count in source_counts.items()
        if count > 1
    ]
    errors.extend(
        f"Fan-in detected: '{tgt}' receives from {count} sources"
        for tgt, count in target_counts.items()
        if count > 1
    )
    return errors


def _dag_cycle_errors(graph: GraphConfig) -> list[str]:
    """Detect cycles in the DAG using DFS."""
    adj: dict[str, set[str]] = {}
    for c in graph.connections:
        adj.setdefault(c.src_node, set()).add(c.dst_node)

    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(set(graph.nodes) | _RESERVED_NAMES, white)

    def _dfs_cycle(u: str) -> bool:
        color[u] = gray
        for v in adj.get(u, set()):
            if color.get(v, white) == gray:
                return True
            if color.get(v, white) == white and _dfs_cycle(v):
                return True
        color[u] = black
        return False

    return ["Cycle detected in graph"] if _dfs_cycle(_AUDIO_IN) else []


def _dag_connectivity_errors(graph: GraphConfig) -> list[str]:
    """Check that every node is reachable from audio_in and can reach audio_out."""
    adj: dict[str, set[str]] = {}
    rev_adj: dict[str, set[str]] = {}
    for c in graph.connections:
        adj.setdefault(c.src_node, set()).add(c.dst_node)
        rev_adj.setdefault(c.dst_node, set()).add(c.src_node)

    forward_reachable: set[str] = set()
    queue: deque[str] = deque([_AUDIO_IN])
    while queue:
        node = queue.popleft()
        if node in forward_reachable:
            continue
        forward_reachable.add(node)
        queue.extend(adj.get(node, set()))

    backward_reachable: set[str] = set()
    queue = deque([_AUDIO_OUT])
    while queue:
        node = queue.popleft()
        if node in backward_reachable:
            continue
        backward_reachable.add(node)
        queue.extend(rev_adj.get(node, set()))

    errors = [
        f"Node '{node_id}' is not reachable from audio_in"
        for node_id in graph.nodes
        if node_id not in forward_reachable
    ]
    errors.extend(
        f"Node '{node_id}' cannot reach audio_out"
        for node_id in graph.nodes
        if node_id not in backward_reachable
    )
    return errors


def _dag_mixer_input_errors(graph: GraphConfig) -> list[str]:
    """Validate mixer input counts."""
    return [
        f"Node '{nid}': mixer expects {ncfg.mixer_inputs} inputs but has {incoming} "
        "incoming connections"
        for nid, ncfg in graph.nodes.items()
        if ncfg.node_type == "mixer"
        for incoming in [sum(1 for c in graph.connections if c.dst_node == nid)]
        if incoming != ncfg.mixer_inputs
    ]


def validate_linear_chain(graph: GraphConfig) -> list[str]:
    """
    Validate that a graph represents a linear chain.

    Checks for:
    - audio_in and audio_out endpoints present in connections
    - No duplicate node IDs (handled at parse time)
    - No fan-out (node appears as source in multiple connections)
    - No fan-in (node appears as target in multiple connections)
    - No cycles
    - All nodes referenced in connections exist
    - MIDI channel values in valid range (1-16)
    - CC numbers in valid range (0-127)

    Returns:
        List of error messages (empty if valid).

    """
    sources, targets = _graph_sources_targets(graph)
    reserved = _RESERVED_NAMES

    errors = []
    if _AUDIO_IN not in sources:
        errors.append("Connections must include 'audio_in' as a source")
    if _AUDIO_OUT not in targets:
        errors.append("Connections must include 'audio_out' as a target")
    errors.extend(_reserved_name_errors(graph, reserved))
    errors.extend(
        f"Node '{node_id}': mixer nodes are not allowed in linear chains"
        for node_id, node in graph.nodes.items()
        if node.node_type == "mixer"
    )
    errors.extend(_linear_structure_errors(graph))
    errors.extend(_reference_errors(graph, sources, targets, reserved))
    errors.extend(_midi_cc_errors(graph))
    return errors


def extract_chain_order(graph: GraphConfig) -> list[str]:
    """
    Walk connections from audio_in to audio_out, returning ordered node IDs.

    Args:
        graph: A validated linear chain graph.

    Returns:
        Ordered list of node IDs from first to last in the chain.

    Raises:
        ValidationError: If the chain is broken or non-linear.

    """
    # Build adjacency: source -> target
    adjacency: dict[str, str] = {}
    for c in graph.connections:
        if c.src_node in adjacency:
            msg = f"Fan-out at '{c.src_node}': non-linear graph"
            raise ValidationError(msg)
        adjacency[c.src_node] = c.dst_node

    if "audio_in" not in adjacency:
        msg = "No connection from 'audio_in'"
        raise ValidationError(msg)

    order: list[str] = []
    current = adjacency.get("audio_in")

    visited: set[str] = set()
    while current and current != "audio_out":
        if current in visited:
            msg = f"Cycle detected at '{current}'"
            raise ValidationError(msg)
        if current not in graph.nodes:
            msg = f"Connection references unknown node '{current}'"
            raise ValidationError(msg)
        visited.add(current)
        order.append(current)
        current = adjacency.get(current)

    if current != "audio_out":
        msg = "Chain does not reach 'audio_out'"
        raise ValidationError(msg)

    return order


def resolve_chain(
    graph: GraphConfig,
    export_dirs: dict[str, Path],
    version: str,
) -> list[ResolvedChainNode]:
    """
    Resolve a chain graph into fully parsed nodes with manifests.

    Args:
        graph: Validated graph config.
        export_dirs: Mapping of export name -> path to gen~ export directory.
        version: Version string for manifests.

    Returns:
        List of ResolvedChainNode in chain order.

    Raises:
        ValidationError: If exports cannot be found or parsed.

    """
    order = extract_chain_order(graph)

    resolved: list[ResolvedChainNode] = []
    for i, node_id in enumerate(order):
        node_config = graph.nodes[node_id]

        # Assign default MIDI channel if not specified
        if node_config.midi_channel is None:
            node_config.midi_channel = i + 1

        export_name = node_config.export
        if export_name not in export_dirs:
            msg = (
                f"Node '{node_id}': export '{export_name}' not found. "
                f"Available exports: {sorted(export_dirs.keys())}"
            )
            raise ValidationError(
                msg
            )

        export_path = export_dirs[export_name]
        try:
            parser = GenExportParser(export_path)
            export_info = parser.parse()
        except Exception as e:
            msg = f"Node '{node_id}': failed to parse export '{export_name}': {e}"
            raise ValidationError(
                msg
            ) from e

        manifest = manifest_from_export_info(export_info, [], version)

        resolved.append(
            ResolvedChainNode(
                config=node_config,
                index=i,
                export_info=export_info,
                manifest=manifest,
            )
        )

    return resolved


# ---------------------------------------------------------------------------
# Phase 2: DAG validation, topological sort, buffer allocation, resolution
# ---------------------------------------------------------------------------

_RESERVED_NAMES = {"audio_in", "audio_out"}


def validate_dag(graph: GraphConfig) -> list[str]:
    """
    Validate that a graph represents a valid DAG (possibly non-linear).

    Checks:
    1. audio_in/audio_out present in connections
    2. Reserved names not used as node IDs
    3. All referenced nodes exist
    4. All nodes connected
    5. Cycle detection via DFS
    6. Connectivity: all nodes reachable from audio_in and reverse-reachable
       from audio_out
    7. Mixer input count matches incoming connection count
    8. MIDI channel/CC validation

    Returns:
        List of error messages (empty if valid).

    """
    sources, targets = _graph_sources_targets(graph)
    errors = []
    if _AUDIO_IN not in sources:
        errors.append("Connections must include 'audio_in' as a source")
    if _AUDIO_OUT not in targets:
        errors.append("Connections must include 'audio_out' as a target")
    errors.extend(_reserved_name_errors(graph, _RESERVED_NAMES))
    errors.extend(_reference_errors(graph, sources, targets, _RESERVED_NAMES))
    errors.extend(_dag_cycle_errors(graph))
    errors.extend(_dag_connectivity_errors(graph))
    errors.extend(_dag_mixer_input_errors(graph))
    errors.extend(_midi_cc_errors(graph))
    return errors


def topological_sort(graph: GraphConfig) -> list[str]:
    """
    Return a topological ordering of node IDs using Kahn's algorithm.

    Excludes audio_in and audio_out from the result.

    Args:
        graph: A validated DAG graph.

    Returns:
        Ordered list of node IDs in valid execution order.

    Raises:
        ValidationError: If a cycle is detected.

    """
    # Build in-degree map for non-reserved nodes
    in_degree: dict[str, int] = dict.fromkeys(graph.nodes, 0)
    adj: dict[str, list[str]] = {nid: [] for nid in graph.nodes}

    for c in graph.connections:
        src = c.src_node
        dst = c.dst_node
        if dst in _RESERVED_NAMES:
            continue
        if src not in _RESERVED_NAMES:
            adj[src].append(dst)
        # Count all incoming edges (including from audio_in)
        in_degree[dst] = in_degree.get(dst, 0) + 1

    # Seed queue with nodes whose only inputs come from audio_in
    # (i.e. in-degree from non-reserved sources is zero after removing audio_in edges)
    # Actually, we count ALL incoming edges including audio_in, then seed with 0-degree
    queue: deque[str] = deque()
    for nid in graph.nodes:
        # Count edges from audio_in separately
        audio_in_edges = sum(
            1
            for c in graph.connections
            if c.src_node == "audio_in" and c.dst_node == nid
        )
        # In-degree from non-audio_in sources
        real_in = in_degree[nid] - audio_in_edges
        if real_in == 0:
            queue.append(nid)

    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for successor in adj[node]:
            in_degree[successor] -= 1
            # Check if all non-audio_in predecessors are processed
            audio_in_edges = sum(
                1
                for c in graph.connections
                if c.src_node == "audio_in" and c.dst_node == successor
            )
            remaining = in_degree[successor] - audio_in_edges
            if remaining == 0:
                queue.append(successor)

    if len(result) != len(graph.nodes):
        msg = "Cycle detected in graph during topological sort"
        raise ValidationError(msg)

    return result


def allocate_edge_buffers(
    graph: GraphConfig,
    resolved_nodes: dict[str, ResolvedChainNode],
    _topo_order: list[str],
) -> tuple[list[EdgeBuffer], int]:
    """
    Allocate intermediate buffers for DAG edges.

    Fan-out edges from the same source share one buffer_id (zero-copy,
    since gen~ does not mutate input buffers). Edges from audio_in use
    the hardware input buffer (buffer_id = -1, not allocated).

    Args:
        graph: Validated DAG graph config.
        resolved_nodes: Mapping of node_id -> ResolvedChainNode.
        topo_order: Topological ordering of node IDs.

    Returns:
        Tuple of (list of EdgeBuffer, total number of allocated buffers).

    """
    # Assign buffer IDs per source node (fan-out sharing)
    source_buffer: dict[str, int] = {}
    next_buffer_id = 0

    edge_buffers: list[EdgeBuffer] = []

    for c in graph.connections:
        if c.dst_node == "audio_out":
            # audio_out edges: the last node writes directly to the output
            # We still track them but they use the source's buffer
            pass

        if c.src_node == "audio_in":
            # audio_in edges use hardware input buffer (no allocation)
            num_channels = 2  # hardware stereo input
            edge_buffers.append(
                EdgeBuffer(
                    buffer_id=-1,
                    src_node=c.src_node,
                    dst_node=c.dst_node,
                    dst_input_index=c.dst_input_index,
                    num_channels=num_channels,
                )
            )
            continue

        # Allocate or reuse buffer for this source
        if c.src_node not in source_buffer:
            source_buffer[c.src_node] = next_buffer_id
            next_buffer_id += 1

        buf_id = source_buffer[c.src_node]

        # Channel count from source's output count
        src_node = resolved_nodes[c.src_node]
        num_channels = src_node.manifest.num_outputs

        edge_buffers.append(
            EdgeBuffer(
                buffer_id=buf_id,
                src_node=c.src_node,
                dst_node=c.dst_node,
                dst_input_index=c.dst_input_index,
                num_channels=num_channels,
            )
        )

    return edge_buffers, next_buffer_id


def _resolve_dag_gen_node(
    node_id: str,
    node_config: ChainNodeConfig,
    export_dirs: dict[str, Path],
    version: str,
) -> tuple[ExportInfo, Manifest]:
    """Resolve a gen node into export info and manifest."""
    export_name = node_config.export
    if export_name not in export_dirs:
        msg = (
            f"Node '{node_id}': export '{export_name}' not found. "
            f"Available exports: {sorted(export_dirs.keys())}"
        )
        raise ValidationError(msg)

    export_path = export_dirs[export_name]
    try:
        parser = GenExportParser(export_path)
        export_info = parser.parse()
    except Exception as e:
        msg = f"Node '{node_id}': failed to parse export '{export_name}': {e}"
        raise ValidationError(msg) from e

    manifest = manifest_from_export_info(export_info, [], version)
    return export_info, manifest


def _resolve_dag_mixer_node(
    node_id: str,
    node_config: ChainNodeConfig,
    graph: GraphConfig,
) -> Manifest:
    """Build a synthetic manifest for a mixer node."""
    incoming = [c for c in graph.connections if c.dst_node == node_id]
    n_inputs = len(incoming)
    params = [
        ParamInfo(
            index=j,
            name=f"gain_{j}",
            has_minmax=True,
            min=0.0,
            max=2.0,
            default=1.0,
        )
        for j in range(n_inputs)
    ]
    return Manifest(
        gen_name=f"mixer_{node_id}",
        num_inputs=n_inputs,
        num_outputs=max(2, node_config.mixer_inputs),
        params=params,
    )


def resolve_dag(
    graph: GraphConfig,
    export_dirs: dict[str, Path],
    version: str,
) -> list[ResolvedChainNode]:
    """
    Resolve a DAG graph into fully parsed nodes with manifests.

    For gen~ nodes: parses export, creates manifest (same as resolve_chain).
    For mixer nodes: constructs a synthetic Manifest with gain_N parameters.

    Args:
        graph: Validated DAG graph config.
        export_dirs: Mapping of export name -> path to gen~ export directory.
        version: Version string for manifests.

    Returns:
        List of ResolvedChainNode in topological order.

    Raises:
        ValidationError: If exports cannot be found or parsed.

    """
    topo_order = topological_sort(graph)

    resolved: list[ResolvedChainNode] = []
    for i, node_id in enumerate(topo_order):
        node_config = graph.nodes[node_id]

        # Assign default MIDI channel if not specified
        if node_config.midi_channel is None:
            node_config.midi_channel = i + 1

        if node_config.node_type == "mixer":
            manifest = _resolve_dag_mixer_node(node_id, node_config, graph)
            resolved.append(
                ResolvedChainNode(
                    config=node_config,
                    index=i,
                    export_info=None,
                    manifest=manifest,
                )
            )
        else:
            export_info, manifest = _resolve_dag_gen_node(
                node_id,
                node_config,
                export_dirs,
                version,
            )
            resolved.append(
                ResolvedChainNode(
                    config=node_config,
                    index=i,
                    export_info=export_info,
                    manifest=manifest,
                )
            )

    # Second pass: refine mixer output channels based on resolved input channels
    resolved_map = {n.config.id: n for n in resolved}
    for node in resolved:
        if node.config.node_type == "mixer":
            incoming = [c for c in graph.connections if c.dst_node == node.config.id]
            max_ch = 0
            for c in incoming:
                if c.src_node == "audio_in":
                    max_ch = max(max_ch, 2)  # hardware stereo
                elif c.src_node in resolved_map:
                    max_ch = max(max_ch, resolved_map[c.src_node].manifest.num_outputs)
            if max_ch > 0:
                node.manifest.num_outputs = max_ch

    return resolved
