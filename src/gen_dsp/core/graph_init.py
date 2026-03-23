"""
Graph-based project initialization logic.

Extracted from cli.py to decouple orchestration from argparse. Functions
here accept explicit parameters instead of argparse.Namespace objects.
"""

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gen_dsp.core.graph import GraphConfig, ResolvedChainNode
    from gen_dsp.core.project import ProjectConfig


def _emit(message: str) -> None:
    """Write a line to standard output."""
    sys.stdout.write(f"{message}\n")


def _emit_err(message: str) -> None:
    """Write a line to standard error."""
    sys.stderr.write(f"{message}\n")


@dataclass
class ChainInitRequest:
    """Parameters for chain project initialization."""

    output_dir: Path
    name: str
    config: "ProjectConfig"
    apply_patches: bool = True
    dry_run: bool = False
    board: str | None = None


def _legacy_arg(
    args: tuple[object, ...],
    index: int,
    default: object,
) -> object:
    """Return a legacy positional argument if it exists."""
    return args[index] if len(args) > index else default


def _build_request(
    func_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> ChainInitRequest:
    """Build a chain init request from legacy positional/keyword inputs."""
    request = kwargs.pop("request", None)
    if request is None and args and isinstance(args[0], ChainInitRequest):
        request = args[0]

    if request is None:
        output_dir = kwargs.pop("output_dir", _legacy_arg(args, 0, None))
        name = kwargs.pop("name", _legacy_arg(args, 1, None))
        config = kwargs.pop("config", _legacy_arg(args, 2, None))
        apply_patches_value = kwargs.pop(
            "apply_patches",
            _legacy_arg(args, 3, None),
        )
        apply_patches = (
            True if apply_patches_value is None else bool(apply_patches_value)
        )
        dry_run_value = kwargs.pop("dry_run", _legacy_arg(args, 4, None))
        dry_run = False if dry_run_value is None else bool(dry_run_value)
        board = kwargs.pop("board", _legacy_arg(args, 5, None))
        if output_dir is None or name is None or config is None:
            msg = f"{func_name} requires output_dir, name, and config"
            raise TypeError(msg)
        request = ChainInitRequest(
            output_dir=Path(output_dir),
            name=str(name),
            config=config,  # type: ignore[arg-type]
            apply_patches=apply_patches,
            dry_run=dry_run,
            board=board if board is None else str(board),
        )

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        msg = f"Unexpected keyword arguments: {unexpected}"
        raise TypeError(msg)

    return request


def resolve_export_dirs(
    base_dir: Path,
    graph: "GraphConfig",
    extra_exports: list[Path] | None = None,
) -> dict[str, Path]:
    """
    Resolve export directories from base path and explicit overrides.

    For gen~ nodes only (mixer nodes have no export).

    Args:
        base_dir: Base directory to search for exports.
        graph: Parsed graph configuration.
        extra_exports: Optional list of explicit export path overrides.

    Returns:
        Dict mapping export names to their resolved directory paths.

    """
    export_dirs: dict[str, Path] = {}

    for node_config in graph.nodes.values():
        if node_config.export is None:
            continue  # mixer nodes have no export
        candidate = base_dir / node_config.export / "gen"
        if candidate.is_dir():
            export_dirs[node_config.export] = candidate
        else:
            candidate = base_dir / node_config.export
            if candidate.is_dir():
                export_dirs[node_config.export] = candidate

    if extra_exports:
        for export_path in extra_exports:
            resolved = export_path.resolve()
            export_dirs[resolved.name] = resolved

    return export_dirs


def copy_and_patch_exports(
    nodes: "list[ResolvedChainNode]",
    output_dir: Path,
    request: ChainInitRequest,
) -> None:
    """
    Copy gen~ exports and apply patches for gen~ nodes.

    Args:
        nodes: List of resolved chain/DAG nodes.
        output_dir: Target directory for copied exports.
        request: Initialization request controlling patch application.

    """
    for node in nodes:
        if node.config.node_type != "gen" or node.export_info is None:
            continue
        export_dest = output_dir / f"gen_{node.config.export}"
        if export_dest.exists():
            shutil.rmtree(export_dest)
        shutil.copytree(node.export_info.path, export_dest)

    if request.apply_patches:
        patcher_module = importlib.import_module("gen_dsp.core.patcher")
        patcher_cls = patcher_module.Patcher

        for node in nodes:
            if node.config.node_type != "gen" or node.export_info is None:
                continue
            export_dest = output_dir / f"gen_{node.config.export}"
            patcher = patcher_cls(export_dest)
            patcher.apply_all()


def _emit_chain_dry_run(
    request: ChainInitRequest,
    chain: "list[ResolvedChainNode]",
) -> None:
    """Print a dry-run summary for linear chain generation."""
    _emit(f"Would create chain project at: {request.output_dir}")
    _emit("  Platform: circle (chain mode)")
    if request.board:
        _emit(f"  Board: {request.board}")
    _emit(f"  Nodes: {len(chain)}")
    for node in chain:
        export_label = node.config.export or "(built-in)"
        _emit(
            f"    [{node.index}] {node.config.id}: {export_label} "
            f"({node.manifest.num_inputs}in/{node.manifest.num_outputs}out, "
            f"{node.manifest.num_params} params, MIDI ch {node.config.midi_channel})"
        )


def _emit_chain_success(
    request: ChainInitRequest,
    chain: "list[ResolvedChainNode]",
) -> None:
    """Print a success summary for linear chain generation."""
    _emit(f"Chain project created at: {request.output_dir}")
    _emit("  Platform: circle (chain mode)")
    _emit(f"  Nodes: {len(chain)}")
    for node in chain:
        _emit(
            f"    [{node.index}] {node.config.id}: {node.config.export} "
            f"({node.manifest.num_inputs}in/{node.manifest.num_outputs}out)"
        )
    _emit("")
    _emit("Next steps:")
    _emit(f"  cd {request.output_dir}")
    _emit("  make")


def _emit_dag_dry_run(
    request: ChainInitRequest,
    dag_nodes: "list[ResolvedChainNode]",
    num_buffers: int,
) -> None:
    """Print a dry-run summary for DAG generation."""
    _emit(f"Would create DAG project at: {request.output_dir}")
    _emit("  Platform: circle (DAG mode)")
    if request.board:
        _emit(f"  Board: {request.board}")
    _emit(f"  Nodes: {len(dag_nodes)}")
    _emit(f"  Intermediate buffers: {num_buffers}")
    for node in dag_nodes:
        export_label = (
            node.config.export or f"(mixer, {node.config.mixer_inputs} inputs)"
        )
        _emit(
            f"    [{node.index}] {node.config.id}: {export_label} "
            f"({node.manifest.num_inputs}in/{node.manifest.num_outputs}out, "
            f"{node.manifest.num_params} params, MIDI ch {node.config.midi_channel})"
        )


def _emit_dag_success(
    request: ChainInitRequest,
    dag_nodes: "list[ResolvedChainNode]",
    num_buffers: int,
) -> None:
    """Print a success summary for DAG generation."""
    _emit(f"DAG project created at: {request.output_dir}")
    _emit("  Platform: circle (DAG mode)")
    _emit(f"  Nodes: {len(dag_nodes)}")
    _emit(f"  Intermediate buffers: {num_buffers}")
    for node in dag_nodes:
        ntype = "mixer" if node.config.node_type == "mixer" else node.config.export
        _emit(
            f"    [{node.index}] {node.config.id}: {ntype} "
            f"({node.manifest.num_inputs}in/{node.manifest.num_outputs}out)"
        )
    _emit("")
    _emit("Next steps:")
    _emit(f"  cd {request.output_dir}")
    _emit("  make")


def init_chain_linear(
    graph: "GraphConfig",
    export_dirs: dict[str, Path],
    *args: object,
    **kwargs: object,
) -> int:
    """
    Generate a linear chain project (Phase 1 path).

    Args:
        graph: Parsed graph configuration.
        export_dirs: Mapping of export names to directories.
        *args: Backward-compatible positional request fields.
        **kwargs: Backward-compatible keyword request fields.

    Returns:
            Exit code (0 for success, 1 for error).

    """
    graph_module = importlib.import_module("gen_dsp.core.graph")
    errors_module = importlib.import_module("gen_dsp.errors")
    platforms_module = importlib.import_module("gen_dsp.platforms")
    circle_module = importlib.import_module("gen_dsp.platforms.circle")
    resolve_chain = graph_module.resolve_chain
    gen_ext_error = errors_module.GenExtError
    platform_cls = platforms_module.Platform
    circle_platform_cls = circle_module.CirclePlatform

    request = _build_request("init_chain_linear", args, kwargs)

    try:
        chain = resolve_chain(graph, export_dirs, platform_cls.GENEXT_VERSION)
    except gen_ext_error as e:
        _emit_err(f"Error resolving chain: {e}")
        return 1

    if request.dry_run:
        _emit_chain_dry_run(request, chain)
        return 0

    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)

        platform = circle_platform_cls()
        platform.generate_chain_project(
            chain, graph, request.output_dir, request.name, request.config
        )

        # Copy gen~ exports and apply patches
        copy_and_patch_exports(chain, request.output_dir, request)

        _emit_chain_success(request, chain)
    except gen_ext_error as e:
        _emit_err(f"Error creating chain project: {e}")
        return 1

    return 0


def init_chain_dag(
    graph: "GraphConfig",
    export_dirs: dict[str, Path],
    *args: object,
    **kwargs: object,
) -> int:
    """
    Generate a DAG project (Phase 2 path).

    Args:
        graph: Parsed graph configuration.
        export_dirs: Mapping of export names to directories.
        *args: Backward-compatible positional request fields.
        **kwargs: Backward-compatible keyword request fields.

    Returns:
            Exit code (0 for success, 1 for error).

    """
    graph_module = importlib.import_module("gen_dsp.core.graph")
    errors_module = importlib.import_module("gen_dsp.errors")
    platforms_module = importlib.import_module("gen_dsp.platforms")
    circle_module = importlib.import_module("gen_dsp.platforms.circle")
    allocate_edge_buffers = graph_module.allocate_edge_buffers
    resolve_dag = graph_module.resolve_dag
    gen_ext_error = errors_module.GenExtError
    platform_cls = platforms_module.Platform
    circle_platform_cls = circle_module.CirclePlatform

    request = _build_request("init_chain_dag", args, kwargs)

    try:
        dag_nodes = resolve_dag(graph, export_dirs, platform_cls.GENEXT_VERSION)
    except gen_ext_error as e:
        _emit_err(f"Error resolving DAG: {e}")
        return 1

    resolved_map = {n.config.id: n for n in dag_nodes}
    topo_order = [n.config.id for n in dag_nodes]
    edge_buffers, num_buffers = allocate_edge_buffers(graph, resolved_map, topo_order)

    if request.dry_run:
        _emit_dag_dry_run(request, dag_nodes, num_buffers)
        return 0

    try:
        request.output_dir.mkdir(parents=True, exist_ok=True)

        platform = circle_platform_cls()
        dag_context = circle_module.DagProjectContext(
            dag_nodes=dag_nodes,
            graph=graph,
            edge_buffers=edge_buffers,
            num_buffers=num_buffers,
        )
        platform.generate_dag_project(
            dag_context,
            request.output_dir,
            request.name,
            request.config,
        )

        # Copy gen~ exports and apply patches (gen~ nodes only)
        copy_and_patch_exports(dag_nodes, request.output_dir, request)

        _emit_dag_success(request, dag_nodes, num_buffers)
    except gen_ext_error as e:
        _emit_err(f"Error creating DAG project: {e}")
        return 1

    return 0
