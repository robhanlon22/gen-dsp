"""
Command-line interface for gen_dsp.

Usage:
    gen-dsp <source> -p <platform> [--no-build] [--dry-run]
    gen-dsp compile <file>
    gen-dsp validate <file>
    gen-dsp dot <file>
    gen-dsp sim <file> [options]
    gen-dsp build [project-path] [-p <platform>]
    gen-dsp detect <export-path> [--json]
    gen-dsp patch <target-path> [--dry-run]
    gen-dsp chain <export-dir> --graph <chain.json> -n NAME [-p circle]
    gen-dsp list
    gen-dsp cache
    gen-dsp manifest <export-path>
"""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import TextIO

from gen_dsp import __version__
from gen_dsp.core.builder import Builder
from gen_dsp.core.parser import GenExportParser
from gen_dsp.core.patcher import Patcher
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.errors import GenExtError
from gen_dsp.platforms import get_platform, list_platforms
from gen_dsp.platforms.base import Platform

# Known subcommands for two-phase dispatch.
SUBCOMMANDS = {
    "compile",
    "validate",
    "dot",
    "sim",
    "build",
    "detect",
    "patch",
    "list",
    "cache",
    "manifest",
    "chain",
}


def _write(stream: TextIO, message: str = "") -> None:
    """Write a single line to a text stream."""
    if message.endswith("\n"):
        stream.write(message)
    else:
        stream.write(f"{message}\n")


def _stdout(message: str = "") -> None:
    """Write a single line to stdout."""
    _write(sys.stdout, message)


def _stderr(message: str = "") -> None:
    """Write a single line to stderr."""
    _write(sys.stderr, message)


def _report_block(header: str, items: list[object]) -> None:
    """Report a block of errors or configuration messages."""
    _stderr(header)
    for item in items:
        _stderr(f"  - {item}")


def _print_help() -> None:
    """Print top-level help text."""
    platforms = ", ".join(list_platforms())
    _stdout(
        f"""\
usage: gen-dsp <source> -p <platform> [options]
       gen-dsp <command> [args]

gen-dsp {__version__} -- generate buildable audio DSP plugins

Default command (auto-detects source type):
  gen-dsp <dir>           gen~ export directory
  gen-dsp <file.gdsp>     graph DSL file
  gen-dsp <file.json>     graph JSON file

  -p, --platform PLATFORM   Target platform (required): {platforms}
  -n, --name NAME           Plugin name (default: inferred from source)
  -o, --output DIR          Output directory (default: <name>_<platform>)
  --no-build                Skip building after project creation
  --dry-run                 Show what would be done without creating files
  --buffers NAME [NAME ...]
  --no-patch                Skip platform patches
  --no-shared-cache         Disable shared OS cache for FetchContent downloads
  --cache-dir DIR           Explicit FetchContent cache directory
  --board BOARD             Board variant (daisy, circle)
  --no-midi                 Disable MIDI note handling
  --midi-gate NAME          MIDI gate parameter name
  --midi-freq NAME          MIDI frequency parameter name
  --midi-vel NAME           MIDI velocity parameter name
  --midi-freq-unit {{hz,midi}}
  --voices N                Polyphony voices (default: 1)
  --inputs-as-params [NAME ...]
                            Remap signal inputs to params (all or named)

Subcommands:
  compile <file>            Compile graph to C++ (stdout or -o dir)
  validate <file>           Validate a graph file
  dot <file>                Generate DOT visualization
  sim <file>                Simulate graph (WAV in/out)
  build [dir]               Build an existing project
  detect <dir>              Analyze a gen~ export
  patch <dir>               Apply platform-specific patches
  chain <dir>               Multi-plugin chain mode (Circle)
  list                      List available platforms
  cache                     Show cached SDKs
  manifest <dir>            Emit JSON manifest for a gen~ export

Options:
  -V, --version             Show version
  -h, --help                Show this help
"""
    )


def _make_default_parser() -> argparse.ArgumentParser:
    """Parser for the default command: <source> -p <platform> [flags]."""
    parser = argparse.ArgumentParser(
        prog="gen-dsp",
        add_help=False,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to gen~ export directory, .gdsp file, or graph JSON file",
    )
    parser.add_argument(
        "-p",
        "--platform",
        choices=list_platforms(),
        required=True,
        help="Target platform",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Name for the plugin (default: inferred from source)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: ./<name>_<platform>)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building after project creation",
    )
    parser.add_argument(
        "--buffers",
        nargs="+",
        help="Buffer names (overrides auto-detection)",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Don't apply platform patches (exp2f fix)",
    )
    parser.add_argument(
        "--no-shared-cache",
        action="store_true",
        help=(
            "Disable shared OS cache for FetchContent downloads "
            "(CMake-based platforms)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Explicit FetchContent cache directory (baked into CMakeLists.txt)",
    )
    parser.add_argument(
        "--board",
        help="Board variant for embedded platforms (daisy, circle)",
    )
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Disable MIDI note handling even if gate/freq params are detected",
    )
    parser.add_argument(
        "--midi-gate",
        metavar="NAME",
        help="Parameter name to use as MIDI gate (implies MIDI enabled)",
    )
    parser.add_argument(
        "--midi-freq",
        metavar="NAME",
        help="Parameter name to use as MIDI frequency (implies MIDI enabled)",
    )
    parser.add_argument(
        "--midi-vel",
        metavar="NAME",
        help="Parameter name to use as MIDI velocity (implies MIDI enabled)",
    )
    parser.add_argument(
        "--midi-freq-unit",
        choices=["hz", "midi"],
        default="hz",
        help="Frequency unit: hz (mtof conversion, default) or midi (raw note number)",
    )
    parser.add_argument(
        "--voices",
        type=int,
        default=1,
        metavar="N",
        help="Number of polyphony voices (default: 1 = monophonic, requires MIDI)",
    )
    parser.add_argument(
        "--inputs-as-params",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Remap signal inputs to parameters. "
        "No names = remap all; with names = remap only those inputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating files",
    )
    return parser


def _make_subcommand_parser() -> argparse.ArgumentParser:
    """Parser with all subcommands registered."""
    parser = argparse.ArgumentParser(
        prog="gen-dsp",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    # build command
    build_parser = subparsers.add_parser("build", help="Build an existing project")
    build_parser.add_argument(
        "project_path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to the project directory (default: current directory)",
    )
    build_parser.add_argument(
        "-p",
        "--platform",
        choices=list_platforms(),
        default="pd",
        help="Target platform (default: pd)",
    )
    build_parser.add_argument(
        "--clean", action="store_true", help="Clean before building"
    )
    build_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show build output"
    )

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Analyze a gen~ export")
    detect_parser.add_argument(
        "export_path", type=Path, help="Path to gen~ export directory"
    )
    detect_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # patch command
    patch_parser = subparsers.add_parser(
        "patch", help="Apply platform-specific patches"
    )
    patch_parser.add_argument(
        "target_path", type=Path, help="Path to project or gen~ export dir"
    )
    patch_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    # list command
    subparsers.add_parser("list", help="List available target platforms")

    # cache command
    subparsers.add_parser("cache", help="Show cached SDKs and dependencies")

    # manifest command
    manifest_parser = subparsers.add_parser("manifest", help="Emit JSON manifest")
    manifest_parser.add_argument(
        "export_path", type=Path, help="Path to gen~ export directory"
    )
    manifest_parser.add_argument(
        "--buffers", nargs="+", help="Buffer names (overrides auto-detection)"
    )

    # chain command
    chain_parser = subparsers.add_parser(
        "chain", help="Multi-plugin chain mode (Circle)"
    )
    chain_parser.add_argument(
        "export_path",
        type=Path,
        help="Path to gen~ export directory (base for chain nodes)",
    )
    chain_parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="JSON graph file for multi-plugin chain",
    )
    chain_parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="Name for the chain project",
    )
    chain_parser.add_argument(
        "-p",
        "--platform",
        default="circle",
        help="Target platform (default: circle)",
    )
    chain_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: ./<name>)",
    )
    chain_parser.add_argument(
        "--export",
        type=Path,
        action="append",
        dest="exports",
        help="Additional export path (can be repeated)",
    )
    chain_parser.add_argument(
        "--no-patch", action="store_true", help="Skip platform patches"
    )
    chain_parser.add_argument("--board", help="Board variant")
    chain_parser.add_argument("--no-build", action="store_true", help="Skip building")
    chain_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )

    # graph subcommands (compile, validate, dot, sim)
    try:
        graph_cli = import_module("gen_dsp.graph.cli")
    except ImportError:
        pass
    else:
        graph_cli.add_compile_parser(subparsers)
        graph_cli.add_validate_parser(subparsers)
        graph_cli.add_dot_parser(subparsers)
        graph_cli.add_sim_parser(subparsers)

    return parser


def _require_graph_support() -> bool:
    """Check whether graph-specific support is available."""
    try:
        graph_module = import_module("gen_dsp.graph")
        require_dsp_graph = graph_module.__dict__["_require_dsp_graph"]
        require_dsp_graph()
    except ImportError as e:
        _stderr(f"Error: {e}")
        return False
    return True


def _load_graph_source(graph_path: Path) -> object:
    """Load a graph from JSON or .gdsp."""
    graph_module = import_module("gen_dsp.graph.models")
    graph_cls = graph_module.Graph
    graph_cli = import_module("gen_dsp.graph.cli")
    graph_errors = graph_cli.__dict__["_gdsp_errors"]()
    validation_error = import_module("pydantic").ValidationError

    try:
        if graph_path.suffix == ".gdsp":
            parse_file = import_module("gen_dsp.graph.dsl").parse_file
            parsed = parse_file(graph_path)
            if isinstance(parsed, graph_cls):
                return parsed
            message = "parsed file did not produce a Graph"
            raise ValueError(message)

        text = graph_path.read_text()
        data = json.loads(text)
        return graph_cls.model_validate(data)
    except graph_errors as e:
        message = f"invalid graph: {e}"
        raise ValueError(message) from e
    except validation_error as e:
        message = f"invalid graph: {e}"
        raise ValueError(message) from e


def _report_validation_block(header: str, errors: list[object]) -> bool:
    """Print a block of validation errors, returning True when any exist."""
    if not errors:
        return False
    _report_block(header, errors)
    return True


def _report_build_result(
    result: object,
    *,
    show_stderr: bool,
    show_stdout_tail: bool,
) -> bool:
    """Report a builder result and return whether it succeeded."""
    if result.success:
        _stdout("Build successful!")
        if result.output_file:
            _stdout(f"Output: {result.output_file}")
        return True

    _stderr("Build failed!")
    if show_stderr and result.stderr:
        _stderr(result.stderr.rstrip("\n"))
    elif show_stdout_tail and result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            _stderr(line)
    return False


def _print_next_steps(project_dir: Path, platform: str) -> None:
    """Print follow-up shell commands after generation."""
    _stdout()
    _stdout("Next steps:")
    _stdout(f"  cd {project_dir}")
    platform_impl = get_platform(platform)
    for instruction in platform_impl.get_build_instructions():
        _stdout(f"  {instruction}")


def _report_graph_dry_run(
    args: argparse.Namespace,
    graph: object,
    output_dir: Path,
) -> None:
    """Print the dry-run summary for graph sources."""
    _stdout(f"Would create project at: {output_dir}")
    _stdout(f"  Source: dsp-graph ({args.source.name})")
    _stdout(f"  Graph: {graph.name}")
    _stdout(f"  Platform: {args.platform}")
    _stdout(f"  Inputs: {len(graph.inputs)}")
    _stdout(f"  Outputs: {len(graph.outputs)}")
    _stdout(f"  Parameters: {len(graph.params)}")
    if not args.no_build:
        _stdout("  Would build after creating")


def _report_export_dry_run(
    args: argparse.Namespace,
    export_info: object,
    buffers: list[str] | None,
    output_dir: Path,
) -> None:
    """Print the dry-run summary for export sources."""
    _stdout(f"Would create project at: {output_dir}")
    _stdout(f"  Export: {export_info.name}")
    _stdout(f"  Platform: {args.platform}")
    if args.board:
        _stdout(f"  Board: {args.board}")
    _stdout(f"  Inputs: {export_info.num_inputs}")
    _stdout(f"  Outputs: {export_info.num_outputs}")
    _stdout(f"  Parameters: {export_info.num_params}")
    _stdout(f"  Buffers: {buffers or '(none)'}")
    if export_info.has_exp2f_issue and not args.no_patch:
        _stdout("  Would apply exp2f -> exp2 patch")
    if not args.no_build:
        _stdout("  Would build after creating")


def _validate_export_run_args(args: argparse.Namespace) -> str | None:
    """Validate export-run flags that are independent of project parsing."""
    if args.board and args.platform not in ("daisy", "circle"):
        return "Error: --board is only valid for daisy and circle"
    if args.voices < 1:
        return "Error: --voices must be >= 1"
    if args.voices > 1 and args.no_midi:
        return "Error: --voices > 1 requires MIDI (incompatible with --no-midi)"
    return None


def _prepare_export_inputs(
    args: argparse.Namespace,
    export_path: Path,
) -> tuple[object, list[str]] | None:
    """Parse export metadata and validate argument combinations."""
    if args.name is None:
        args.name = export_path.name
        if not args.name:
            _stderr("Error: could not infer name from export path")
            return None

    try:
        parser = GenExportParser(export_path)
        export_info = parser.parse()
    except GenExtError as e:
        _stderr(f"Error parsing export: {e}")
        return None

    buffers = args.buffers or export_info.buffers
    invalid = parser.validate_buffer_names(buffers)
    if invalid:
        _stderr(f"Error: Invalid buffer names: {invalid}")
        _stderr("Buffer names must be valid C identifiers.")
        return None

    error = _validate_export_run_args(args)
    if error is not None:
        _stderr(error)
        return None

    return export_info, buffers


def _report_graph_project_created(
    args: argparse.Namespace,
    graph: object,
    project_dir: Path,
) -> None:
    """Report graph project creation."""
    _stdout(f"Project created at: {project_dir}")
    _stdout("  Source: dsp-graph")
    _stdout(f"  Platform: {args.platform}")
    if graph.params:
        _stdout(f"  Parameters: {', '.join(p.name for p in graph.params)}")


def _report_export_project_created(
    args: argparse.Namespace,
    buffers: list[str] | None,
    project_dir: Path,
) -> None:
    """Report export project creation."""
    _stdout(f"Project created at: {project_dir}")
    _stdout(f"  External name: {args.name}~")
    _stdout(f"  Platform: {args.platform}")
    if buffers:
        _stdout(f"  Buffers: {', '.join(buffers)}")


def _prepare_default_graph_run(
    args: argparse.Namespace,
    graph_path: Path,
) -> tuple[object, ProjectConfig, Path] | None:
    """Prepare the graph-source default-command run."""
    if not _require_graph_support():
        return None

    if args.name is None:
        args.name = graph_path.stem
        if not args.name:
            _stderr("Error: could not infer name from graph file")
            return None

    try:
        graph = _load_graph_source(graph_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as e:
        _stderr(f"Error loading graph: {e}")
        return None

    validate_graph = import_module("gen_dsp.graph.validate").validate_graph
    if _report_validation_block(
        "Graph validation errors:",
        validate_graph(graph),
    ):
        return None

    config = ProjectConfig(
        name=args.name,
        platform=args.platform,
        buffers=[],
        apply_patches=False,
        output_dir=args.output,
        shared_cache=not getattr(args, "no_shared_cache", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    if _report_validation_block("Configuration errors:", config.validate()):
        return None

    output_dir = (
        args.output or Path.cwd() / "build" / f"{args.name}_{args.platform}"
    )
    return graph, config, output_dir


def _prepare_default_export_run(
    args: argparse.Namespace,
    export_path: Path,
) -> tuple[object, ProjectConfig, list[str], Path] | None:
    """Prepare the export-source default-command run."""
    prepared = _prepare_export_inputs(args, export_path)
    if prepared is None:
        return None

    export_info, buffers = prepared
    config = ProjectConfig(
        name=args.name,
        platform=args.platform,
        buffers=buffers,
        apply_patches=not args.no_patch,
        output_dir=args.output,
        shared_cache=not args.no_shared_cache,
        cache_dir=args.cache_dir,
        board=args.board,
        no_midi=args.no_midi,
        midi_gate=args.midi_gate,
        midi_freq=args.midi_freq,
        midi_vel=args.midi_vel,
        midi_freq_unit=args.midi_freq_unit,
        num_voices=args.voices,
        inputs_as_params=args.inputs_as_params,
    )
    if _report_validation_block("Configuration errors:", config.validate()):
        return None

    output_dir = (
        args.output or Path.cwd() / "build" / f"{args.name}_{args.platform}"
    )
    return export_info, config, buffers, output_dir


def _generate_project(
    generator: object,
    output_dir: Path,
    error_label: str,
) -> Path | None:
    """Generate a project directory from a prepared generator."""
    try:
        return generator.generate(output_dir)
    except (GenExtError, OSError, ValueError) as e:
        _stderr(f"{error_label}: {e}")
        return None


def _build_or_skip(project_dir: Path, platform: str, *, no_build: bool) -> int:
    """Build a project or print the next steps."""
    if no_build:
        _print_next_steps(project_dir, platform)
        return 0

    try:
        builder = Builder(project_dir)
        result = builder.build(target_platform=platform)
    except GenExtError as e:
        _stderr(f"Build error: {e}")
        return 1
    if not _report_build_result(
        result,
        show_stderr=True,
        show_stdout_tail=False,
    ):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_default(argv: list[str]) -> int:
    """Handle the default command: <source> -p <platform> [flags]."""
    parser = _make_default_parser()
    args = parser.parse_args(argv)

    source = args.source.resolve()

    # Auto-detect source type
    if source.is_file() and source.suffix in (".gdsp", ".json"):
        return _cmd_default_graph(args, source)
    if source.is_dir():
        return _cmd_default_export(args, source)
    _stderr(f"Error: source not found or unrecognized type: {source}")
    _stderr("Expected: directory (gen~ export), .gdsp file, or .json file")
    return 1


def _cmd_default_graph(args: argparse.Namespace, graph_path: Path) -> int:
    """Handle default command with a graph file source."""
    prepared = _prepare_default_graph_run(args, graph_path)
    if prepared is None:
        return 1

    graph, config, output_dir = prepared
    if args.dry_run:
        _report_graph_dry_run(args, graph, output_dir)
        return 0

    try:
        generator = ProjectGenerator.from_graph(graph, config)
        project_dir = _generate_project(
            generator,
            output_dir,
            "Error creating project",
        )
    except (GenExtError, OSError, ValueError) as e:
        _stderr(f"Error creating project: {e}")
        return 1
    if project_dir is None:
        return 1

    _report_graph_project_created(args, graph, project_dir)
    return _build_or_skip(project_dir, args.platform, no_build=args.no_build)


def _cmd_default_export(args: argparse.Namespace, export_path: Path) -> int:
    """Handle default command with a gen~ export directory source."""
    prepared = _prepare_default_export_run(args, export_path)
    if prepared is None:
        return 1

    export_info, config, buffers, output_dir = prepared
    if args.dry_run:
        _report_export_dry_run(args, export_info, buffers, output_dir)
        return 0

    try:
        generator = ProjectGenerator(export_info, config)
        project_dir = _generate_project(
            generator,
            output_dir,
            "Error creating project",
        )
    except (GenExtError, OSError, ValueError) as e:
        _stderr(f"Error creating project: {e}")
        return 1
    if project_dir is None:
        return 1

    _report_export_project_created(args, buffers, project_dir)
    return _build_or_skip(project_dir, args.platform, no_build=args.no_build)


def cmd_build(args: argparse.Namespace) -> int:
    """Handle the build command."""
    project_path = args.project_path.resolve()

    if not project_path.is_dir():
        _stderr(f"Error: Project directory not found: {project_path}")
        return 1

    try:
        builder = Builder(project_path)
        result = builder.build(
            target_platform=args.platform,
            clean=args.clean,
            verbose=args.verbose,
        )
    except GenExtError as e:
        _stderr(f"Error: {e}")
        return 1
    else:
        return (
            0
            if _report_build_result(
                result,
                show_stderr=not args.verbose,
                show_stdout_tail=not args.verbose,
            )
            else 1
        )


def cmd_detect(args: argparse.Namespace) -> int:
    """Handle the detect command."""
    export_path = args.export_path.resolve()

    try:
        parser = GenExportParser(export_path)
        info = parser.parse()
    except GenExtError as e:
        _stderr(f"Error: {e}")
        return 1
    else:
        if args.json:
            data = {
                "name": info.name,
                "path": str(info.path),
                "num_inputs": info.num_inputs,
                "num_outputs": info.num_outputs,
                "num_params": info.num_params,
                "buffers": info.buffers,
                "has_exp2f_issue": info.has_exp2f_issue,
                "cpp_file": str(info.cpp_path) if info.cpp_path else None,
                "h_file": str(info.h_path) if info.h_path else None,
            }
            _stdout(json.dumps(data, indent=2))
        else:
            _stdout(f"Gen~ Export: {info.name}")
            _stdout(f"  Path: {info.path}")
            _stdout(f"  Signal inputs: {info.num_inputs}")
            _stdout(f"  Signal outputs: {info.num_outputs}")
            _stdout(f"  Parameters: {info.num_params}")
            _stdout(f"  Buffers: {info.buffers or '(none detected)'}")
            if info.has_exp2f_issue:
                _stdout("  Patch needed: exp2f -> exp2 (macOS compatibility)")

    return 0


def cmd_patch(args: argparse.Namespace) -> int:
    """Handle the patch command."""
    target_path = args.target_path.resolve()

    if not target_path.is_dir():
        _stderr(f"Error: Directory not found: {target_path}")
        return 1

    patcher = Patcher(target_path)

    if args.dry_run:
        needed = patcher.check_patches_needed()
        if not any(needed.values()):
            _stdout("No patches needed.")
            return 0

        _stdout("Patches that would be applied:")
        for name, is_needed in needed.items():
            if is_needed:
                _stdout(f"  - {name}")
        return 0

    results = patcher.apply_all()

    if not results:
        _stdout("No patches needed or applicable.")
        return 0

    for result in results:
        if result.applied:
            _stdout(f"Applied: {result.patch_name}")
            _stdout(f"  File: {result.file_path}")
            _stdout(f"  {result.message}")
        else:
            _stdout(f"Skipped: {result.patch_name}")
            _stdout(f"  {result.message}")

    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    """Handle the list command."""
    for name in list_platforms():
        _stdout(name)
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    """Handle the manifest command."""
    manifest_from_export_info = import_module(
        "gen_dsp.core.manifest"
    ).manifest_from_export_info

    export_path = args.export_path.resolve()

    try:
        parser = GenExportParser(export_path)
        export_info = parser.parse()
    except GenExtError as e:
        _stderr(f"Error parsing export: {e}")
        return 1
    else:
        buffers = args.buffers or export_info.buffers
        manifest = manifest_from_export_info(
            export_info,
            buffers,
            Platform.GENEXT_VERSION,
        )
        _stdout(manifest.to_json())
        return 0


def cmd_cache(_args: argparse.Namespace) -> int:
    """Handle the cache command."""
    get_cache_dir = import_module("gen_dsp.core.cache").get_cache_dir
    daisy_module = import_module("gen_dsp.platforms.daisy")
    libdaisy_version = daisy_module.LIBDAISY_VERSION
    resolve_libdaisy_dir = daisy_module.__dict__["_resolve_libdaisy_dir"]
    resolve_rack_dir = import_module("gen_dsp.platforms.vcvrack").__dict__[
        "_resolve_rack_dir"
    ]

    env_cache = os.environ.get("GEN_DSP_CACHE_DIR")
    if env_cache:
        cache_dir = Path(env_cache)
        _stdout(f"Cache directory: {cache_dir}  (GEN_DSP_CACHE_DIR)")
    else:
        cache_dir = get_cache_dir()
        _stdout(f"Cache directory: {cache_dir}")
    _stdout()

    _stdout("FetchContent (clap, lv2, sc, vst3):")
    if cache_dir.is_dir():
        src_dirs = sorted(
            d.name
            for d in cache_dir.iterdir()
            if d.is_dir()
            and d.name.endswith("-src")
            and d.name not in ("rack-sdk-src", "libdaisy-src")
        )
        if src_dirs:
            for name in src_dirs:
                sdk_name = name.removesuffix("-src")
                _stdout(f"  {sdk_name}  ({cache_dir / name})")
        else:
            _stdout("  (empty)")
    else:
        _stdout("  (not created)")
    _stdout()

    rack_dir = resolve_rack_dir()
    rack_present = (rack_dir / "Makefile").is_file()
    _stdout("Rack SDK (vcvrack):")
    _stdout(f"  Path: {rack_dir}")
    _stdout(f"  Status: {'present' if rack_present else 'not downloaded'}")
    _stdout()

    libdaisy_dir = resolve_libdaisy_dir()
    libdaisy_present = (libdaisy_dir / "core" / "Makefile").is_file()
    libdaisy_built = (libdaisy_dir / "build" / "libdaisy.a").is_file()
    _stdout(f"libDaisy {libdaisy_version} (daisy):")
    _stdout(f"  Path: {libdaisy_dir}")
    if libdaisy_built:
        _stdout("  Status: built")
    elif libdaisy_present:
        _stdout("  Status: cloned (not built)")
    else:
        _stdout("  Status: not cloned")

    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    """Handle the chain command (multi-plugin chain mode, Circle only)."""
    graph_module = import_module("gen_dsp.core.graph")
    parse_graph = graph_module.parse_graph
    validate_dag = graph_module.validate_dag
    validate_linear_chain = graph_module.validate_linear_chain
    graph_init_module = import_module("gen_dsp.core.graph_init")
    init_chain_dag = graph_init_module.init_chain_dag
    init_chain_linear = graph_init_module.init_chain_linear
    resolve_export_dirs = graph_init_module.resolve_export_dirs

    if args.platform != "circle":
        _stderr(
            "Error: chain command is currently only supported for the "
            "circle platform"
        )
        return 1

    graph_path = args.graph.resolve()

    try:
        graph = parse_graph(graph_path)
    except GenExtError as e:
        _stderr(f"Error parsing graph: {e}")
        return 1
    else:
        linear_errors = validate_linear_chain(graph)
        is_linear = len(linear_errors) == 0

        if not is_linear:
            dag_errors = validate_dag(graph)
            if dag_errors:
                _report_block("Graph validation errors:", dag_errors)
                return 1

        export_dirs = resolve_export_dirs(
            args.export_path.resolve(),
            graph,
            args.exports,
        )

        output_dir = args.output or Path.cwd() / "build" / args.name
        output_dir = Path(output_dir).resolve()

        config = ProjectConfig(
            name=args.name,
            platform="circle",
            buffers=[],
            apply_patches=not args.no_patch,
            output_dir=args.output,
            board=args.board,
        )

        if is_linear:
            return init_chain_linear(
                graph,
                export_dirs,
                output_dir,
                args.name,
                config,
                apply_patches=not args.no_patch,
                dry_run=args.dry_run,
                board=args.board,
            )
        return init_chain_dag(
            graph,
            export_dirs,
            output_dir,
            args.name,
            config,
            apply_patches=not args.no_patch,
            dry_run=args.dry_run,
            board=args.board,
        )


def _dispatch_subcommand(argv: list[str]) -> int:
    """Parse and dispatch a subcommand."""
    parser = _make_subcommand_parser()
    args = parser.parse_args(argv)

    handlers = {
        "build": cmd_build,
        "detect": cmd_detect,
        "patch": cmd_patch,
        "list": cmd_list,
        "cache": cmd_cache,
        "manifest": cmd_manifest,
        "chain": cmd_chain,
    }

    # Add graph subcommand handlers if available
    try:
        graph_cli = import_module("gen_dsp.graph.cli")
    except ImportError:
        pass
    else:
        handlers["compile"] = graph_cli.cmd_compile
        handlers["validate"] = graph_cli.cmd_validate
        handlers["dot"] = graph_cli.cmd_dot
        handlers["sim"] = graph_cli.cmd_simulate

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    _print_help()
    return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    argv = argv if argv is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0

    if argv[0] in ("-V", "--version"):
        _stdout(f"gen-dsp {__version__}")
        return 0

    if argv[0] in SUBCOMMANDS:
        return _dispatch_subcommand(argv)
    return _cmd_default(argv)


if __name__ == "__main__":
    sys.exit(main())
