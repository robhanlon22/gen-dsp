"""
Command-line interface for dsp-graph (internal module).

This module provides the graph subcommand implementations for gen-dsp's CLI.
It can also be used standalone via ``main()`` for testing.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from importlib import import_module
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from gen_dsp.graph.compile import compile_graph, compile_graph_to_file
from gen_dsp.graph.models import Graph
from gen_dsp.graph.optimize import optimize_graph
from gen_dsp.graph.validate import validate_graph
from gen_dsp.graph.visualize import graph_to_dot, graph_to_dot_file

_WAV_HEADER_MIN_SIZE = 44
_PCM_FORMAT_TAG = 1
_IEEE_FLOAT_FORMAT_TAG = 3
_PCM16_BITS = 16
_PCM32_BITS = 32
_PCM16_SCALE = 32768.0
_PCM32_SCALE = 2147483648.0


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


def _load_graph(path: str) -> Graph:
    """
    Load and parse a graph file (JSON or .gdsp).

    Auto-detects format by file extension: ``.gdsp`` files are parsed via
    the DSL parser; everything else is treated as JSON.
    """
    p = Path(path)
    if p.suffix == ".gdsp":
        parse_file = import_module("gen_dsp.graph.dsl").parse_file
        result = parse_file(p)
        if isinstance(result, Graph):
            return result
        message = "Parsed .gdsp file did not produce a Graph"
        raise ValueError(message)
    text = p.read_text()
    data = json.loads(text)
    return Graph.model_validate(data)


# ---------------------------------------------------------------------------
# WAV I/O helpers (float32 via RIFF, no external deps beyond numpy)
# ---------------------------------------------------------------------------


def _read_wav(path: str) -> tuple[list[list[float]], int]:
    """
    Read a WAV file and return (channels, sample_rate).

    Each channel is a list of float samples. Supports PCM16, PCM32 (tag 1)
    and float32 (tag 3).
    """
    np = import_module("numpy")
    with Path(path).open("rb") as f:
        data = f.read()
    if (
        len(data) < _WAV_HEADER_MIN_SIZE
        or data[:4] != b"RIFF"
        or data[8:12] != b"WAVE"
    ):
        message = f"Not a valid WAV file: {path}"
        raise ValueError(message)

    fmt_tag, n_channels, sample_rate, bits_per_sample, audio_data = _parse_wav_chunks(
        data,
        path,
    )
    samples = _decode_wav_samples(
        np,
        fmt_tag=fmt_tag,
        bits_per_sample=bits_per_sample,
        audio_data=audio_data,
    )
    channels = _deinterleave_samples(samples, n_channels)

    if n_channels > 1:
        _stderr(

                f"info: {path}: {n_channels} channels, using first channel "
                f"({len(channels[0])} samples)"

        )

    return channels, sample_rate


def _write_wav(path: str, data: list[float], sample_rate: int) -> None:
    """Write a mono float32 WAV file."""
    np = import_module("numpy")

    samples = np.array(data, dtype=np.float32)
    raw = samples.tobytes()
    n_channels = 1
    bits_per_sample = _PCM32_BITS
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8

    with Path(path).open("wb") as f:
        # RIFF header
        data_size = len(raw)
        file_size = 36 + data_size
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", _IEEE_FLOAT_FORMAT_TAG))
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(raw)


def _parse_wav_chunks(data: bytes, path: str) -> tuple[int, int, int, int, bytes]:
    """Extract WAV format and payload chunks."""
    pos = 12
    fmt_tag = 0
    n_channels = 0
    sample_rate = 0
    bits_per_sample = 0
    audio_data = b""

    while pos < len(data) - 8:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        chunk_data = data[pos + 8 : pos + 8 + chunk_size]

        if chunk_id == b"fmt ":
            fmt_tag = struct.unpack_from("<H", chunk_data, 0)[0]
            n_channels = struct.unpack_from("<H", chunk_data, 2)[0]
            sample_rate = struct.unpack_from("<I", chunk_data, 4)[0]
            bits_per_sample = struct.unpack_from("<H", chunk_data, 14)[0]
        elif chunk_id == b"data":
            audio_data = chunk_data

        pos += 8 + chunk_size
        if chunk_size % 2 == 1:
            pos += 1

    if not audio_data:
        message = f"No data chunk in WAV file: {path}"
        raise ValueError(message)

    return fmt_tag, n_channels, sample_rate, bits_per_sample, audio_data


def _decode_wav_samples(
    np_module: object,
    *,
    fmt_tag: int,
    bits_per_sample: int,
    audio_data: bytes,
) -> object:
    """Decode a WAV payload into a NumPy array."""
    if fmt_tag == _PCM_FORMAT_TAG:
        if bits_per_sample == _PCM16_BITS:
            return np_module.frombuffer(audio_data, dtype=np_module.int16).astype(
                np_module.float32
            ) / _PCM16_SCALE
        if bits_per_sample == _PCM32_BITS:
            return np_module.frombuffer(audio_data, dtype=np_module.int32).astype(
                np_module.float32
            ) / _PCM32_SCALE
        message = f"Unsupported PCM bit depth: {bits_per_sample}"
        raise ValueError(message)
    if fmt_tag == _IEEE_FLOAT_FORMAT_TAG:
        return np_module.frombuffer(audio_data, dtype=np_module.float32).copy()
    message = f"Unsupported WAV format tag: {fmt_tag}"
    raise ValueError(message)


def _deinterleave_samples(samples: object, n_channels: int) -> list[list[float]]:
    """Convert interleaved samples into per-channel lists."""
    n_frames = len(samples) // n_channels
    return [samples[ch::n_channels][:n_frames].tolist() for ch in range(n_channels)]


def _parse_param_overrides(param_specs: list[str] | None) -> dict[str, float]:
    """Parse NAME=VALUE overrides from the CLI."""
    params: dict[str, float] = {}
    for spec in param_specs or []:
        if "=" not in spec:
            message = f"invalid param spec (expected NAME=VALUE): {spec}"
            raise ValueError(message)
        name, val_str = spec.split("=", 1)
        try:
            params[name] = float(val_str)
        except ValueError as e:
            message = f"invalid param value: {val_str}"
            raise ValueError(message) from e
    return params


def _load_simulation_inputs(
    graph: Graph,
    input_specs: list[str] | None,
    np_module: object,
) -> tuple[dict[str, object] | None, int | None]:
    """Load WAV inputs for simulation."""
    input_ids = {inp.id for inp in graph.inputs}
    if not input_specs:
        return None, None

    inputs: dict[str, object] = {}
    wav_sr: int | None = None
    for spec in input_specs:
        if "=" in spec:
            name, wav_path = spec.split("=", 1)
        else:
            wav_path = spec
            unmapped = input_ids - set(inputs.keys())
            if not unmapped:
                message = f"no unmapped input for file: {wav_path}"
                raise ValueError(message)
            name = sorted(unmapped)[0]

        if name not in input_ids:
            message = f"unknown input '{name}'"
            raise ValueError(message)

        channels, sr = _read_wav(wav_path)
        if wav_sr is None:
            wav_sr = sr
        inputs[name] = np_module.array(channels[0], dtype=np_module.float32)

    if input_ids and set(inputs.keys()) != input_ids:
        missing = input_ids - set(inputs.keys())
        if missing:
            message = f"missing input(s): {', '.join(sorted(missing))}"
            raise ValueError(message)

    return inputs, wav_sr


def _resolve_sample_rate(
    sample_rate_arg: float | None,
    wav_sr: int | None,
) -> float:
    """Resolve the effective sample rate for simulation."""
    sample_rate = float(sample_rate_arg) if sample_rate_arg else 0.0
    if sample_rate == 0.0 and wav_sr:
        return float(wav_sr)
    return sample_rate


def _prepare_simulation_run(
    args: argparse.Namespace,
    np: object,
) -> tuple[object, object, dict[str, object] | None, int, float] | None:
    """Load the graph and simulation inputs."""
    try:
        simulate = import_module("gen_dsp.graph.simulate").simulate
        graph = _load_graph(args.file)
        if args.optimize:
            graph, _stats = optimize_graph(graph)

        params = _parse_param_overrides(args.param)
        inputs, wav_sr = _load_simulation_inputs(graph, args.input, np)
        n_samples = args.samples or 0
        sample_rate = _resolve_sample_rate(args.sample_rate, wav_sr)
    except FileNotFoundError as e:
        _stderr(f"error: {e}")
        return None
    except json.JSONDecodeError as e:
        _stderr(f"error: invalid JSON: {e}")
        return None
    except ValidationError as e:
        _stderr(f"error: invalid graph: {e}")
        return None
    except ValueError as e:
        _stderr(f"error: {e}")
        return None
    except _gdsp_errors() as e:
        _stderr(f"error: {e}")
        return None

    return simulate, graph, inputs, n_samples, sample_rate, params


def _report_simulation_error(error: Exception) -> None:
    """Print a simulation error with the same labels as the CLI."""
    if isinstance(error, FileNotFoundError):
        _stderr(f"error: {error}")
    elif isinstance(error, json.JSONDecodeError):
        _stderr(f"error: invalid JSON: {error}")
    elif isinstance(error, ValidationError):
        _stderr(f"error: invalid graph: {error}")
    else:
        _stderr(f"error: {error}")


def _execute_simulation(
    simulate: object,
    prepared: tuple[
        object,
        object,
        dict[str, object] | None,
        int,
        float,
        dict[str, float],
    ],
) -> object | None:
    """Run the graph simulation and report runtime failures."""
    graph, inputs, n_samples, sample_rate, params = prepared[1:]
    try:
        return simulate(
            graph,
            inputs=inputs,
            n_samples=n_samples,
            params=params or None,
            sample_rate=sample_rate,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as e:
        _report_simulation_error(e)
        return None
    except _gdsp_errors() as e:
        _report_simulation_error(e)
        return None


def _write_simulation_outputs(
    result: object,
    graph: object,
    sample_rate: float,
    output_dir: str | None,
) -> None:
    """Write simulation outputs to WAV files."""
    out_dir = Path(output_dir) if output_dir else Path()
    out_dir.mkdir(parents=True, exist_ok=True)

    sr_out = int(sample_rate) if sample_rate > 0.0 else int(graph.sample_rate)
    for out_id, arr in result.outputs.items():
        wav_path = out_dir / f"{out_id}.wav"
        _write_wav(str(wav_path), arr.tolist(), sr_out)
        _stdout(f"wrote {wav_path} ({len(arr)} samples, {sr_out} Hz)")


def _run_simulation(
    args: argparse.Namespace,
    np: object,
) -> int:
    """Run graph simulation after loading inputs."""
    prepared = _prepare_simulation_run(args, np)
    if prepared is None:
        return 1

    simulate, graph, inputs, n_samples, sample_rate, _params = prepared
    if not inputs and n_samples == 0:
        _stderr(
            "error: -n/--samples is required when no input files are provided"
        )
        return 1

    result = _execute_simulation(
        simulate,
        prepared,
    )
    if result is None:
        return 1

    _write_simulation_outputs(result, graph, sample_rate, args.output)

    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers (public)
# ---------------------------------------------------------------------------


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile a graph to C++ (raw output, no platform adapter)."""
    status = 0
    try:
        graph = _load_graph(args.file)
        if args.optimize:
            graph, _stats = optimize_graph(graph)
        if args.output:
            compile_graph_to_file(graph, args.output)
        else:
            sys.stdout.write(compile_graph(graph))
    except FileNotFoundError as e:
        _stderr(f"error: {e}")
        status = 1
    except json.JSONDecodeError as e:
        _stderr(f"error: invalid JSON: {e}")
        status = 1
    except ValidationError as e:
        _stderr(f"error: invalid graph: {e}")
        status = 1
    except ValueError as e:
        _stderr(f"error: {e}")
        status = 1
    except _gdsp_errors() as e:
        _stderr(f"error: {e}")
        status = 1
    return status


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a graph file."""
    status = 0
    try:
        graph = _load_graph(args.file)
        warn = getattr(args, "warn_unmapped_params", False)
        errors = validate_graph(graph, warn_unmapped_params=warn)

        has_errors = any(e.severity == "error" for e in errors)
        has_warnings = any(e.severity == "warning" for e in errors)

        for err in errors:
            prefix = "warning" if err.severity == "warning" else "error"
            _stderr(f"{prefix}: {err}")

        if has_errors:
            status = 1
        elif has_warnings:
            _stdout("valid (with warnings)")
        else:
            _stdout("valid")
    except FileNotFoundError as e:
        _stderr(f"error: {e}")
        status = 1
    except json.JSONDecodeError as e:
        _stderr(f"error: invalid JSON: {e}")
        status = 1
    except ValidationError as e:
        _stderr(f"error: invalid graph: {e}")
        status = 1
    except ValueError as e:
        _stderr(f"error: {e}")
        status = 1
    except _gdsp_errors() as e:
        _stderr(f"error: {e}")
        status = 1
    return status


def cmd_dot(args: argparse.Namespace) -> int:
    """Generate DOT visualization."""
    status = 0
    try:
        graph = _load_graph(args.file)
        if args.output:
            graph_to_dot_file(graph, args.output)
        else:
            sys.stdout.write(graph_to_dot(graph))
    except FileNotFoundError as e:
        _stderr(f"error: {e}")
        status = 1
    except json.JSONDecodeError as e:
        _stderr(f"error: invalid JSON: {e}")
        status = 1
    except ValidationError as e:
        _stderr(f"error: invalid graph: {e}")
        status = 1
    except ValueError as e:
        _stderr(f"error: {e}")
        status = 1
    except _gdsp_errors() as e:
        _stderr(f"error: {e}")
        status = 1
    return status


def cmd_simulate(args: argparse.Namespace) -> int:
    """Simulate a graph (WAV in/out)."""
    try:
        np = import_module("numpy")
    except ImportError:
        _stderr(
            "error: numpy is required for simulation. Install with: "
            "pip install gen-dsp[sim]"
        )
        return 1
    return _run_simulation(args, np)


def _gdsp_errors() -> tuple[type[Exception], ...]:
    """Return GDSP error types for exception handling."""
    dsl_module = import_module("gen_dsp.graph.dsl")
    syntax_error = dsl_module.GDSPSyntaxError
    compile_error = dsl_module.GDSPCompileError
    return (syntax_error, compile_error)


# ---------------------------------------------------------------------------
# Individual parser-builder functions for gen-dsp's CLI
# ---------------------------------------------------------------------------

_FILE_HELP = "Graph file (.gdsp or .json)"


def add_compile_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the 'compile' subcommand (raw C++ output, no platform adapter)."""
    p = subparsers.add_parser("compile", help="Compile graph to C++")
    p.add_argument("file", help=_FILE_HELP)
    p.add_argument("-o", "--output", help="Output directory")
    p.add_argument("--optimize", action="store_true", help="Apply optimization passes")


def add_validate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the 'validate' subcommand."""
    p = subparsers.add_parser("validate", help="Validate graph")
    p.add_argument("file", help=_FILE_HELP)
    p.add_argument(
        "--warn-unmapped-params",
        action="store_true",
        help="Warn on unmapped subgraph params falling back to defaults",
    )


def add_dot_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the 'dot' subcommand."""
    p = subparsers.add_parser("dot", help="Generate DOT visualization")
    p.add_argument("file", help=_FILE_HELP)
    p.add_argument("-o", "--output", help="Output directory")


def add_sim_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the 'sim' subcommand."""
    p = subparsers.add_parser("sim", help="Simulate graph (WAV in/out)")
    p.add_argument("file", help=_FILE_HELP)
    p.add_argument(
        "-i",
        "--input",
        action="append",
        metavar="[NAME=]FILE",
        help="Map audio input to WAV file (repeatable)",
    )
    p.add_argument("-o", "--output", help="Output directory (default: current dir)")
    p.add_argument(
        "-n", "--samples", type=int, help="Number of samples (required for generators)"
    )
    p.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        help="Set parameter (repeatable)",
    )
    p.add_argument("--sample-rate", type=float, help="Override sample rate")
    p.add_argument("--optimize", action="store_true", help="Optimize before simulation")


# ---------------------------------------------------------------------------
# Standalone entry point (for testing)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point for dsp-graph CLI (for testing)."""
    parser = argparse.ArgumentParser(
        prog="dsp-graph",
        description="Compile, validate, visualize, and simulate DSP signal graphs.",
    )
    sub = parser.add_subparsers(dest="command")

    # compile (no -p in standalone mode either)
    p_compile = sub.add_parser("compile", help="Compile graph to C++")
    p_compile.add_argument("file", help=_FILE_HELP)
    p_compile.add_argument("-o", "--output", help="Output directory")
    p_compile.add_argument(
        "--optimize", action="store_true", help="Apply optimization passes"
    )

    # validate
    p_validate = sub.add_parser("validate", help="Validate graph")
    p_validate.add_argument("file", help=_FILE_HELP)
    p_validate.add_argument(
        "--warn-unmapped-params",
        action="store_true",
        help="Warn on unmapped subgraph params falling back to defaults",
    )

    # dot
    p_dot = sub.add_parser("dot", help="Generate DOT visualization")
    p_dot.add_argument("file", help=_FILE_HELP)
    p_dot.add_argument("-o", "--output", help="Output directory")

    # sim
    p_sim = sub.add_parser("sim", help="Simulate graph (WAV in/out)")
    p_sim.add_argument("file", help=_FILE_HELP)
    p_sim.add_argument(
        "-i",
        "--input",
        action="append",
        metavar="[NAME=]FILE",
        help="Map audio input to WAV file (repeatable)",
    )
    p_sim.add_argument("-o", "--output", help="Output directory (default: current dir)")
    p_sim.add_argument(
        "-n", "--samples", type=int, help="Number of samples (required for generators)"
    )
    p_sim.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        help="Set parameter (repeatable)",
    )
    p_sim.add_argument("--sample-rate", type=float, help="Override sample rate")
    p_sim.add_argument(
        "--optimize", action="store_true", help="Optimize before simulation"
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 1

    if args.command == "compile":
        return cmd_compile(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "dot":
        return cmd_dot(args)
    if args.command == "sim":
        return cmd_simulate(args)

    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
