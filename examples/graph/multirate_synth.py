"""
Multi-rate synth: control-rate envelope with audio-rate oscillator.

A generator (no audio inputs) with two-tier processing:
  - Control-rate: SmoothParam tracks the amplitude target, updating
    every 32 samples.
  - Audio-rate: sine oscillator runs per-sample, scaled by the
    held envelope value.

Usage:
    python examples/graph/multirate_synth.py -p sc [-o OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

from gen_dsp.core.builder import Builder
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.graph import (
    AudioOutput,
    BinOp,
    Graph,
    Param,
    SinOsc,
    SmoothParam,
)
from gen_dsp.graph.visualize import graph_to_dot_file


def make_graph() -> Graph:
    """Build the example graph."""
    return Graph(
        name="multirate_synth",
        sample_rate=48000.0,
        control_interval=32,
        control_nodes=["env"],
        inputs=[],
        outputs=[AudioOutput(id="out1", source="scaled")],
        params=[
            Param(name="freq", min=20.0, max=20000.0, default=440.0),
            Param(name="amp", min=0.0, max=1.0, default=0.0),
        ],
        nodes=[
            # Control-rate: smooth amplitude envelope
            SmoothParam(id="env", a="amp", coeff=0.995),
            # Audio-rate: sine oscillator and gain
            SinOsc(id="osc", freq="freq"),
            BinOp(id="scaled", op="mul", a="osc", b="env"),
        ],
    )


def main() -> None:
    """Run the example CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p", "--platform", default=None, help="Target platform (clap, vst3, au, ...)"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available platforms"
    )
    parser.add_argument(
        "-b", "--build", action="store_true", help="Build after generating"
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "-d", "--dot", action="store_true", help="Generate Graphviz DOT graph as PDF"
    )
    args = parser.parse_args()

    if args.list:
        platforms = ", ".join(ProjectConfig.list_platforms())
        sys.stdout.write(f"Available platforms: {platforms}\n")
        return
    graph = make_graph()
    if args.dot:
        dot_path = graph_to_dot_file(graph, args.output or Path())
        sys.stdout.write(f"DOT: {dot_path}\n")
        return
    if not args.platform:
        parser.error("-p/--platform is required (use -l to list available platforms)")

    output = args.output or Path(f"build/examples/{graph.name}_{args.platform}")
    config = ProjectConfig(name=graph.name, platform=args.platform)
    gen = ProjectGenerator.from_graph(graph, config)
    project_dir = gen.generate(output_dir=output)

    sys.stdout.write(f"Project generated at: {project_dir}\n")
    if args.build:
        result = Builder(project_dir).build(args.platform, verbose=True)
        status = "succeeded" if result.success else "failed"
        sys.stdout.write(f"Build {status}: {result}\n")
    else:
        build_cmd = f"cd {project_dir} && cmake -B build && cmake --build build"
        sys.stdout.write(f"Build with: {build_cmd}\n")


if __name__ == "__main__":
    main()
