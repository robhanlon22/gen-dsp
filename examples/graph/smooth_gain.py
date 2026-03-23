"""
Smoothed stereo gain with control-rate parameter interpolation.

Demonstrates multi-rate processing: the gain parameter is smoothed at
control rate (once per 64-sample block) to eliminate zipper noise, then
applied per-sample in the audio-rate inner loop.

Usage:
    python examples/graph/smooth_gain.py -p vst3 [-o OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

from gen_dsp.core.builder import Builder
from gen_dsp.core.project import ProjectConfig, ProjectGenerator
from gen_dsp.graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Graph,
    Param,
    SmoothParam,
)
from gen_dsp.graph.visualize import graph_to_dot_file


def make_graph() -> Graph:
    """Build the example graph."""
    return Graph(
        name="smooth_gain",
        sample_rate=48000.0,
        control_interval=64,
        control_nodes=["smooth_vol"],
        inputs=[AudioInput(id="in1"), AudioInput(id="in2")],
        outputs=[
            AudioOutput(id="out1", source="scaled1"),
            AudioOutput(id="out2", source="scaled2"),
        ],
        params=[Param(name="vol", min=0.0, max=2.0, default=1.0)],
        nodes=[
            # Control-rate: smooth the volume param (updates every 64 samples)
            SmoothParam(id="smooth_vol", a="vol", coeff=0.99),
            # Audio-rate: apply smoothed gain per sample
            BinOp(id="scaled1", op="mul", a="in1", b="smooth_vol"),
            BinOp(id="scaled2", op="mul", a="in2", b="smooth_vol"),
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
