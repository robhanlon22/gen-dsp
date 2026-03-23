"""
Mono chorus effect using a modulated delay line.

An LFO modulates the delay tap position around a 10ms center to create
the characteristic pitch wobble. Demonstrates SinOsc with DelayLine.

Usage:
    python examples/graph/chorus.py -p vst3 [-o OUTPUT_DIR]
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
    DelayLine,
    DelayRead,
    DelayWrite,
    Graph,
    Param,
    SinOsc,
)
from gen_dsp.graph.visualize import graph_to_dot_file


def make_graph() -> Graph:
    """Build the example graph."""
    return Graph(
        name="chorus",
        inputs=[AudioInput(id="in1")],
        outputs=[AudioOutput(id="out1", source="mix_out")],
        params=[
            Param(name="rate", min=0.1, max=5.0, default=1.5),
            Param(name="depth", min=0.0, max=1.0, default=0.5),
            Param(name="mix", min=0.0, max=1.0, default=0.5),
        ],
        nodes=[
            # LFO modulates delay time around center (~10ms = 441 samples)
            SinOsc(id="lfo", freq="rate"),
            BinOp(id="depth_samp", op="mul", a="depth", b=220.0),
            BinOp(id="mod", op="mul", a="lfo", b="depth_samp"),
            BinOp(id="tap", op="add", a=441.0, b="mod"),
            # Delay line
            DelayLine(id="dline", max_samples=2048),
            DelayWrite(id="dwrite", delay="dline", value="in1"),
            DelayRead(id="delayed", delay="dline", tap="tap"),
            # Dry/wet mix
            BinOp(id="inv_mix", op="sub", a=1.0, b="mix"),
            BinOp(id="dry", op="mul", a="in1", b="inv_mix"),
            BinOp(id="wet", op="mul", a="delayed", b="mix"),
            BinOp(id="mix_out", op="add", a="dry", b="wet"),
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
