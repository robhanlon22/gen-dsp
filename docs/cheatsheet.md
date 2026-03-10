# gen-dsp CLI Cheatsheet

## Default Command -- Generate a Plugin Project

```bash
gen-dsp <source> -p <platform> [options]
```

Source type is auto-detected: directory (gen~ export), `.gdsp` file, or `.json` file.

| Option | Description |
|--------|-------------|
| `-p, --platform PLATFORM` | Target platform (required): `au`, `chuck`, `circle`, `clap`, `daisy`, `lv2`, `max`, `pd`, `sc`, `vcvrack`, `vst3`, `webaudio` |
| `-n, --name NAME` | Plugin name (default: inferred from source) |
| `-o, --output DIR` | Output directory (default: `<name>_<platform>`) |
| `--no-build` | Skip building after project creation |
| `--dry-run` | Show what would be done without creating files |
| `--buffers NAME [...]` | Explicit buffer names (overrides auto-detection) |
| `--no-patch` | Skip platform patches (e.g. `exp2f` fix) |
| `--no-shared-cache` | Disable shared OS cache for FetchContent downloads |
| `--board BOARD` | Board variant (daisy, circle) |
| `--no-midi` | Disable MIDI note handling |
| `--midi-gate NAME` | MIDI gate parameter name |
| `--midi-freq NAME` | MIDI frequency parameter name |
| `--midi-vel NAME` | MIDI velocity parameter name |
| `--midi-freq-unit {hz,midi}` | Unit for MIDI frequency parameter |
| `--voices N` | Polyphony voices (default: 1) |
| `--inputs-as-params [NAME ...]` | Remap signal inputs to parameters (no names = all; with names = only those) |

Examples:

```bash
gen-dsp ./my_export -p vst3
gen-dsp ./my_export -p clap -n myeffect -o ./build
gen-dsp ./my_export -p daisy --board pod
gen-dsp ./my_export -p pd --no-build --buffers sample envelope
gen-dsp synth.gdsp -p clap --midi-freq freq --midi-gate gate --voices 4
```

## build -- Build an Existing Project

```bash
gen-dsp build [project-path] [-p PLATFORM] [--clean] [-v]
```

| Option | Description |
|--------|-------------|
| `project-path` | Path to project directory (default: current directory) |
| `-p, --platform PLATFORM` | Target platform (default: `pd`) |
| `--clean` | Clean before building |
| `-v, --verbose` | Show build output |

## detect -- Analyze a gen~ Export

```bash
gen-dsp detect <export-path> [--json]
```

Shows export name, signal I/O counts, parameters, detected buffers, and needed patches.

## manifest -- Emit JSON Manifest

```bash
gen-dsp manifest <export-path> [--buffers NAME ...]
```

Outputs a JSON manifest describing I/O counts, parameters with ranges, and buffers.

## patch -- Apply Platform-Specific Patches

```bash
gen-dsp patch <target-path> [--dry-run]
```

Currently applies the `exp2f -> exp2` fix for macOS compatibility with Max 9 exports.

## chain -- Multi-Plugin Chain Mode (Circle)

```bash
gen-dsp chain <export-path> --graph GRAPH -n NAME [-p PLATFORM] [-o OUTPUT] [options]
```

| Option | Description |
|--------|-------------|
| `export-path` | Base directory for gen~ exports |
| `--graph GRAPH` | JSON graph file for multi-plugin chain (required) |
| `-n, --name NAME` | Name for the chain project (required) |
| `-p, --platform PLATFORM` | Target platform (default: `circle`) |
| `-o, --output OUTPUT` | Output directory (default: `./<name>`) |
| `--export EXPORTS` | Additional export path (repeatable) |
| `--board BOARD` | Board variant |
| `--no-patch` | Skip platform patches |
| `--no-build` | Skip building |
| `--dry-run` | Show what would be done |

## compile -- Compile Graph to C++ (requires gen-dsp[graph])

```bash
gen-dsp compile <file> [-o DIR] [--optimize]
```

Compiles a `.gdsp` or `.json` graph file to C++. Outputs to stdout by default, or to a directory with `-o`.

## validate -- Validate a Graph File (requires gen-dsp[graph])

```bash
gen-dsp validate <file> [--warn-unmapped-params]
```

Checks graph connectivity and type correctness.

## dot -- Generate DOT Visualization (requires gen-dsp[graph])

```bash
gen-dsp dot <file> [-o DIR]
```

Generates a Graphviz DOT file for the graph.

## sim -- Simulate a Graph (requires gen-dsp[sim])

```bash
gen-dsp sim <file> [-i [NAME=]FILE] [-o DIR] [-n SAMPLES] [--param NAME=VALUE] [--sample-rate SR] [--optimize]
```

| Option | Description |
|--------|-------------|
| `-i, --input [NAME=]FILE` | Map audio input to WAV file (repeatable) |
| `-o, --output DIR` | Output directory (default: current dir) |
| `-n, --samples N` | Number of samples (required for generators) |
| `--param NAME=VALUE` | Set parameter (repeatable) |
| `--sample-rate SR` | Override sample rate |
| `--optimize` | Optimize before simulation |

## list -- List Available Platforms

```bash
gen-dsp list
```

## cache -- Show Cached SDKs

```bash
gen-dsp cache
```

## Global Options

| Option | Description |
|--------|-------------|
| `-V, --version` | Show version |
| `-h, --help` | Show help |
