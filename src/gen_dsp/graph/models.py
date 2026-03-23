"""Pydantic models for DSP graph nodes and graph containers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Type alias for node input references: either a node/input/param ID or a literal float.
Ref = str | float


# ---------------------------------------------------------------------------
# Param & I/O declarations
# ---------------------------------------------------------------------------


class Param(BaseModel):
    """model."""

    name: str
    min: float = 0.0
    max: float = 1.0
    default: float = 0.0


class AudioInput(BaseModel):
    """model."""

    id: str


class AudioOutput(BaseModel):
    """model."""

    id: str
    source: str  # node ID that feeds this output


# ---------------------------------------------------------------------------
# Node types (discriminated union on "op")
# ---------------------------------------------------------------------------


class BinOp(BaseModel):
    """model."""

    id: str
    op: Literal[
        "add",
        "sub",
        "mul",
        "div",
        "min",
        "max",
        "mod",
        "pow",
        "atan2",
        "hypot",
        "absdiff",
        "step",
        "and",
        "or",
        "xor",
        "rsub",
        "rdiv",
        "rmod",
        "gtp",
        "ltp",
        "gtep",
        "ltep",
        "eqp",
        "neqp",
        "fastpow",
    ]
    a: Ref
    b: Ref


class UnaryOp(BaseModel):
    """model."""

    id: str
    op: Literal[
        "sin",
        "cos",
        "tanh",
        "exp",
        "log",
        "abs",
        "sqrt",
        "neg",
        "floor",
        "ceil",
        "round",
        "sign",
        "atan",
        "asin",
        "acos",
        "tan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "exp2",
        "log2",
        "log10",
        "fract",
        "trunc",
        "not",
        "bool",
        "mtof",
        "ftom",
        "atodb",
        "dbtoa",
        "phasewrap",
        "degrees",
        "radians",
        "mstosamps",
        "sampstoms",
        "t60",
        "t60time",
        "fixdenorm",
        "fixnan",
        "isdenorm",
        "isnan",
        "fastsin",
        "fastcos",
        "fasttan",
        "fastexp",
    ]
    a: Ref


class Clamp(BaseModel):
    """model."""

    id: str
    op: Literal["clamp"] = "clamp"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Constant(BaseModel):
    """model."""

    id: str
    op: Literal["constant"] = "constant"
    value: float


class History(BaseModel):
    """model."""

    id: str
    op: Literal["history"] = "history"
    init: float = 0.0
    input: str  # node ID whose value is stored for next sample


class DelayLine(BaseModel):
    """model."""

    id: str
    op: Literal["delay"] = "delay"
    max_samples: int = 48000


class DelayRead(BaseModel):
    """model."""

    id: str
    op: Literal["delay_read"] = "delay_read"
    delay: str  # delay line ID
    tap: Ref  # tap position: node ID or literal
    interp: Literal["none", "linear", "cubic"] = "none"


class DelayWrite(BaseModel):
    """model."""

    id: str
    op: Literal["delay_write"] = "delay_write"
    delay: str  # delay line ID
    value: Ref  # node ID or literal to write


class Phasor(BaseModel):
    """model."""

    id: str
    op: Literal["phasor"] = "phasor"
    freq: Ref


class Noise(BaseModel):
    """model."""

    id: str
    op: Literal["noise"] = "noise"


class Compare(BaseModel):
    """model."""

    id: str
    op: Literal["gt", "lt", "gte", "lte", "eq", "neq"]
    a: Ref
    b: Ref


class Select(BaseModel):
    """model."""

    id: str
    op: Literal["select"] = "select"
    cond: Ref
    a: Ref
    b: Ref


class Wrap(BaseModel):
    """model."""

    id: str
    op: Literal["wrap"] = "wrap"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Fold(BaseModel):
    """model."""

    id: str
    op: Literal["fold"] = "fold"
    a: Ref
    lo: Ref = 0.0
    hi: Ref = 1.0


class Mix(BaseModel):
    """model."""

    id: str
    op: Literal["mix"] = "mix"
    a: Ref
    b: Ref
    t: Ref


class Delta(BaseModel):
    """model."""

    id: str
    op: Literal["delta"] = "delta"
    a: Ref


class Change(BaseModel):
    """model."""

    id: str
    op: Literal["change"] = "change"
    a: Ref


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class Biquad(BaseModel):
    """model."""

    id: str
    op: Literal["biquad"] = "biquad"
    a: Ref
    b0: Ref
    b1: Ref
    b2: Ref
    a1: Ref
    a2: Ref


class SVF(BaseModel):
    """model."""

    id: str
    op: Literal["svf"] = "svf"
    a: Ref
    freq: Ref
    q: Ref
    mode: Literal["lp", "hp", "bp", "notch"]


class OnePole(BaseModel):
    """model."""

    id: str
    op: Literal["onepole"] = "onepole"
    a: Ref
    coeff: Ref


class DCBlock(BaseModel):
    """model."""

    id: str
    op: Literal["dcblock"] = "dcblock"
    a: Ref


class Allpass(BaseModel):
    """model."""

    id: str
    op: Literal["allpass"] = "allpass"
    a: Ref
    coeff: Ref


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------


class SinOsc(BaseModel):
    """model."""

    id: str
    op: Literal["sinosc"] = "sinosc"
    freq: Ref


class TriOsc(BaseModel):
    """model."""

    id: str
    op: Literal["triosc"] = "triosc"
    freq: Ref


class SawOsc(BaseModel):
    """model."""

    id: str
    op: Literal["sawosc"] = "sawosc"
    freq: Ref


class PulseOsc(BaseModel):
    """model."""

    id: str
    op: Literal["pulseosc"] = "pulseosc"
    freq: Ref
    width: Ref


# ---------------------------------------------------------------------------
# State / Timing
# ---------------------------------------------------------------------------


class SampleHold(BaseModel):
    """model."""

    id: str
    op: Literal["sample_hold"] = "sample_hold"
    a: Ref
    trig: Ref


class Latch(BaseModel):
    """model."""

    id: str
    op: Literal["latch"] = "latch"
    a: Ref
    trig: Ref


class Accum(BaseModel):
    """model."""

    id: str
    op: Literal["accum"] = "accum"
    incr: Ref
    reset: Ref


class Counter(BaseModel):
    """model."""

    id: str
    op: Literal["counter"] = "counter"
    trig: Ref
    max: Ref


class Elapsed(BaseModel):
    """model."""

    id: str
    op: Literal["elapsed"] = "elapsed"


class MulAccum(BaseModel):
    """model."""

    id: str
    op: Literal["mulaccum"] = "mulaccum"
    incr: Ref
    reset: Ref


class RateDiv(BaseModel):
    """model."""

    id: str
    op: Literal["rate_div"] = "rate_div"
    a: Ref
    divisor: Ref


class SmoothParam(BaseModel):
    """model."""

    id: str
    op: Literal["smooth"] = "smooth"
    a: Ref
    coeff: Ref


class Slide(BaseModel):
    """model."""

    id: str
    op: Literal["slide"] = "slide"
    a: Ref  # input signal
    up: Ref  # slide-up rate (samples)
    down: Ref  # slide-down rate (samples)


class ADSR(BaseModel):
    """model."""

    id: str
    op: Literal["adsr"] = "adsr"
    gate: Ref  # >0 = attack/sustain, <=0 = release
    attack: Ref  # attack time (ms)
    decay: Ref  # decay time (ms)
    sustain: Ref  # sustain level [0,1]
    release: Ref  # release time (ms)


class Peek(BaseModel):
    """model."""

    id: str
    op: Literal["peek"] = "peek"
    a: Ref


# ---------------------------------------------------------------------------
# Range mapping
# ---------------------------------------------------------------------------


class Scale(BaseModel):
    """model."""

    id: str
    op: Literal["scale"] = "scale"
    a: Ref
    in_lo: Ref = 0.0
    in_hi: Ref = 1.0
    out_lo: Ref = 0.0
    out_hi: Ref = 1.0


# ---------------------------------------------------------------------------
# Identity / Named constants / Smoothstep
# ---------------------------------------------------------------------------


class Pass(BaseModel):
    """model."""

    id: str
    op: Literal["pass"] = "pass"
    a: Ref


class NamedConstant(BaseModel):
    """model."""

    id: str
    op: Literal[
        "pi",
        "e",
        "twopi",
        "halfpi",
        "invpi",
        "degtorad",
        "radtodeg",
        "sqrt2",
        "sqrt1_2",
        "ln2",
        "ln10",
        "log2e",
        "log10e",
        "phi",
    ]


class SampleRate(BaseModel):
    """model."""

    id: str
    op: Literal["samplerate"] = "samplerate"


class Smoothstep(BaseModel):
    """model."""

    id: str
    op: Literal["smoothstep"] = "smoothstep"
    a: Ref
    edge0: Ref
    edge1: Ref


# ---------------------------------------------------------------------------
# Subgraph / Macro
# ---------------------------------------------------------------------------


class Subgraph(BaseModel):
    """model."""

    id: str
    op: Literal["subgraph"] = "subgraph"
    graph: Graph
    inputs: dict[str, Ref]
    params: dict[str, Ref] = Field(default_factory=dict)
    output: str = ""


# ---------------------------------------------------------------------------
# Buffer / Table
# ---------------------------------------------------------------------------


class Buffer(BaseModel):
    """model."""

    id: str
    op: Literal["buffer"] = "buffer"
    size: int = 48000
    fill: Literal["zeros", "sine"] = "zeros"


class BufRead(BaseModel):
    """model."""

    id: str
    op: Literal["buf_read"] = "buf_read"
    buffer: str  # Buffer node ID
    index: Ref  # read position (float, will be clamped)
    interp: Literal["none", "linear", "cubic"] = "none"


class BufWrite(BaseModel):
    """model."""

    id: str
    op: Literal["buf_write"] = "buf_write"
    buffer: str  # Buffer node ID
    index: Ref  # write position
    value: Ref  # value to write


class Splat(BaseModel):
    """model."""

    id: str
    op: Literal["splat"] = "splat"
    buffer: str  # Buffer node ID
    index: Ref  # write position
    value: Ref  # value to add (overdub)


class BufSize(BaseModel):
    """model."""

    id: str
    op: Literal["buf_size"] = "buf_size"
    buffer: str  # Buffer node ID


class Cycle(BaseModel):
    """model."""

    id: str
    op: Literal["cycle"] = "cycle"
    buffer: str  # Buffer node ID
    phase: Ref  # [0, 1) phase, wraps


class Wave(BaseModel):
    """model."""

    id: str
    op: Literal["wave"] = "wave"
    buffer: str  # Buffer node ID
    phase: Ref  # [-1, 1] phase, maps to full buffer


class Lookup(BaseModel):
    """model."""

    id: str
    op: Literal["lookup"] = "lookup"
    buffer: str  # Buffer node ID
    index: Ref  # [0, 1] index, clamped


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class GateRoute(BaseModel):
    """model."""

    id: str
    op: Literal["gate_route"] = "gate_route"
    a: Ref  # input signal
    index: Ref  # 1-based channel selector (0 = mute all)
    count: int  # number of output channels


class GateOut(BaseModel):
    """model."""

    id: str
    op: Literal["gate_out"] = "gate_out"
    gate: str  # GateRoute node ID
    channel: int  # 1-based channel number this output reads


class Selector(BaseModel):
    """model."""

    id: str
    op: Literal["selector"] = "selector"
    index: Ref  # 1-based input selector (0 = zero output)
    inputs: list[Ref]  # the N inputs to select from


# Discriminated union of all node types
Node = Annotated[
    BinOp
    | UnaryOp
    | Clamp
    | Constant
    | History
    | DelayLine
    | DelayRead
    | DelayWrite
    | Phasor
    | Noise
    | Compare
    | Select
    | Wrap
    | Fold
    | Mix
    | Delta
    | Change
    | Biquad
    | SVF
    | OnePole
    | DCBlock
    | Allpass
    | SinOsc
    | TriOsc
    | SawOsc
    | PulseOsc
    | SampleHold
    | Latch
    | Accum
    | Counter
    | Elapsed
    | MulAccum
    | RateDiv
    | SmoothParam
    | Slide
    | ADSR
    | Peek
    | Scale
    | Pass
    | NamedConstant
    | SampleRate
    | Smoothstep
    | Subgraph
    | Buffer
    | BufRead
    | BufWrite
    | Splat
    | BufSize
    | Cycle
    | Wave
    | Lookup
    | GateRoute
    | GateOut
    | Selector,
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Top-level graph
# ---------------------------------------------------------------------------


class Graph(BaseModel):
    """model."""

    name: str
    sample_rate: float = 44100.0
    control_interval: int = 0  # 0 = disabled; >0 = samples per control block
    control_nodes: list[str] = []  # node IDs that run at control rate
    inputs: list[AudioInput] = []
    outputs: list[AudioOutput] = []
    params: list[Param] = []
    nodes: list[Node] = []


# Resolve circular reference: Subgraph.graph -> Graph -> list[Node] -> Subgraph
Subgraph.model_rebuild()
