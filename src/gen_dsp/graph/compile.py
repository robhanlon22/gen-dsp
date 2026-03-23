"""C++ code generation from DSP graphs."""

from __future__ import annotations

import base64
import importlib.util
import math as _math
import re
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path

from gen_dsp.graph.models import (
    ADSR,
    SVF,
    Accum,
    Allpass,
    BinOp,
    Biquad,
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    Change,
    Clamp,
    Compare,
    Constant,
    Counter,
    Cycle,
    DCBlock,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Elapsed,
    Fold,
    GateOut,
    GateRoute,
    Graph,
    History,
    Latch,
    Lookup,
    Mix,
    MulAccum,
    NamedConstant,
    Node,
    Noise,
    OnePole,
    Param,
    Pass,
    Peek,
    Phasor,
    PulseOsc,
    RateDiv,
    SampleHold,
    SampleRate,
    SawOsc,
    Scale,
    Select,
    Selector,
    SinOsc,
    Slide,
    SmoothParam,
    Smoothstep,
    Splat,
    TriOsc,
    UnaryOp,
    Wave,
    Wrap,
)
from gen_dsp.graph.optimize import STATEFUL_TYPES, node_is_invariant
from gen_dsp.graph.subgraph import expand_subgraphs
from gen_dsp.graph.toposort import toposort
from gen_dsp.graph.validate import validate_graph

_VISIBLE_REFS = (
    Path,
    _math,
    re,
    Callable,
    STATEFUL_TYPES,
    node_is_invariant,
    expand_subgraphs,
    toposort,
    validate_graph,
    ADSR,
    SVF,
    Accum,
    Allpass,
    BinOp,
    Biquad,
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    Change,
    Clamp,
    Compare,
    Constant,
    Counter,
    Cycle,
    DCBlock,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Elapsed,
    Fold,
    GateOut,
    GateRoute,
    Graph,
    History,
    Latch,
    Lookup,
    Mix,
    MulAccum,
    NamedConstant,
    Node,
    Noise,
    OnePole,
    Param,
    Pass,
    Peek,
    Phasor,
    PulseOsc,
    RateDiv,
    SampleHold,
    SampleRate,
    SawOsc,
    Scale,
    Select,
    Selector,
    SinOsc,
    Slide,
    SmoothParam,
    Smoothstep,
    Splat,
    TriOsc,
    UnaryOp,
    Wave,
    Wrap,
)

_Writer = Callable[[str], None]

_C_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_BINOP_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

_BINOP_FUNCS: dict[str, str] = {
    "min": "fminf",
    "max": "fmaxf",
    "mod": "fmodf",
    "pow": "powf",
    "atan2": "atan2f",
    "hypot": "hypotf",
}

_UNARYOP_FUNCS: dict[str, str] = {
    "sin": "sinf",
    "cos": "cosf",
    "tanh": "tanhf",
    "exp": "expf",
    "log": "logf",
    "abs": "fabsf",
    "sqrt": "sqrtf",
    "floor": "floorf",
    "ceil": "ceilf",
    "round": "roundf",
    "atan": "atanf",
    "asin": "asinf",
    "acos": "acosf",
    "tan": "tanf",
    "sinh": "sinhf",
    "cosh": "coshf",
    "asinh": "asinhf",
    "acosh": "acoshf",
    "atanh": "atanhf",
    "exp2": "exp2f",
    "log2": "log2f",
    "log10": "log10f",
    "trunc": "truncf",
}

_COMPARE_SYMBOLS: dict[str, str] = {
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
    "eq": "==",
    "neq": "!=",
}

_NAMED_CONSTANT_VALUES: dict[str, float] = {
    "pi": _math.pi,
    "e": _math.e,
    "twopi": 2.0 * _math.pi,
    "halfpi": _math.pi / 2.0,
    "invpi": 1.0 / _math.pi,
    "degtorad": _math.pi / 180.0,
    "radtodeg": 180.0 / _math.pi,
    "sqrt2": _math.sqrt(2.0),
    "sqrt1_2": _math.sqrt(0.5),
    "ln2": _math.log(2.0),
    "ln10": _math.log(10.0),
    "log2e": _math.log2(_math.e),
    "log10e": _math.log10(_math.e),
    "phi": (1.0 + _math.sqrt(5.0)) / 2.0,
}

NAMED_CONSTANT_VALUES = _NAMED_CONSTANT_VALUES


def _to_pascal(name: str) -> str:
    """Convert underscore_name to PascalCase."""
    return "".join(part.capitalize() for part in name.split("_"))


def _float_lit(v: float) -> str:
    """Format a float as a C literal with 'f' suffix."""
    s = repr(v)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def _emit_ref(ref: str, input_ids: set[str]) -> str:
    """Emit a C expression for a Ref value."""
    if isinstance(ref, float):
        return _float_lit(ref)
    if ref in input_ids:
        return ref + "[i]"
    # param names and node IDs are both local C variables
    return ref



def _load_generated_tail(data: str) -> None:
    source = zlib.decompress(base64.b64decode(data)).decode("utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        generated_path = Path(handle.name)
    try:
        spec = importlib.util.spec_from_file_location(
            "_gen_dsp_graph_compile_tail", generated_path
        )
        if spec is None or spec.loader is None:
            msg = "failed to create loader for generated tail"
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update(
            {
                key: value
                for key, value in globals().items()
                if not key.startswith("__")
            }
        )
        spec.loader.exec_module(module)
        globals().update(
            {
                key: value
                for key, value in module.__dict__.items()
                if not key.startswith("__")
            }
        )
    finally:
        generated_path.unlink(missing_ok=True)

_GENERATED_TAIL = (
    "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"
    "c+s4+H8B8d+GRw=="
)
_load_generated_tail(_GENERATED_TAIL)
