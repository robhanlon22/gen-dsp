"""
MIDI-to-CV auto-detection and mapping for gen-dsp instruments.

Scans gen~ parameter names to detect MIDI-mappable parameters (gate, freq, vel)
and produces compile-time constants for the platform wrappers.

Only activates for 0-input (generator) plugins. Effects are never MIDI-mapped.
"""

from dataclasses import dataclass

from gen_dsp.core.manifest import Manifest

# Parameter name patterns for auto-detection (case-sensitive, gen~ uses lowercase)
_GATE_NAMES = {"gate"}
_FREQ_NAMES = {"freq", "frequency", "pitch"}
_VEL_NAMES = {"vel", "velocity"}


@dataclass
class MidiMapping:
    """
    Compile-time MIDI-to-parameter mapping for instrument plugins.

    Attributes:
        enabled: Whether MIDI note handling code should be generated.
        gate_idx: Parameter index for gate (None if not mapped).
        freq_idx: Parameter index for frequency (None if not mapped).
        vel_idx: Parameter index for velocity (None if not mapped).
        freq_unit: "hz" for mtof conversion, "midi" for raw note number.

    """

    enabled: bool
    gate_idx: int | None = None
    freq_idx: int | None = None
    vel_idx: int | None = None
    freq_unit: str = "hz"
    num_voices: int = 1


@dataclass
class MidiDetectionOptions:
    """Explicit MIDI detection overrides."""

    no_midi: bool = False
    midi_gate: str | None = None
    midi_freq: str | None = None
    midi_vel: str | None = None
    midi_freq_unit: str = "hz"


def detect_midi_mapping(
    manifest: Manifest,
    options: MidiDetectionOptions | None = None,
    **legacy_kwargs: object,
) -> MidiMapping:
    """
    Detect MIDI parameter mapping from a manifest.

    Detection rules:
    1. Only for 0-input plugins (generators). Effects -> disabled.
    2. If no_midi is True -> disabled.
    3. If any explicit --midi-* name is provided, use those (implies enabled).
    4. Otherwise auto-detect by scanning param names for known patterns.
       Gate is required for auto-detection to activate.

    Args:
        manifest: The plugin manifest with param metadata.
        options: Explicit override settings.
        legacy_kwargs: Deprecated keyword overrides kept for compatibility.

    Returns:
        MidiMapping with detected/configured indices.

    """
    if options is None:
        options = MidiDetectionOptions(
            no_midi=bool(legacy_kwargs.pop("no_midi", False)),
            midi_gate=legacy_kwargs.pop("midi_gate", None),
            midi_freq=legacy_kwargs.pop("midi_freq", None),
            midi_vel=legacy_kwargs.pop("midi_vel", None),
            midi_freq_unit=str(legacy_kwargs.pop("midi_freq_unit", "hz")),
        )
    elif legacy_kwargs:
        options = MidiDetectionOptions(
            no_midi=bool(legacy_kwargs.pop("no_midi", options.no_midi)),
            midi_gate=legacy_kwargs.pop("midi_gate", options.midi_gate),
            midi_freq=legacy_kwargs.pop("midi_freq", options.midi_freq),
            midi_vel=legacy_kwargs.pop("midi_vel", options.midi_vel),
            midi_freq_unit=str(
                legacy_kwargs.pop("midi_freq_unit", options.midi_freq_unit)
            ),
        )

    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        msg = f"Unexpected keyword arguments: {unexpected}"
        raise TypeError(msg)

    disabled = MidiMapping(enabled=False)

    # Explicit opt-out
    if options.no_midi:
        return disabled

    # Build name->index lookup
    param_by_name: dict[str, int] = {p.name: p.index for p in manifest.params}

    # Check for explicit overrides -- these bypass the num_inputs guard
    # because the user knows their patch topology better than auto-detection
    has_explicit = (
        options.midi_gate is not None
        or options.midi_freq is not None
        or options.midi_vel is not None
    )

    if has_explicit:
        gate_idx = param_by_name.get(options.midi_gate) if options.midi_gate else None
        freq_idx = param_by_name.get(options.midi_freq) if options.midi_freq else None
        vel_idx = param_by_name.get(options.midi_vel) if options.midi_vel else None
        return MidiMapping(
            enabled=True,
            gate_idx=gate_idx,
            freq_idx=freq_idx,
            vel_idx=vel_idx,
            freq_unit=options.midi_freq_unit,
        )

    # Auto-detection only applies to generators (0 inputs).
    # Effects (num_inputs > 0) require explicit --midi-* flags.
    if manifest.num_inputs > 0:
        return disabled

    # Auto-detection: scan param names
    gate_idx = _find_param_index(param_by_name, _GATE_NAMES)

    # Gate is required for auto-detection
    if gate_idx is None:
        return disabled

    freq_idx = _find_param_index(param_by_name, _FREQ_NAMES)
    vel_idx = _find_param_index(param_by_name, _VEL_NAMES)

    return MidiMapping(
        enabled=True,
        gate_idx=gate_idx,
        freq_idx=freq_idx,
        vel_idx=vel_idx,
        freq_unit=options.midi_freq_unit,
    )


def build_midi_defines(midi_mapping: MidiMapping | None) -> str:
    r"""
    Build CMake compile definition lines for a MIDI mapping.

    Returns an empty string if MIDI is disabled, or newline+indent-separated
    definition strings like "MIDI_ENABLED=1\\n    MIDI_GATE_IDX=5".

    Shared by all CMake-based platforms (CLAP, VST3, etc.).
    """
    if midi_mapping is None or not midi_mapping.enabled:
        return ""

    defs = ["MIDI_ENABLED=1"]
    if midi_mapping.gate_idx is not None:
        defs.append(f"MIDI_GATE_IDX={midi_mapping.gate_idx}")
    if midi_mapping.freq_idx is not None:
        defs.append(f"MIDI_FREQ_IDX={midi_mapping.freq_idx}")
        freq_hz = 1 if midi_mapping.freq_unit == "hz" else 0
        defs.append(f"MIDI_FREQ_UNIT_HZ={freq_hz}")
    if midi_mapping.vel_idx is not None:
        defs.append(f"MIDI_VEL_IDX={midi_mapping.vel_idx}")
    if midi_mapping.num_voices > 1:
        defs.append(f"NUM_VOICES={midi_mapping.num_voices}")
    return "\n    ".join(defs)


def _find_param_index(param_by_name: dict[str, int], names: set[str]) -> int | None:
    """Find the first matching param index from a set of candidate names."""
    for name in names:
        if name in param_by_name:
            return param_by_name[name]
    return None
