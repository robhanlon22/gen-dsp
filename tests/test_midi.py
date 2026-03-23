"""Tests for MIDI-to-CV auto-detection and mapping."""

from gen_dsp.core.manifest import Manifest, ParamInfo
from gen_dsp.core.midi import MidiMapping, build_midi_defines, detect_midi_mapping

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2


def _make_manifest(num_inputs: int, params: list[ParamInfo]) -> Manifest:
    """Helper to create a minimal Manifest with given I/O and params."""
    return Manifest(
        gen_name="test_gen",
        num_inputs=num_inputs,
        num_outputs=2,
        params=params,
    )


def _param(index: int, name: str) -> ParamInfo:
    """Helper to create a minimal ParamInfo."""
    return ParamInfo(
        index=index, name=name, has_minmax=True, min=0.0, max=1.0, default=0.0
    )


class TestMidiAutoDetection:
    """Test auto-detection logic for MIDI parameter mapping."""

    def test_effect_disabled(self) -> object:
        """Test test effect disabled."""
        manifest = _make_manifest(2, [_param(0, "gate"), _param(1, "freq")])
        result = detect_midi_mapping(manifest)
        assert not result.enabled

    def test_generator_with_gate_freq_vel(self) -> object:
        """Test test generator with gate freq vel."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "freq"),
                _param(1, "gate"),
                _param(2, "vel"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.enabled
        assert result.gate_idx == NUM_1
        assert result.freq_idx == NUM_0
        assert result.vel_idx == NUM_2

    def test_generator_gate_only(self) -> object:
        """Test test generator gate only."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "cutoff"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.enabled
        assert result.gate_idx == NUM_0
        assert result.freq_idx is None
        assert result.vel_idx is None

    def test_generator_freq_no_gate_disabled(self) -> object:
        """Test test generator freq no gate disabled."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "freq"),
                _param(1, "cutoff"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert not result.enabled

    def test_no_params_disabled(self) -> object:
        """Test test no params disabled."""
        manifest = _make_manifest(0, [])
        result = detect_midi_mapping(manifest)
        assert not result.enabled

    def test_no_midi_flag_overrides(self) -> object:
        """Test test no midi flag overrides."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "freq"),
            ],
        )
        result = detect_midi_mapping(manifest, no_midi=True)
        assert not result.enabled

    def test_frequency_alias(self) -> object:
        """Test test frequency alias."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "frequency"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.enabled
        assert result.freq_idx == NUM_1

    def test_pitch_alias(self) -> object:
        """Test test pitch alias."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "pitch"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.enabled
        assert result.freq_idx == NUM_1

    def test_velocity_alias(self) -> object:
        """Test test velocity alias."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "velocity"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.enabled
        assert result.vel_idx == NUM_1

    def test_case_sensitive(self) -> object:
        """Test test case sensitive."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "Gate"),
                _param(1, "FREQ"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert not result.enabled


class TestMidiExplicitOverrides:
    """Test explicit --midi-* CLI flag overrides."""

    def test_explicit_gate_overrides_name(self) -> object:
        """Test test explicit gate overrides name."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "trig"),
                _param(1, "note"),
            ],
        )
        result = detect_midi_mapping(manifest, midi_gate="trig")
        assert result.enabled
        assert result.gate_idx == NUM_0
        assert result.freq_idx is None

    def test_explicit_freq_overrides_name(self) -> object:
        """Test test explicit freq overrides name."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "note"),
            ],
        )
        result = detect_midi_mapping(manifest, midi_freq="note")
        assert result.enabled
        assert (
            result.gate_idx is None
        )  # no explicit gate, no auto-detect in explicit mode
        assert result.freq_idx == NUM_1

    def test_explicit_all_three(self) -> object:
        """Test test explicit all three."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "trig"),
                _param(1, "note"),
                _param(2, "amp"),
            ],
        )
        result = detect_midi_mapping(
            manifest, midi_gate="trig", midi_freq="note", midi_vel="amp"
        )
        assert result.enabled
        assert result.gate_idx == NUM_0
        assert result.freq_idx == NUM_1
        assert result.vel_idx == NUM_2

    def test_explicit_nonexistent_param(self) -> object:
        """Test test explicit nonexistent param."""
        manifest = _make_manifest(0, [_param(0, "gate")])
        result = detect_midi_mapping(manifest, midi_gate="nonexistent")
        assert result.enabled  # explicit flag implies enabled
        assert result.gate_idx is None

    def test_explicit_overrides_on_effect(self) -> object:
        """Explicit --midi-* flags enable MIDI even for effects (num_inputs > 0).

        The user knows their patch topology -- if they explicitly request MIDI
        on an effect, the num_inputs guard should not silently disable it.
        See https://github.com/shakfu/gen-dsp/issues/2
        """
        manifest = _make_manifest(2, [_param(0, "gate")])
        result = detect_midi_mapping(manifest, midi_gate="gate")
        assert result.enabled
        assert result.gate_idx == NUM_0

    def test_freq_unit_midi(self) -> object:
        """Test test freq unit midi."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "freq"),
            ],
        )
        result = detect_midi_mapping(manifest, midi_freq_unit="midi")
        assert result.enabled
        assert result.freq_unit == "midi"

    def test_freq_unit_default_hz(self) -> object:
        """Test test freq unit default hz."""
        manifest = _make_manifest(
            0,
            [
                _param(0, "gate"),
                _param(1, "freq"),
            ],
        )
        result = detect_midi_mapping(manifest)
        assert result.freq_unit == "hz"


class TestMidiPolyphony:
    """Test polyphony (NUM_VOICES) in MIDI mapping and defines."""

    def test_num_voices_default(self) -> object:
        """Test test num voices default."""
        mapping = MidiMapping(enabled=True, gate_idx=0)
        assert mapping.num_voices == NUM_1

    def test_build_midi_defines_voices(self) -> object:
        """Test test build midi defines voices."""
        mapping = MidiMapping(enabled=True, gate_idx=0, num_voices=8)
        defines = build_midi_defines(mapping)
        assert "NUM_VOICES=8" in defines

    def test_build_midi_defines_mono_no_voices_define(self) -> object:
        """Test test build midi defines mono no voices define."""
        mapping = MidiMapping(enabled=True, gate_idx=0, num_voices=1)
        defines = build_midi_defines(mapping)
        assert "NUM_VOICES" not in defines

    def test_build_midi_defines_disabled_no_voices(self) -> object:
        """Test test build midi defines disabled no voices."""
        mapping = MidiMapping(enabled=False, num_voices=8)
        defines = build_midi_defines(mapping)
        assert defines == ""

    def test_voices_with_full_mapping(self) -> object:
        """Test test voices with full mapping."""
        mapping = MidiMapping(
            enabled=True, gate_idx=0, freq_idx=1, vel_idx=2, num_voices=4
        )
        defines = build_midi_defines(mapping)
        assert "MIDI_ENABLED=1" in defines
        assert "MIDI_GATE_IDX=0" in defines
        assert "MIDI_FREQ_IDX=1" in defines
        assert "MIDI_VEL_IDX=2" in defines
        assert "NUM_VOICES=4" in defines
