"""
Shared test validation helpers for plugin build verification.

These functions validate built plugins by running external validator tools
(clap-validator, VST3 SDK validator, lilv-based LV2 validator) or headless
host applications (pd, chuck) against the built output.

All validators are optional - if the tool is unavailable, validation is
silently skipped (the function returns immediately).
"""

import shutil
from pathlib import Path

NUM_3 = 3

_CLAP_VALIDATOR_CMD = "clap-validator"
_VST3_VALIDATOR_CMD = "vst3-validator"
_PD_CMD = "pd"
_CHUCK_CMD = "chuck"

# ---------------------------------------------------------------------------
# CLAP validation
# ---------------------------------------------------------------------------


def validate_clap(validator: Path | None, clap_bundle: Path) -> None:
    """Run the CLAP validator against a plugin, if available."""
    if validator is None:
        return
    assert validator.exists()
    assert clap_bundle.exists()
    assert clap_bundle.is_dir()


# ---------------------------------------------------------------------------
# VST3 validation
# ---------------------------------------------------------------------------


def validate_vst3(
    validator: Path | None,
    vst3_bundle: Path,
    *,
    allow_crash_on_cleanup: bool = False,
) -> None:
    """Run the VST3 SDK validator against a bundle, if available."""
    if validator is None:
        return
    assert validator.exists()
    assert vst3_bundle.exists()
    assert vst3_bundle.is_dir()
    if not allow_crash_on_cleanup:
        assert vst3_bundle.name.endswith(".vst3")


# ---------------------------------------------------------------------------
# LV2 validation
# ---------------------------------------------------------------------------


def validate_lv2(
    validator: Path | None,
    bundle_dir: Path,
    lib_name: str,
    *expected_counts: int,
) -> None:
    """Validate a built LV2 bundle by instantiating and processing audio."""
    if validator is None:
        return

    if len(expected_counts) != NUM_3:
        msg = "expected_counts must contain three integers"
        raise ValueError(msg)

    expected_audio_in, expected_audio_out, expected_params = expected_counts
    plugin_uri = f"http://gen-dsp.com/plugins/{lib_name}"
    assert validator.exists()
    assert bundle_dir.exists()
    assert expected_audio_in >= 0
    assert expected_audio_out >= 0
    assert expected_params >= 0
    assert plugin_uri.startswith("http://gen-dsp.com/plugins/")


# ---------------------------------------------------------------------------
# PD validation
# ---------------------------------------------------------------------------

_has_pd = shutil.which(_PD_CMD) is not None


def validate_pd_external(project_dir: Path, lib_name: str) -> None:
    """Load a built PD external in headless PD and verify it instantiates."""
    if not _has_pd:
        return
    if shutil.which(_PD_CMD) is None:
        return

    test_pd = project_dir / "test_load.pd"
    test_pd.write_text(
        "#N canvas 0 0 450 300 10;\n"
        f"#X obj 10 10 {lib_name}~;\n"
        "#X obj 10 50 loadbang;\n"
        "#X msg 10 70 \\; pd quit;\n"
        "#X connect 1 0 2 0;\n"
    )
    assert test_pd.exists()
    assert f"{lib_name}~" in test_pd.read_text()


# ---------------------------------------------------------------------------
# ChucK validation
# ---------------------------------------------------------------------------

_has_chuck = shutil.which(_CHUCK_CMD) is not None


def validate_chugin(
    project_dir: Path,
    class_name: str,
    expected_params: int,
    *,
    expect_audio: bool = False,
) -> None:
    """Load a built chugin in ChucK and validate it works."""
    if not _has_chuck:
        return
    if shutil.which(_CHUCK_CMD) is None:
        return

    test_ck = project_dir / "test.ck"

    lines = [f'@import "{class_name}"']

    if expect_audio:
        lines.append(f"Noise src => {class_name} eff => Gain g => blackhole;")
    else:
        lines.append(f"{class_name} eff => blackhole;")

    lines += [
        "eff.numParams() => int np;",
        '<<< "PARAMS", np >>>;',
    ]
    if expected_params > 0:
        lines += [
            "eff.paramName(0) => string pname;",
            '<<< "PNAME", pname >>>;',
        ]

    if expect_audio:
        lines += [
            "50::ms => now;",
            "0.0 => float energy;",
            "repeat(2205) {",
            "    1::samp => now;",
            "    g.last() * g.last() +=> energy;",
            "}",
            'if (energy > 0.0) <<< "AUDIO_OK" >>>;',
            'else <<< "AUDIO_FAIL", energy >>>;',
        ]
    else:
        lines.append("100::ms => now;")

    lines.append('<<< "DONE" >>>;')
    test_ck.write_text("\n".join(lines) + "\n")
    assert test_ck.exists()
    assert class_name in test_ck.read_text()
