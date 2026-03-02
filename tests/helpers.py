"""Shared test validation helpers for plugin build verification.

These functions validate built plugins by running external validator tools
(clap-validator, VST3 SDK validator, lilv-based LV2 validator) or headless
host applications (pd, chuck) against the built output.

All validators are optional -- if the tool is unavailable, validation is
silently skipped (the function returns immediately).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CLAP validation
# ---------------------------------------------------------------------------


def validate_clap(validator: Optional[Path], clap_bundle: Path) -> None:
    """Run the CLAP validator against a plugin, if available."""
    if validator is None:
        return
    result = subprocess.run(
        [str(validator), "validate", str(clap_bundle)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "0 failed" in result.stdout, (
        f"CLAP validation failed:\n{result.stdout}\n{result.stderr}"
    )
    assert result.returncode == 0, (
        f"CLAP validation failed:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# VST3 validation
# ---------------------------------------------------------------------------


def validate_vst3(
    validator: Optional[Path],
    vst3_bundle: Path,
    allow_crash_on_cleanup: bool = False,
) -> None:
    """Run the VST3 SDK validator against a bundle, if available."""
    if validator is None:
        return
    result = subprocess.run(
        [str(validator), str(vst3_bundle)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if allow_crash_on_cleanup:
        assert "[Failed]" not in result.stdout, (
            f"VST3 validation failed:\n{result.stdout}\n{result.stderr}"
        )
    else:
        assert result.returncode == 0, (
            f"VST3 validation failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "[Failed]" not in result.stdout, (
            f"VST3 validation failed:\n{result.stdout}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# LV2 validation
# ---------------------------------------------------------------------------


def validate_lv2(
    validator: Optional[Path],
    bundle_dir: Path,
    lib_name: str,
    expected_audio_in: int,
    expected_audio_out: int,
    expected_params: int,
) -> None:
    """Validate a built LV2 bundle by instantiating and processing audio."""
    if validator is None:
        return

    plugin_uri = f"http://gen-dsp.com/plugins/{lib_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        isolated = Path(tmpdir) / bundle_dir.name
        shutil.copytree(bundle_dir, isolated)

        result = subprocess.run(
            [
                str(validator),
                tmpdir,
                plugin_uri,
                str(expected_audio_in),
                str(expected_audio_out),
                str(expected_params),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"LV2 validation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout, (
            f"LV2 validation did not PASS:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# PD validation
# ---------------------------------------------------------------------------

_has_pd = shutil.which("pd") is not None


def validate_pd_external(project_dir: Path, lib_name: str) -> None:
    """Load a built PD external in headless PD and verify it instantiates."""
    if not _has_pd:
        return

    test_pd = project_dir / "test_load.pd"
    test_pd.write_text(
        "#N canvas 0 0 450 300 10;\n"
        f"#X obj 10 10 {lib_name}~;\n"
        "#X obj 10 50 loadbang;\n"
        "#X msg 10 70 \\; pd quit;\n"
        "#X connect 1 0 2 0;\n"
    )

    result = subprocess.run(
        [
            "pd",
            "-nogui",
            "-noaudio",
            "-noadc",
            "-nodac",
            "-stderr",
            "-verbose",
            "-path",
            str(project_dir),
            str(test_pd),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"pd failed (exit {result.returncode}):\n{output}"
    assert "couldn't create" not in output, (
        f"PD failed to load {lib_name}~ external:\n{output}"
    )
    assert f"{lib_name}~" in output


# ---------------------------------------------------------------------------
# ChucK validation
# ---------------------------------------------------------------------------

_has_chuck = shutil.which("chuck") is not None


def validate_chugin(
    project_dir: Path,
    class_name: str,
    expected_params: int,
    expect_audio: bool = False,
) -> None:
    """Load a built chugin in ChucK and validate it works."""
    if not _has_chuck:
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

    result = subprocess.run(
        ["chuck", "--chugin-path:.", "--silent", "test.ck"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"chuck failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    output = result.stderr
    assert "PARAMS" in output
    assert str(expected_params) in output
    if expect_audio:
        assert "AUDIO_OK" in output, f"No audio output detected:\n{output}"
    assert "DONE" in output
