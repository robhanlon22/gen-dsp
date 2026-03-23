"""
Template access utilities for gen_dsp.

Templates are bundled with the package and accessed via these utilities.
"""

from pathlib import Path

_TEMPLATES_ROOT = Path(__file__).parent


def get_templates_dir(platform: str = "") -> Path:
    """
    Get the path to a platform's templates directory.

    Args:
        platform: Platform subdirectory name (e.g. "pd", "clap").
                  If empty, returns the root templates directory.

    Returns:
        Path to the templates directory.

    """
    if platform:
        return _TEMPLATES_ROOT / platform
    return _TEMPLATES_ROOT


# Backward-compatible aliases (one per platform)
def get_pd_templates_dir() -> Path:
    """Return the bundled Pure Data template directory."""
    return get_templates_dir("pd")


def get_max_templates_dir() -> Path:
    """Return the bundled Max template directory."""
    return get_templates_dir("max")


def get_chuck_templates_dir() -> Path:
    """Return the bundled ChucK template directory."""
    return get_templates_dir("chuck")


def get_au_templates_dir() -> Path:
    """Return the bundled Audio Unit template directory."""
    return get_templates_dir("au")


def get_clap_templates_dir() -> Path:
    """Return the bundled CLAP template directory."""
    return get_templates_dir("clap")


def get_vst3_templates_dir() -> Path:
    """Return the bundled VST3 template directory."""
    return get_templates_dir("vst3")


def get_lv2_templates_dir() -> Path:
    """Return the bundled LV2 template directory."""
    return get_templates_dir("lv2")


def get_sc_templates_dir() -> Path:
    """Return the bundled SuperCollider template directory."""
    return get_templates_dir("sc")


def get_vcvrack_templates_dir() -> Path:
    """Return the bundled VCV Rack template directory."""
    return get_templates_dir("vcvrack")


def get_daisy_templates_dir() -> Path:
    """Return the bundled Daisy template directory."""
    return get_templates_dir("daisy")


def get_circle_templates_dir() -> Path:
    """Return the bundled Circle template directory."""
    return get_templates_dir("circle")


def get_webaudio_templates_dir() -> Path:
    """Return the bundled Web Audio template directory."""
    return get_templates_dir("webaudio")


def get_standalone_templates_dir() -> Path:
    """Return the bundled standalone template directory."""
    return get_templates_dir("standalone")


def get_csound_templates_dir() -> Path:
    """Return the bundled Csound template directory."""
    return get_templates_dir("csound")


def get_auv3_templates_dir() -> Path:
    """Return the bundled AUv3 template directory."""
    return get_templates_dir("auv3")
