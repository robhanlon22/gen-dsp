"""
VST3 plugin platform implementation.

Generates cross-platform VST3 plugins (.vst3 bundles) using CMake and
the Steinberg VST3 SDK. The SDK is fetched at configure time via CMake
FetchContent -- no vendoring required.
"""

import hashlib
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from string import Template

from gen_dsp.core.manifest import Manifest, build_remap_defines
from gen_dsp.core.midi import build_midi_defines
from gen_dsp.core.project import ProjectConfig
from gen_dsp.errors import ProjectError
from gen_dsp.platforms.cmake_platform import CMakePlatform
from gen_dsp.templates import get_vst3_templates_dir


@dataclass(frozen=True)
class Vst3CmakeContext:
    """Inputs for rendering the VST3 CMake template."""

    gen_name: str
    lib_name: str
    num_inputs: int
    num_outputs: int
    fuid: tuple[int, int, int, int]
    use_shared_cache: str
    cache_dir: str
    midi_defines: str
    remap_defines: str


class Vst3Platform(CMakePlatform):
    """VST3 plugin platform implementation using CMake."""

    name = "vst3"

    @property
    def extension(self) -> str:
        """Get the file extension for VST3 plugins."""
        return ".vst3"

    def generate_project(
        self,
        manifest: Manifest,
        output_dir: Path,
        lib_name: str,
        config: ProjectConfig | None = None,
    ) -> None:
        """Generate VST3 project files."""
        templates_dir = get_vst3_templates_dir()
        if not templates_dir.is_dir():
            msg = f"VST3 templates not found at {templates_dir}"
            raise ProjectError(msg)

        # Copy static files
        static_files = [
            "gen_ext_vst3.cpp",
            "gen_ext_common_vst3.h",
            "_ext_vst3.cpp",
            "vst3_buffer.h",
        ]

        for filename in static_files:
            src = templates_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

        self.generate_ext_header(output_dir, "vst3")
        self.copy_remap_header(output_dir)
        self.copy_voice_alloc_header(output_dir, config)

        # Generate FUID from lib_name
        fuid = self._generate_fuid(lib_name)

        # Resolve shared cache settings
        use_shared_cache, cache_dir = self.resolve_shared_cache(config)

        # Build MIDI compile definitions
        midi_mapping = config.midi_mapping if config else None
        midi_defines = build_midi_defines(midi_mapping)
        remap_defines = build_remap_defines(manifest)

        # Generate CMakeLists.txt
        cmake_context = Vst3CmakeContext(
            gen_name=manifest.gen_name,
            lib_name=lib_name,
            num_inputs=manifest.num_inputs,
            num_outputs=manifest.num_outputs,
            fuid=fuid,
            use_shared_cache=use_shared_cache,
            cache_dir=cache_dir,
            midi_defines=midi_defines,
            remap_defines=remap_defines,
        )
        self._generate_cmakelists(
            templates_dir / "CMakeLists.txt.template",
            output_dir / "CMakeLists.txt",
            cmake_context,
        )

        # Generate gen_buffer.h using base class method
        self.generate_buffer_header(
            templates_dir / "gen_buffer.h.template",
            output_dir / "gen_buffer.h",
            manifest.buffers,
            header_comment="Buffer configuration for gen_dsp VST3 wrapper",
        )

        # Create build directory
        (output_dir / "build").mkdir(exist_ok=True)

    def _generate_fuid(self, lib_name: str) -> tuple[int, int, int, int]:
        """
        Generate a deterministic 128-bit FUID from the library name.

        Uses MD5 of 'com.gen-dsp.vst3.<lib_name>' split into 4 x uint32.
        Returns tuple of 4 integers.
        """
        digest = hashlib.blake2b(
            f"com.gen-dsp.vst3.{lib_name}".encode(),
            digest_size=16,
        ).digest()
        return struct.unpack(">IIII", digest)

    def _generate_cmakelists(
        self,
        template_path: Path,
        output_path: Path,
        context: Vst3CmakeContext,
    ) -> None:
        """Generate CMakeLists.txt from template."""
        if not template_path.exists():
            msg = f"CMakeLists.txt template not found at {template_path}"
            raise ProjectError(msg)

        template_content = template_path.read_text(encoding="utf-8")
        template = Template(template_content)
        content = template.safe_substitute(
            gen_name=context.gen_name,
            lib_name=context.lib_name,
            genext_version=self.GENEXT_VERSION,
            num_inputs=context.num_inputs,
            num_outputs=context.num_outputs,
            fuid_0=f"0x{context.fuid[0]:08X}",
            fuid_1=f"0x{context.fuid[1]:08X}",
            fuid_2=f"0x{context.fuid[2]:08X}",
            fuid_3=f"0x{context.fuid[3]:08X}",
            use_shared_cache=context.use_shared_cache,
            cache_dir=context.cache_dir,
            midi_defines=context.midi_defines,
            remap_defines=context.remap_defines,
        )
        output_path.write_text(content, encoding="utf-8")

    def find_output(self, project_dir: Path) -> Path | None:
        """Find the built VST3 plugin bundle."""
        build_dir = project_dir / "build"
        if build_dir.is_dir():
            for f in build_dir.glob("**/*.vst3"):
                if f.is_dir():
                    return f
        return None
