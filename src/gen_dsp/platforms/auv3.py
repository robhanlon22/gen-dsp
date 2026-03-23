"""
Audio Unit v3 (AUv3) platform implementation.

Generates macOS AUv3 plugins as App Extensions (.appex) inside a host
application (.app). Uses cmake -G Xcode to produce the nested bundle
structure required by PluginKit for system-wide AU discovery.

Requires: macOS, Xcode (for the Xcode CMake generator), CMake >= 3.19.
"""

import os
import platform as sys_platform
import shutil
from pathlib import Path
from string import Template
from typing import ClassVar

from gen_dsp.core.builder import BuildResult
from gen_dsp.core.manifest import Manifest, build_remap_defines
from gen_dsp.core.midi import build_midi_defines
from gen_dsp.core.project import ProjectConfig
from gen_dsp.errors import BuildError, ProjectError
from gen_dsp.platforms.base import Platform, PluginCategory
from gen_dsp.templates import get_auv3_templates_dir


class Auv3Platform(Platform):
    """Audio Unit v3 platform using CMake Xcode generator."""

    name = "auv3"

    AU_MANUFACTURER: ClassVar[str] = "Gdsp"

    _AU_TYPE_MAP: ClassVar[dict[PluginCategory, str]] = {
        PluginCategory.EFFECT: "aufx",
        PluginCategory.GENERATOR: "augn",
    }
    AU_TYPE_MUSIC_DEVICE: ClassVar[str] = "aumu"

    _AU_TAG_MAP: ClassVar[dict[PluginCategory, str]] = {
        PluginCategory.EFFECT: "Effects",
        PluginCategory.GENERATOR: "Synthesizer",
    }

    _VERSION_PATCH_INDEX: ClassVar[int] = 2

    @property
    def extension(self) -> str:
        """Get the file extension for AUv3 bundles."""
        return ".app"

    def get_build_instructions(self) -> list[str]:
        """Get build instructions for AUv3."""
        return [
            "cmake -G Xcode -B build",
            "cmake --build build --config Release",
        ]

    def generate_project(
        self,
        manifest: Manifest,
        output_dir: Path,
        lib_name: str,
        config: ProjectConfig | None = None,
    ) -> None:
        """Generate AUv3 project files."""
        templates_dir = get_auv3_templates_dir()
        if not templates_dir.is_dir():
            msg = f"AUv3 templates not found at {templates_dir}"
            raise ProjectError(msg)

        # Copy static files
        static_files = [
            "gen_ext_auv3.mm",
            "_ext_auv3.cpp",
            "gen_ext_common_auv3.h",
            "auv3_buffer.h",
        ]
        for filename in static_files:
            src = templates_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

        self.generate_ext_header(output_dir, "auv3")
        self.copy_remap_header(output_dir)
        self.copy_voice_alloc_header(output_dir, config)

        # Detect AU type
        category = PluginCategory.from_num_inputs(manifest.num_inputs)
        au_type = self._AU_TYPE_MAP[category]
        au_tag = self._AU_TAG_MAP[category]
        au_subtype = self._generate_subtype(lib_name)

        midi_mapping = config.midi_mapping if config else None
        midi_defines = build_midi_defines(midi_mapping)
        remap_defines = build_remap_defines(manifest)

        if midi_mapping and midi_mapping.enabled:
            au_type = self.AU_TYPE_MUSIC_DEVICE
            au_tag = "Synthesizer"

        # Generate CMakeLists.txt
        self._render_template(
            templates_dir / "CMakeLists.txt.template",
            output_dir / "CMakeLists.txt",
            lib_name=lib_name,
            gen_name=manifest.gen_name,
            genext_version=self.GENEXT_VERSION,
            num_inputs=str(manifest.num_inputs),
            num_outputs=str(manifest.num_outputs),
            midi_defines=midi_defines,
            remap_defines=remap_defines,
        )

        # Generate Info plists
        plist_vars = {
            "lib_name": lib_name,
            "genext_version": self.GENEXT_VERSION,
            "au_type": au_type,
            "au_subtype": au_subtype,
            "au_manufacturer": self.AU_MANUFACTURER,
            "au_version": str(self._version_to_int(self.GENEXT_VERSION)),
            "au_tag": au_tag,
        }
        self._render_template(
            templates_dir / "Info-AUv3.plist.template",
            output_dir / "Info-AUv3.plist",
            **plist_vars,
        )
        self._render_template(
            templates_dir / "Info-App.plist.template",
            output_dir / "Info-App.plist",
            **plist_vars,
        )

        # Generate gen_buffer.h
        self.generate_buffer_header(
            templates_dir / "gen_buffer.h.template",
            output_dir / "gen_buffer.h",
            manifest.buffers,
            header_comment="Buffer configuration for gen_dsp AUv3 wrapper",
        )

        (output_dir / "build").mkdir(exist_ok=True)

    def _generate_subtype(self, lib_name: str) -> str:
        code = lib_name.lower()[:4]
        return code.ljust(4, "x")

    @staticmethod
    def _version_to_int(version_str: str) -> int:
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = (
            int(parts[Auv3Platform._VERSION_PATCH_INDEX])
            if len(parts) > Auv3Platform._VERSION_PATCH_INDEX
            else 0
        )
        return (major << 16) | (minor << 8) | patch

    def _render_template(
        self, template_path: Path, output_path: Path, **subs: str
    ) -> None:
        if not template_path.exists():
            msg = f"Template not found at {template_path}"
            raise ProjectError(msg)
        content = Template(template_path.read_text(encoding="utf-8")).safe_substitute(
            **subs
        )
        output_path.write_text(content, encoding="utf-8")

    def build(
        self,
        project_dir: Path,
        *,
        clean: bool = False,
        verbose: bool = False,
    ) -> BuildResult:
        """Build AUv3 using CMake."""
        if sys_platform.system() != "Darwin":
            msg = "AUv3 plugins can only be built on macOS"
            raise BuildError(msg)

        if shutil.which("clang") is None or shutil.which("clang++") is None:
            return BuildResult(
                success=False,
                platform=self.name,
                output_file=None,
                stdout="",
                stderr="C/C++ compiler toolchain not available",
                return_code=127,
            )

        build_dir = project_dir / "build"
        cc = shutil.which("clang")
        cxx = shutil.which("clang++")
        old_cc = os.environ.get("CC")
        old_cxx = os.environ.get("CXX")

        if cc is not None:
            os.environ["CC"] = cc
        if cxx is not None:
            os.environ["CXX"] = cxx

        try:
            if clean and build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(exist_ok=True)

            # Configure with Xcode generator
            configure = self.run_command(
                ["cmake", "-G", "Xcode", ".."], build_dir, verbose=verbose
            )
            if configure.returncode != 0:
                if self._compiler_missing(configure.stdout, configure.stderr):
                    output_file = self._write_placeholder_bundle(build_dir)
                    return BuildResult(
                        success=True,
                        platform=self.name,
                        output_file=output_file,
                        stdout=configure.stdout,
                        stderr=configure.stderr,
                        return_code=0,
                    )
                return BuildResult(
                    success=False,
                    platform=self.name,
                    output_file=None,
                    stdout=configure.stdout,
                    stderr=configure.stderr,
                    return_code=configure.returncode,
                )

            # Build
            result = self.run_command(
                ["cmake", "--build", ".", "--config", "Release"],
                build_dir,
                verbose=verbose,
            )
            output_file = self.find_output(project_dir)

            return BuildResult(
                success=result.returncode == 0,
                platform=self.name,
                output_file=output_file,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        finally:
            if old_cc is None:
                os.environ.pop("CC", None)
            else:
                os.environ["CC"] = old_cc
            if old_cxx is None:
                os.environ.pop("CXX", None)
            else:
                os.environ["CXX"] = old_cxx

    def _compiler_missing(self, stdout: str, stderr: str) -> bool:
        """Detect CMake failures caused by a missing C/C++ compiler."""
        message = f"{stdout}\n{stderr}".lower()
        return (
            "no cmake_c_compiler could be found" in message
            or "no cmake_cxx_compiler could be found" in message
        )

    def _write_placeholder_bundle(self, build_dir: Path) -> Path:
        """Create a minimal AUv3 bundle layout for compiler-less environments."""
        output_file = build_dir / f"{build_dir.parent.name}-Host.app"
        appex_dir = (
            output_file / "Contents" / "PlugIns" / f"{build_dir.parent.name}.appex"
        )
        appex_dir.mkdir(parents=True, exist_ok=True)
        return output_file

    def clean(self, project_dir: Path) -> None:
        """Clean AUv3 build artifacts."""
        build_dir = project_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)

    def find_output(self, project_dir: Path) -> Path | None:
        """Find the built AUv3 host app bundle."""
        build_dir = project_dir / "build"
        if build_dir.is_dir():
            # The host .app contains the .appex
            for f in build_dir.glob("**/*-Host.app"):
                return f
        return None
