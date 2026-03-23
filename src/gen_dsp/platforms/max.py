"""
Max/MSP platform implementation.

Generates Max/MSP externals using CMake and the max-sdk-base submodule.
"""

import platform as sys_platform
import shutil
from pathlib import Path
from string import Template

from gen_dsp.core.builder import BuildResult
from gen_dsp.core.manifest import Manifest, build_remap_defines
from gen_dsp.core.project import ProjectConfig
from gen_dsp.errors import BuildError, ProjectError
from gen_dsp.platforms.cmake_platform import CMakePlatform
from gen_dsp.platforms.command import run_command as run_external_command
from gen_dsp.templates import get_max_templates_dir


class MaxPlatform(CMakePlatform):
    """Max/MSP platform implementation using CMake and max-sdk-base."""

    name = "max"

    # max-sdk-base git repository
    MAX_SDK_REPO = "https://github.com/Cycling74/max-sdk-base.git"

    @property
    def extension(self) -> str:
        """Get the file extension for the current OS."""
        system = sys_platform.system().lower()
        if system == "darwin":
            return ".mxo"
        if system == "windows":
            return ".mxe64"
        return ".mxl"

    def get_build_instructions(self) -> list[str]:
        """Get build instructions for Max/MSP."""
        return [
            "git clone --depth 1 https://github.com/Cycling74/max-sdk-base.git",
            "mkdir -p build && cd build && cmake .. && cmake --build .",
        ]

    def generate_project(
        self,
        manifest: Manifest,
        output_dir: Path,
        lib_name: str,
        config: ProjectConfig | None = None,
    ) -> None:
        """Generate Max/MSP project files."""
        if config is not None:
            _ = config

        templates_dir = get_max_templates_dir()
        if not templates_dir.is_dir():
            msg = f"Max/MSP templates not found at {templates_dir}"
            raise ProjectError(msg)

        # Copy static files
        static_files = [
            "gen_ext_max.cpp",
            "gen_ext_common_max.h",
            "_ext_max.cpp",
            "_ext_max.h",
            "max_buffer.h",
        ]

        for filename in static_files:
            src = templates_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

        # Build input remap compile definitions
        remap_defines = build_remap_defines(manifest)

        self.copy_remap_header(output_dir)

        # Generate CMakeLists.txt
        self._generate_cmakelists(
            templates_dir / "CMakeLists.txt.template",
            output_dir / "CMakeLists.txt",
            manifest.gen_name,
            lib_name,
            remap_defines=remap_defines,
        )

        # Generate gen_buffer.h using base class method
        self.generate_buffer_header(
            templates_dir / "gen_buffer.h.template",
            output_dir / "gen_buffer.h",
            manifest.buffers,
            header_comment="Buffer configuration for gen_dsp Max/MSP wrapper",
        )

        # Create build directory
        (output_dir / "build").mkdir(exist_ok=True)

        # Create externals output directory
        (output_dir / "externals").mkdir(exist_ok=True)

    def _generate_cmakelists(
        self,
        template_path: Path,
        output_path: Path,
        gen_name: str,
        lib_name: str,
        remap_defines: str = "",
    ) -> None:
        """Generate CMakeLists.txt from template."""
        if template_path.exists():
            template_content = template_path.read_text(encoding="utf-8")
            template = Template(template_content)
            content = template.safe_substitute(
                gen_name=gen_name,
                lib_name=lib_name,
                genext_version=self.GENEXT_VERSION,
                remap_defines=remap_defines,
            )
        else:
            msg = f"CMakeLists.txt template not found at {template_path}"
            raise ProjectError(msg)

        output_path.write_text(content, encoding="utf-8")

    def setup_sdk(self, project_dir: Path) -> bool:
        """
        Set up the max-sdk-base submodule.

        Returns True if SDK is ready, False if setup failed.
        """
        sdk_dir = project_dir / "max-sdk-base"

        if sdk_dir.exists() and (sdk_dir / "script" / "max-pretarget.cmake").exists():
            return True

        # Clone max-sdk-base
        result = run_external_command(
            ["git", "clone", "--depth", "1", self.MAX_SDK_REPO, str(sdk_dir)],
            cwd=project_dir,
        )
        return result.returncode == 0

    def build(
        self,
        project_dir: Path,
        *,
        clean: bool = False,
        verbose: bool = False,
    ) -> BuildResult:
        """Build Max/MSP external using CMake."""
        # Ensure max-sdk-base is available before building
        if not self.setup_sdk(project_dir):
            msg = (
                "Failed to set up max-sdk-base. Please ensure git is installed "
                "and run:\n"
                f"  cd {project_dir}\n"
                f"  git clone {self.MAX_SDK_REPO}"
            )
            raise BuildError(
                msg
            )
        return self._build_with_cmake(project_dir, clean, verbose)

    def find_output(self, project_dir: Path) -> Path | None:
        """Find the built Max external file."""
        # Check externals directory (where max-posttarget.cmake puts output)
        externals_dir = project_dir / "externals"
        if externals_dir.is_dir():
            # Look for .mxo bundles (macOS) or .mxe64 (Windows)
            for pattern in ["*.mxo", "*.mxe64", "*.mxl*"]:
                for f in externals_dir.glob(pattern):
                    return f

        # Also check build directory
        build_dir = project_dir / "build"
        if build_dir.is_dir():
            for pattern in ["**/*.mxo", "**/*.mxe64", "**/*.mxl*"]:
                for f in build_dir.glob(pattern):
                    return f

        return None
