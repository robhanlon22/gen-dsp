"""
LV2 plugin platform implementation.

Generates cross-platform LV2 plugins (.lv2 bundle directories) using CMake
and the LV2 C API (header-only, ISC licensed). LV2 headers are fetched
at configure time via CMake FetchContent.

LV2 bundles contain:
  - manifest.ttl  (plugin discovery metadata)
  - <name>.ttl    (port definitions: control + audio)
  - <name>.so/.dylib (shared library)
"""

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import ClassVar

from gen_dsp.core.manifest import Manifest, ParamInfo, build_remap_defines
from gen_dsp.core.midi import build_midi_defines
from gen_dsp.core.project import ProjectConfig
from gen_dsp.errors import ProjectError
from gen_dsp.platforms.base import PluginCategory
from gen_dsp.platforms.cmake_platform import CMakePlatform
from gen_dsp.templates import get_lv2_templates_dir


@dataclass(frozen=True)
class Lv2PluginTtlContext:
    """Inputs for rendering the LV2 plugin TTL template."""

    lib_name: str
    plugin_uri: str
    num_inputs: int
    num_outputs: int
    num_params: int
    params: list[ParamInfo]


@dataclass(frozen=True)
class Lv2CmakeContext:
    """Inputs for rendering the LV2 CMake template."""

    gen_name: str
    lib_name: str
    num_inputs: int
    num_outputs: int
    num_params: int
    use_shared_cache: str
    cache_dir: str
    midi_defines: str
    remap_defines: str


@dataclass(frozen=True)
class Lv2PortSpec:
    """Descriptor for a family of LV2 audio ports."""

    direction: str
    port_kind: str
    symbol_prefix: str
    name_prefix: str


class Lv2Platform(CMakePlatform):
    """LV2 plugin platform implementation using CMake."""

    name = "lv2"
    LV2_URI_BASE: ClassVar[str] = "http://gen-dsp.com/plugins"

    _LV2_TYPE_MAP: ClassVar[dict[PluginCategory, str]] = {
        PluginCategory.EFFECT: "lv2:Plugin ,\n      lv2:EffectPlugin",
        PluginCategory.GENERATOR: "lv2:Plugin ,\n      lv2:GeneratorPlugin",
    }

    LV2_TYPE_INSTRUMENT: ClassVar[str] = "lv2:Plugin ,\n      lv2:InstrumentPlugin"

    @property
    def extension(self) -> str:
        """Get the extension for LV2 bundles."""
        return ".lv2"

    def generate_project(
        self,
        manifest: Manifest,
        output_dir: Path,
        lib_name: str,
        config: ProjectConfig | None = None,
    ) -> None:
        """Generate LV2 project files."""
        templates_dir = get_lv2_templates_dir()
        if not templates_dir.is_dir():
            msg = f"LV2 templates not found at {templates_dir}"
            raise ProjectError(msg)

        # Copy static files
        static_files = [
            "gen_ext_lv2.cpp",
            "gen_ext_common_lv2.h",
            "_ext_lv2.cpp",
            "lv2_buffer.h",
        ]
        for filename in static_files:
            src = templates_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

        self.generate_ext_header(output_dir, "lv2")
        self.copy_remap_header(output_dir)
        self.copy_voice_alloc_header(output_dir, config)

        # Build MIDI compile definitions
        midi_mapping = config.midi_mapping if config else None
        midi_enabled = midi_mapping is not None and midi_mapping.enabled
        midi_defines = build_midi_defines(midi_mapping)

        # Generate TTL files
        plugin_uri = f"{self.LV2_URI_BASE}/{lib_name}"
        self._generate_manifest_ttl(output_dir, lib_name, plugin_uri)
        plugin_context = Lv2PluginTtlContext(
            lib_name=lib_name,
            plugin_uri=plugin_uri,
            num_inputs=manifest.num_inputs,
            num_outputs=manifest.num_outputs,
            num_params=manifest.num_params,
            params=manifest.params,
        )
        self._generate_plugin_ttl(
            output_dir,
            plugin_context,
            midi_enabled=midi_enabled,
        )

        # Resolve shared cache settings
        use_shared_cache, cache_dir = self.resolve_shared_cache(config)

        # Build input remap compile definitions
        remap_defines = build_remap_defines(manifest)

        # Generate CMakeLists.txt
        cmake_context = Lv2CmakeContext(
            gen_name=manifest.gen_name,
            lib_name=lib_name,
            num_inputs=manifest.num_inputs,
            num_outputs=manifest.num_outputs,
            num_params=manifest.num_params,
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
            header_comment="Buffer configuration for gen_dsp LV2 wrapper",
        )

        # Create build directory
        (output_dir / "build").mkdir(exist_ok=True)

    def _generate_manifest_ttl(
        self,
        output_dir: Path,
        lib_name: str,
        plugin_uri: str,
    ) -> None:
        """Generate manifest.ttl for LV2 plugin discovery."""
        if sys.platform == "darwin":
            binary_ext = "dylib"
        elif sys.platform == "win32":
            binary_ext = "dll"
        else:
            binary_ext = "so"

        content = (
            "@prefix lv2:  <http://lv2plug.in/ns/lv2core#> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "\n"
            f"<{plugin_uri}>\n"
            "    a lv2:Plugin ;\n"
            f"    lv2:binary <{lib_name}.{binary_ext}> ;\n"
            f"    rdfs:seeAlso <{lib_name}.ttl> .\n"
        )
        (output_dir / "manifest.ttl").write_text(content, encoding="utf-8")

    def _generate_plugin_ttl(
        self,
        output_dir: Path,
        context: Lv2PluginTtlContext,
        *,
        midi_enabled: bool = False,
    ) -> None:
        """
        Generate <plugin>.ttl with full port descriptions.

        Port layout matches the C++ code:
          indices 0..num_params-1   = ControlPort InputPort
          indices num_params..+nin  = AudioPort InputPort
          indices above..+nout      = AudioPort OutputPort
          (if MIDI) last index      = AtomPort InputPort (MIDI events)
        """
        # Plugin type
        plugin_type = (
            self.LV2_TYPE_INSTRUMENT
            if midi_enabled
            else self._LV2_TYPE_MAP[PluginCategory.from_num_inputs(context.num_inputs)]
        )

        lines = self._build_plugin_ttl_header(
            context.lib_name,
            context.plugin_uri,
            plugin_type,
            midi_enabled=midi_enabled,
        )
        port_index = 0
        total_ports = context.num_params + context.num_inputs + context.num_outputs
        if midi_enabled:
            total_ports += 1

        control_lines, port_index = self._build_control_port_lines(
            context.params,
            context.num_params,
            port_index,
            total_ports,
        )
        lines.extend(control_lines)

        audio_input_lines, port_index = self._build_audio_port_lines(
            Lv2PortSpec("InputPort", "AudioPort", "in", "Input"),
            context.num_inputs,
            port_index,
            total_ports,
        )
        lines.extend(audio_input_lines)

        audio_output_lines, port_index = self._build_audio_port_lines(
            Lv2PortSpec("OutputPort", "AudioPort", "out", "Output"),
            context.num_outputs,
            port_index,
            total_ports,
        )
        lines.extend(audio_output_lines)

        if midi_enabled:
            lines.extend(self._build_midi_port_lines(port_index))
            port_index += 1

        content = "\n".join(lines) + "\n"
        (output_dir / f"{context.lib_name}.ttl").write_text(content, encoding="utf-8")

    def _build_plugin_ttl_header(
        self,
        lib_name: str,
        plugin_uri: str,
        plugin_type: str,
        *,
        midi_enabled: bool,
    ) -> list[str]:
        """Build the common header lines for plugin.ttl."""
        prefixes = [
            "@prefix doap:  <http://usefulinc.com/ns/doap#> .",
            "@prefix lv2:   <http://lv2plug.in/ns/lv2core#> .",
            "@prefix state: <http://lv2plug.in/ns/ext/state#> .",
            "@prefix urid:  <http://lv2plug.in/ns/ext/urid#> .",
        ]
        if midi_enabled:
            prefixes.extend(
                [
                    "@prefix atom: <http://lv2plug.in/ns/ext/atom#> .",
                    "@prefix midi: <http://lv2plug.in/ns/ext/midi#> .",
                ]
            )

        lines = [
            *prefixes,
            "",
            f"<{plugin_uri}>",
            f"    a {plugin_type} ;",
            f'    doap:name "{lib_name}" ;',
            "    doap:license <http://opensource.org/licenses/isc> ;",
            "    lv2:optionalFeature lv2:hardRTCapable ;",
            "    lv2:extensionData state:interface ;",
        ]
        if midi_enabled:
            lines.append("    lv2:requiredFeature urid:map ;")
        return lines

    def _build_control_port_lines(
        self,
        params: list[ParamInfo],
        num_params: int,
        port_index: int,
        total_ports: int,
    ) -> tuple[list[str], int]:
        """Build TTL lines for control input ports."""
        lines = []
        for i in range(num_params):
            if i < len(params):
                p = params[i]
                symbol = self._sanitize_symbol(p.name)
                pname = p.name
                pmin = p.min
                pmax = p.max
                pdefault = p.default
            else:
                symbol = f"param_{i}"
                pname = f"Parameter {i}"
                pmin = 0.0
                pmax = 1.0
                pdefault = 0.0

            terminator = " ." if port_index == total_ports - 1 else " ;"
            lines.extend(
                [
                    "    lv2:port [",
                    "        a lv2:InputPort , lv2:ControlPort ;",
                    f"        lv2:index {port_index} ;",
                    f'        lv2:symbol "{symbol}" ;',
                    f'        lv2:name "{pname}" ;',
                    f"        lv2:default {pdefault} ;",
                    f"        lv2:minimum {pmin} ;",
                    f"        lv2:maximum {pmax}",
                    f"    ]{terminator}",
                ]
            )
            port_index += 1
        return lines, port_index

    def _build_audio_port_lines(
        self,
        spec: Lv2PortSpec,
        count: int,
        port_index: int,
        total_ports: int,
    ) -> tuple[list[str], int]:
        """Build TTL lines for audio input or output ports."""
        lines = []
        for i in range(count):
            terminator = " ." if port_index == total_ports - 1 else " ;"
            lines.extend(
                [
                    "    lv2:port [",
                    f"        a lv2:{spec.direction} , lv2:{spec.port_kind} ;",
                    f"        lv2:index {port_index} ;",
                    f'        lv2:symbol "{spec.symbol_prefix}{i}" ;',
                    f'        lv2:name "{spec.name_prefix} {i}"',
                    f"    ]{terminator}",
                ]
            )
            port_index += 1
        return lines, port_index

    @staticmethod
    def _build_midi_port_lines(port_index: int) -> list[str]:
        """Build TTL lines for the MIDI atom input port."""
        return [
            "    lv2:port [",
            "        a lv2:InputPort , atom:AtomPort ;",
            f"        lv2:index {port_index} ;",
            '        lv2:symbol "midi_in" ;',
            '        lv2:name "MIDI In" ;',
            "        atom:bufferType atom:Sequence ;",
            "        atom:supports midi:MidiEvent",
            "    ] .",
        ]

    @staticmethod
    def _sanitize_symbol(name: str) -> str:
        """Ensure a parameter name is a valid LV2 symbol (C identifier)."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized or "param"

    def _generate_cmakelists(
        self,
        template_path: Path,
        output_path: Path,
        context: Lv2CmakeContext,
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
            num_params=context.num_params,
            use_shared_cache=context.use_shared_cache,
            cache_dir=context.cache_dir,
            midi_defines=context.midi_defines,
            remap_defines=context.remap_defines,
        )
        output_path.write_text(content, encoding="utf-8")

    def find_output(self, project_dir: Path) -> Path | None:
        """Find the built LV2 bundle directory."""
        build_dir = project_dir / "build"
        if build_dir.is_dir():
            for f in build_dir.glob("**/*.lv2"):
                if f.is_dir():
                    return f
        return None
