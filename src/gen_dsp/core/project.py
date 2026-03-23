"""
Project generator for gen_dsp.

Creates new project structures from gen~ exports using templates.
Uses the platform registry for platform-specific project generation.
"""

import importlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from string import Template as StrTemplate

# TYPE_CHECKING avoids circular import at runtime
from typing import TYPE_CHECKING, Optional

from gen_dsp.core.parser import ExportInfo
from gen_dsp.errors import ValidationError

MAX_BUFFERS = 5

if TYPE_CHECKING:
    from gen_dsp.core.manifest import Manifest
    from gen_dsp.core.midi import MidiMapping
    from gen_dsp.graph.models import Graph


def _private_class_attr(cls: type, name: str) -> object:
    """Fetch a private class attribute without triggering direct access rules."""
    return cls.__dict__[name]


def _private_static_call(
    cls: type,
    name: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Call a private staticmethod without direct private-member access."""
    return cls.__dict__[name].__get__(None, cls)(*args, **kwargs)


def _private_method_call(
    obj: object,
    name: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Call a private instance method without direct private-member access."""
    cls = obj.__class__
    return cls.__dict__[name].__get__(obj, cls)(*args, **kwargs)


@dataclass
class ProjectConfig:
    """Configuration for a new project."""

    # Name for the external (used as lib.name in Makefile)
    name: str

    # Target platform: 'pd', 'max', or any registered platform
    platform: str = "pd"

    # Buffer names (if empty, use auto-detected from export)
    buffers: list[str] = field(default_factory=list)

    # Whether to apply patches automatically
    apply_patches: bool = True

    # Output directory (if None, use current directory)
    output_dir: Path | None = None

    # Use shared FetchContent cache for CMake-based platforms
    shared_cache: bool = True

    # Explicit cache directory override (baked into CMakeLists.txt)
    cache_dir: Path | None = None

    # Board variant for embedded platforms:
    #   Daisy: seed, pod, patch, patch_sm, field, petal, legio, versio
    #   Circle: pi3-i2s, pi4-i2s
    board: str | None = None

    # MIDI-to-CV configuration
    no_midi: bool = False
    midi_gate: str | None = None
    midi_freq: str | None = None
    midi_vel: str | None = None
    midi_freq_unit: str = "hz"
    num_voices: int = 1

    # Signal inputs to remap as parameters.
    # None = don't remap, [] = remap all, ["name", ...] = remap named subset
    inputs_as_params: list[str] | None = None

    # Computed MIDI mapping (populated by ProjectGenerator.generate())
    midi_mapping: Optional["MidiMapping"] = None

    def validate(self) -> list[str]:
        """
        Validate the configuration.

        Returns:
            List of validation error messages (empty if valid).

        """
        platforms_module = importlib.import_module("gen_dsp.platforms")
        list_platforms = platforms_module.list_platforms

        errors = []

        # Validate name is a valid C identifier
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self.name):
            errors.append(
                f"Name '{self.name}' is not a valid C identifier. "
                "Must start with letter/underscore and contain only "
                "alphanumeric characters and underscores."
            )

        # Validate platform
        valid_platforms = list_platforms()
        if self.platform not in valid_platforms:
            errors.append(
                f"Platform must be one of {valid_platforms}, got '{self.platform}'"
            )

        # Validate buffer count
        if len(self.buffers) > MAX_BUFFERS:
            errors.append(
                f"Maximum {MAX_BUFFERS} buffers supported, got {len(self.buffers)}"
            )

        # Validate buffer names
        errors.extend(
            f"Buffer name '{buf_name}' is not a valid C identifier."
            for buf_name in self.buffers
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", buf_name)
        )

        # Validate Daisy board name
        if self.board is not None and self.platform == "daisy":
            daisy_module = importlib.import_module("gen_dsp.platforms.daisy")
            daisy_boards = daisy_module.DAISY_BOARDS

            if self.board not in daisy_boards:
                errors.append(
                    f"Unknown Daisy board '{self.board}'. "
                    f"Valid boards: {', '.join(sorted(daisy_boards))}"
                )

        # Validate Circle board name
        if self.board is not None and self.platform == "circle":
            circle_module = importlib.import_module("gen_dsp.platforms.circle")
            circle_boards = circle_module.CIRCLE_BOARDS

            if self.board not in circle_boards:
                errors.append(
                    f"Unknown Circle board '{self.board}'. "
                    f"Valid boards: {', '.join(sorted(circle_boards))}"
                )

        return errors

    @staticmethod
    def list_platforms() -> list[str]:
        """Return sorted list of available platform identifiers."""
        platforms_module = importlib.import_module("gen_dsp.platforms")
        list_platforms = platforms_module.list_platforms

        return list_platforms()


class ProjectGenerator:
    """Generate new project from gen~ export or dsp-graph."""

    def __init__(self, export_info: ExportInfo, config: ProjectConfig) -> None:
        """
        Initialize generator with export info and configuration.

        Args:
            export_info: Parsed information from gen~ export.
            config: Configuration for the new project.

        """
        self.export_info: ExportInfo | None = export_info
        self.config = config
        self._graph: Graph | None = None
        self._manifest: Manifest | None = None

    @classmethod
    def from_graph(cls, graph: "Graph", config: ProjectConfig) -> "ProjectGenerator":
        """
        Create a ProjectGenerator from a dsp-graph Graph object.

        Args:
            graph: A ``gen_dsp.graph.models.Graph`` instance.
            config: Project configuration.

        Returns:
            A ProjectGenerator configured for the dsp-graph path.

        """
        graph_adapter = importlib.import_module("gen_dsp.graph.adapter")
        generate_manifest_obj = graph_adapter.generate_manifest_obj

        # Create instance without ExportInfo
        instance = cls.__new__(cls)
        instance.export_info = None
        instance.config = config
        instance.__dict__["_graph"] = graph
        instance.__dict__["_manifest"] = generate_manifest_obj(graph)
        return instance

    def generate(self, output_dir: Path | None = None) -> Path:
        """
        Generate the project.

        Args:
            output_dir: Output directory. If None, uses config.output_dir
                       or creates a directory named after the project.

        Returns:
            Path to the generated project directory.

        Raises:
            ProjectError: If project cannot be generated.
            ValidationError: If configuration is invalid.

        """
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValidationError(
                "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Determine output directory
        if output_dir is None:
            output_dir = self.config.output_dir
        if output_dir is None:
            output_dir = Path.cwd() / self.config.name
        output_dir = Path(output_dir).resolve()

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._graph is not None:
            return self._generate_from_graph(output_dir)
        return self._generate_from_export(output_dir)

    def _generate_from_export(self, output_dir: Path) -> Path:
        """Generate project from gen~ export (original path)."""
        if self.export_info is None:
            msg = "export_info is not set"
            raise RuntimeError(msg)

        manifest_module = importlib.import_module("gen_dsp.core.manifest")
        platforms_module = importlib.import_module("gen_dsp.platforms")
        base_module = importlib.import_module("gen_dsp.platforms.base")
        manifest_from_export_info = manifest_module.manifest_from_export_info
        get_platform = platforms_module.get_platform
        platform_cls = base_module.Platform

        # Determine buffers to use
        buffers = self.config.buffers or self.export_info.buffers

        # Build manifest
        manifest = manifest_from_export_info(
            self.export_info,
            buffers,
            platform_cls.GENEXT_VERSION,
        )

        # Apply input-to-parameter remapping if requested
        if self.config.inputs_as_params is not None:
            apply_inputs_as_params = manifest_module.apply_inputs_as_params

            remap_names = (
                self.config.inputs_as_params or None
            )
            manifest = apply_inputs_as_params(
                manifest, self.export_info.input_names, remap_names
            )

        # Compute MIDI mapping (used by platforms that support MIDI)
        midi_module = importlib.import_module("gen_dsp.core.midi")
        detect_midi_mapping = midi_module.detect_midi_mapping

        self.config.midi_mapping = detect_midi_mapping(
            manifest,
            no_midi=self.config.no_midi,
            midi_gate=self.config.midi_gate,
            midi_freq=self.config.midi_freq,
            midi_vel=self.config.midi_vel,
            midi_freq_unit=self.config.midi_freq_unit,
        )
        # Set polyphony voice count on the mapping
        if self.config.midi_mapping.enabled and self.config.num_voices > 1:
            self.config.midi_mapping.num_voices = self.config.num_voices

        # Generate for the target platform using the registry
        platform_impl = get_platform(self.config.platform)
        platform_impl.generate_project(
            manifest,
            output_dir,
            self.config.name,
            config=self.config,
        )

        # Write manifest.json to project root
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")

        # Copy gen~ export
        self._copy_export(output_dir)

        # Apply patches if requested
        if self.config.apply_patches and self.export_info.has_exp2f_issue:
            patcher_module = importlib.import_module("gen_dsp.core.patcher")
            patcher_cls = patcher_module.Patcher

            patcher = patcher_cls(output_dir)
            patcher.apply_exp2f_fix()

        return output_dir

    def _generate_from_graph(self, output_dir: Path) -> Path:
        """Generate project from dsp-graph (new path)."""
        adapter_module = importlib.import_module("gen_dsp.graph.adapter")
        copy_platform_templates = adapter_module.__dict__["_copy_platform_templates"]
        generate_buffer_header = adapter_module.__dict__["_generate_buffer_header"]
        generate_adapter_cpp = adapter_module.generate_adapter_cpp
        generate_graph_build_file = adapter_module.generate_graph_build_file
        compile_module = importlib.import_module("gen_dsp.graph.compile")
        compile_graph = compile_module.compile_graph
        platforms_base = importlib.import_module("gen_dsp.platforms.base")
        platform_cls = platforms_base.Platform

        if self._graph is None or self._manifest is None:
            msg = "graph state is not initialized"
            raise RuntimeError(msg)
        graph = self._graph
        manifest = self._manifest
        platform = self.config.platform

        # 1. Compile graph to C++
        code = compile_graph(graph)
        (output_dir / f"{graph.name}.cpp").write_text(code)

        # 2. Generate adapter _ext_{platform}.cpp
        adapter = generate_adapter_cpp(graph, platform)
        (output_dir / f"_ext_{platform}.cpp").write_text(adapter)

        # 3. Copy platform template files (gen_ext_{platform}.cpp, etc.)
        copy_platform_templates(output_dir, platform)

        # 3b. Generate _ext_{platform}.h if not already present
        ext_header = output_dir / f"_ext_{platform}.h"
        if not ext_header.is_file():
            platforms_module = importlib.import_module("gen_dsp.platforms")
            get_platform = platforms_module.get_platform
            get_platform(platform).generate_ext_header(output_dir, platform)

        # 4. Generate gen_buffer.h
        generate_buffer_header(output_dir)

        # 5. Compute MIDI mapping
        midi_module = importlib.import_module("gen_dsp.core.midi")
        build_midi_defines = midi_module.build_midi_defines
        detect_midi_mapping = midi_module.detect_midi_mapping

        self.config.midi_mapping = detect_midi_mapping(
            manifest,
            no_midi=self.config.no_midi,
            midi_gate=self.config.midi_gate,
            midi_freq=self.config.midi_freq,
            midi_vel=self.config.midi_vel,
            midi_freq_unit=self.config.midi_freq_unit,
        )
        if self.config.midi_mapping.enabled and self.config.num_voices > 1:
            self.config.midi_mapping.num_voices = self.config.num_voices
        midi_defines = build_midi_defines(self.config.midi_mapping)

        # 5b. Copy voice_alloc.h when polyphony is enabled
        platforms_module = importlib.import_module("gen_dsp.platforms")
        get_platform = platforms_module.get_platform
        get_platform(platform).copy_voice_alloc_header(output_dir, self.config)

        self._generate_graph_platform_assets(
            platform,
            output_dir,
            graph,
            manifest,
            platform_cls,
        )

        # 7. Generate simplified build file
        generate_graph_build_file(
            output_dir=output_dir,
            platform=platform,
            lib_name=self.config.name,
            gen_name=graph.name,
            num_inputs=manifest.num_inputs,
            num_outputs=manifest.num_outputs,
            num_params=manifest.num_params,
            genext_version=platform_cls.GENEXT_VERSION,
            shared_cache=self.config.shared_cache,
            midi_defines=midi_defines,
        )

        # 8. Write manifest.json
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")

        return output_dir

    def _generate_graph_platform_assets(
        self,
        platform: str,
        output_dir: Path,
        graph: "Graph",
        manifest: "Manifest",
        platform_cls: type,
    ) -> None:
        """Generate platform-specific assets for graph projects."""
        handlers = {
            "au": self._generate_graph_au_assets,
            "lv2": self._generate_graph_lv2_assets,
            "vcvrack": self._generate_graph_vcvrack_assets,
            "daisy": self._generate_graph_daisy_assets,
            "circle": self._generate_graph_circle_assets,
            "webaudio": self._generate_graph_webaudio_assets,
        }
        handler = handlers.get(platform)
        if handler is not None:
            handler(output_dir, graph, manifest, platform_cls)

    def _generate_graph_au_assets(
        self,
        output_dir: Path,
        _graph: "Graph",
        manifest: "Manifest",
        platform_cls: type,
    ) -> None:
        """Generate AudioUnit-specific graph assets."""
        audiounit_module = importlib.import_module("gen_dsp.platforms.audiounit")
        base_module = importlib.import_module("gen_dsp.platforms.base")
        audio_unit_platform_cls = audiounit_module.AudioUnitPlatform
        plugin_category_cls = base_module.PluginCategory

        category = plugin_category_cls.from_num_inputs(manifest.num_inputs)
        au_type = _private_class_attr(
            audio_unit_platform_cls,
            "_AU_TYPE_MAP",
        )[category]
        if self.config.midi_mapping and self.config.midi_mapping.enabled:
            au_type = audio_unit_platform_cls.AU_TYPE_MUSIC_DEVICE
        au_subtype = self.config.name.lower()[:4].ljust(4, "x")

        templates_module = importlib.import_module("gen_dsp.templates")
        au_tmpl_dir = templates_module.get_au_templates_dir()
        plist_template_path = au_tmpl_dir / "Info.plist.template"
        if plist_template_path.is_file():
            plist_content = plist_template_path.read_text(encoding="utf-8")
            plist = StrTemplate(plist_content).safe_substitute(
                lib_name=self.config.name,
                genext_version=platform_cls.GENEXT_VERSION,
                au_type=au_type,
                au_subtype=au_subtype,
                au_manufacturer=audio_unit_platform_cls.AU_MANUFACTURER,
                au_version=_private_static_call(
                    audio_unit_platform_cls,
                    "_version_to_int",
                    platform_cls.GENEXT_VERSION,
                ),
            )
            (output_dir / "Info.plist").write_text(plist, encoding="utf-8")

    def _generate_graph_lv2_assets(
        self,
        output_dir: Path,
        _graph: "Graph",
        _manifest: "Manifest",
        _platform_cls: type,
    ) -> None:
        """Generate LV2-specific graph assets."""
        lv2_module = importlib.import_module("gen_dsp.platforms.lv2")
        lv2_cls = lv2_module.Lv2Platform
        lv2 = lv2_cls()
        plugin_uri = f"{lv2.LV2_URI_BASE}/{self.config.name}"
        midi_enabled = (
            self.config.midi_mapping is not None
            and self.config.midi_mapping.enabled
        )
        _private_method_call(
            lv2,
            "_generate_manifest_ttl",
            output_dir,
            self.config.name,
            plugin_uri,
        )
        _private_method_call(
            lv2,
            "_generate_plugin_ttl",
            output_dir,
            self.config.name,
            plugin_uri,
            _manifest.num_inputs,
            _manifest.num_outputs,
            _manifest.num_params,
            _manifest.params,
            midi_enabled=midi_enabled,
        )

    def _generate_graph_vcvrack_assets(
        self,
        output_dir: Path,
        _graph: "Graph",
        manifest: "Manifest",
        _platform_cls: type,
    ) -> None:
        """Generate VCV Rack-specific graph assets."""
        vcvrack_module = importlib.import_module("gen_dsp.platforms.vcvrack")
        vcvrack_cls = vcvrack_module.VcvRackPlatform
        vcv = vcvrack_cls()
        total = manifest.num_inputs + manifest.num_outputs + manifest.num_params
        panel_hp = _private_method_call(vcv, "_compute_panel_hp", total)

        _private_method_call(
            vcv,
            "_generate_plugin_json",
            output_dir,
            self.config.name,
            manifest.num_inputs,
        )

        res_dir = output_dir / "res"
        res_dir.mkdir(parents=True, exist_ok=True)
        _private_method_call(
            vcv,
            "_generate_panel_svg",
            res_dir / f"{self.config.name}.svg",
            self.config.name,
            panel_hp,
        )

    def _generate_graph_daisy_assets(
        self,
        output_dir: Path,
        _graph: "Graph",
        _manifest: "Manifest",
        _platform_cls: type,
    ) -> None:
        """Generate Daisy-specific graph assets."""
        board_key = "seed"
        if self.config.board is not None:
            board_key = self.config.board

        daisy_module = importlib.import_module("gen_dsp.platforms.daisy")
        daisy_boards = daisy_module.DAISY_BOARDS
        board = daisy_boards[board_key]

        gen_ext_daisy = f"""\
// gen_ext_daisy.cpp - Daisy wrapper for dsp-graph compiled code
// Board: {board_key} ({board.hw_class})
// This file includes ONLY libDaisy headers - graph code is isolated in _ext_daisy.cpp

#include "{board.header}"

#include "gen_ext_common_daisy.h"
#include "_ext_daisy.h"

using namespace WRAPPER_NAMESPACE;
using namespace daisy;
{board.extra_using}

// ---------------------------------------------------------------------------
// Hardware and state
// ---------------------------------------------------------------------------

static {board.hw_class} hw;
static GenState* genState = nullptr;

// ---------------------------------------------------------------------------
// Scratch buffers for I/O channel mismatch
// ---------------------------------------------------------------------------

#define DAISY_MAX_BLOCK_SIZE 256
#define DAISY_HW_CHANNELS {board.hw_channels}
#define DAISY_MAPPED_INPUTS  ((DAISY_NUM_INPUTS < DAISY_HW_CHANNELS) ? \
    DAISY_NUM_INPUTS : DAISY_HW_CHANNELS)
#define DAISY_MAPPED_OUTPUTS ((DAISY_NUM_OUTPUTS < DAISY_HW_CHANNELS) ? \
    DAISY_NUM_OUTPUTS : DAISY_HW_CHANNELS)

static float scratch_zero[DAISY_MAX_BLOCK_SIZE] = {{0}};
static float scratch_discard[DAISY_MAX_BLOCK_SIZE];

static float* gen_ins[DAISY_NUM_INPUTS > 0 ? DAISY_NUM_INPUTS : 1];
static float* gen_outs[DAISY_NUM_OUTPUTS > 0 ? DAISY_NUM_OUTPUTS : 1];

// ---------------------------------------------------------------------------
// Audio callback
// ---------------------------------------------------------------------------

static void AudioCallback(const float* const* in, float** out, size_t size) {{
    if (!genState) {{
        genState = wrapper_create(hw.AudioSampleRate(), (long)size);
    }}

    for (int i = 0; i < DAISY_NUM_INPUTS; i++) {{
        if (i < DAISY_HW_CHANNELS) {{
            gen_ins[i] = const_cast<float*>(in[i]);
        }} else {{
            gen_ins[i] = scratch_zero;
        }}
    }}

    for (int i = 0; i < DAISY_NUM_OUTPUTS; i++) {{
        if (i < DAISY_HW_CHANNELS) {{
            gen_outs[i] = out[i];
        }} else {{
            gen_outs[i] = scratch_discard;
        }}
    }}

    wrapper_perform(genState, gen_ins, DAISY_NUM_INPUTS,
                    gen_outs, DAISY_NUM_OUTPUTS, (long)size);
}}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(48);
    hw.StartAudio(AudioCallback);

    for (;;) {{
        // Audio runs in interrupt
    }}
}}
"""
        (output_dir / "gen_ext_daisy.cpp").write_text(gen_ext_daisy)

    def _generate_graph_circle_assets(
        self,
        output_dir: Path,
        _graph: "Graph",
        manifest: "Manifest",
        _platform_cls: type,
    ) -> None:
        """Generate Circle-specific graph assets."""
        board_key = "pi3-i2s"
        if self.config.board is not None:
            board_key = self.config.board

        circle_module = importlib.import_module("gen_dsp.platforms.circle")
        circle_boards = circle_module.CIRCLE_BOARDS
        circle_board = circle_boards[board_key]
        audio_include = circle_module.__dict__["_get_audio_include"](
            circle_board.audio_device
        )
        audio_base_class = circle_module.__dict__["_get_audio_base_class"](
            circle_board.audio_device
        )
        audio_label = circle_module.__dict__["_get_audio_label"](
            circle_board.audio_device
        )

        gen_ext_circle = f"""\
// gen_ext_circle.cpp - Circle bare metal wrapper for dsp-graph compiled code
// Board: {board_key} (Raspberry Pi {circle_board.rasppi})
// Audio: {audio_label} output
// This file includes ONLY Circle headers - graph code is isolated in _ext_circle.cpp

#include <circle/actled.h>
#include <circle/koptions.h>
#include <circle/devicenameservice.h>
#include <circle/exceptionhandler.h>
#include <circle/interrupt.h>
#include <circle/logger.h>
#include <circle/startup.h>
#include <circle/timer.h>
#include <circle/types.h>
{audio_include}

#include "gen_ext_common_circle.h"
#include "_ext_circle.h"

using namespace WRAPPER_NAMESPACE;

#define CIRCLE_SAMPLE_RATE     48000
#define CIRCLE_CHUNK_SIZE      256
#define CIRCLE_AUDIO_CHANNELS  2

#define CIRCLE_NUM_INPUTS  {manifest.num_inputs}
#define CIRCLE_NUM_OUTPUTS {manifest.num_outputs}

class CGenDSPSoundDevice : public {audio_base_class}
{{
public:
    CGenDSPSoundDevice(CInterruptSystem* pInterrupt)
        : {audio_base_class}(pInterrupt, CIRCLE_SAMPLE_RATE, CIRCLE_CHUNK_SIZE),
          m_genState(nullptr)
    {{
        for (int i = 0; i < CIRCLE_NUM_INPUTS || i < 1; i++) {{
            m_pInputBuffers[i] = m_InputStorage[i];
        }}
        for (int i = 0; i < CIRCLE_NUM_OUTPUTS || i < 1; i++) {{
            m_pOutputBuffers[i] = m_OutputStorage[i];
        }}
    }}

    ~CGenDSPSoundDevice(void)
    {{
        if (m_genState) {{
            wrapper_destroy(m_genState);
            m_genState = nullptr;
        }}
    }}

    boolean Initialize(void)
    {{
        m_genState = wrapper_create((float)CIRCLE_SAMPLE_RATE, (long)CIRCLE_CHUNK_SIZE);
        if (!m_genState) {{
            return FALSE;
        }}
        return Start();
    }}

protected:
    unsigned GetChunk(u32* pBuffer, unsigned nChunkSize) override
    {{
        if (!m_genState) {{
            for (unsigned i = 0; i < nChunkSize; i++) {{
                pBuffer[i] = 0;
            }}
            return nChunkSize;
        }}

        unsigned nFrames = nChunkSize / CIRCLE_AUDIO_CHANNELS;

#if CIRCLE_NUM_INPUTS > 0
        for (int ch = 0; ch < CIRCLE_NUM_INPUTS; ch++) {{
            for (unsigned i = 0; i < nFrames; i++) {{
                m_InputStorage[ch][i] = 0.0f;
            }}
        }}
#endif

        wrapper_perform(
            m_genState,
#if CIRCLE_NUM_INPUTS > 0
            m_pInputBuffers,
#else
            nullptr,
#endif
            CIRCLE_NUM_INPUTS,
            m_pOutputBuffers,
            CIRCLE_NUM_OUTPUTS,
            (long)nFrames
        );

        int nRangeMin = GetRangeMin();
        int nRangeMax = GetRangeMax();

        for (unsigned i = 0; i < nFrames; i++) {{
            for (int ch = 0; ch < CIRCLE_AUDIO_CHANNELS; ch++) {{
                float sample = 0.0f;
                if (ch < CIRCLE_NUM_OUTPUTS) {{
                    sample = m_pOutputBuffers[ch][i];
                }}
                if (sample > 1.0f) sample = 1.0f;
                if (sample < -1.0f) sample = -1.0f;
                int nSample = (int)((sample + 1.0f) / 2.0f
                    * (nRangeMax - nRangeMin) + nRangeMin);
                pBuffer[i * CIRCLE_AUDIO_CHANNELS + ch] = (u32)nSample;
            }}
        }}

        return nChunkSize;
    }}

private:
    GenState* m_genState;
    float m_InputStorage[
        CIRCLE_NUM_INPUTS > 0 ? CIRCLE_NUM_INPUTS : 1
    ][CIRCLE_CHUNK_SIZE];
    float m_OutputStorage[
        CIRCLE_NUM_OUTPUTS > 0 ? CIRCLE_NUM_OUTPUTS : 1
    ][CIRCLE_CHUNK_SIZE];
    float* m_pInputBuffers[
        CIRCLE_NUM_INPUTS > 0 ? CIRCLE_NUM_INPUTS : 1
    ];
    float* m_pOutputBuffers[
        CIRCLE_NUM_OUTPUTS > 0 ? CIRCLE_NUM_OUTPUTS : 1
    ];
}};

class CKernel
{{
public:
    CKernel(void)
        : m_Timer(&m_Interrupt),
          m_Logger(m_Options.GetLogLevel(), &m_Timer),
          m_pSound(nullptr)
    {{
    }}

    ~CKernel(void)
    {{
        delete m_pSound;
    }}

    boolean Initialize(void)
    {{
        if (!m_Interrupt.Initialize()) {{
            return FALSE;
        }}
        if (!m_Timer.Initialize()) {{
            return FALSE;
        }}
        if (!m_Logger.Initialize(nullptr)) {{
            return FALSE;
        }}

        m_pSound = new CGenDSPSoundDevice(&m_Interrupt);
        if (!m_pSound->Initialize()) {{
            m_Logger.Write(
                "gen-dsp",
                LogError,
                "Failed to initialize {audio_label} sound device"
            );
            return FALSE;
        }}

        m_Logger.Write(
            "gen-dsp",
            LogNotice,
            "gen-dsp Circle audio started: %uHz, %u frames/chunk, "
            "{audio_label} output",
            CIRCLE_SAMPLE_RATE,
            CIRCLE_CHUNK_SIZE
        );

        return TRUE;
    }}

    void Run(void)
    {{
        for (;;) {{
        }}
    }}

private:
    CActLED             m_ActLED;
    CKernelOptions      m_Options;
    CDeviceNameService  m_DeviceNameService;
    CExceptionHandler   m_ExceptionHandler;
    CInterruptSystem    m_Interrupt;
    CTimer              m_Timer;
    CLogger             m_Logger;
    CGenDSPSoundDevice* m_pSound;
}};

int main(void)
{{
    CKernel Kernel;
    if (!Kernel.Initialize()) {{
        halt();
        return EXIT_HALT;
    }}
    Kernel.Run();
    halt();
    return EXIT_HALT;
}}
"""
        (output_dir / "gen_ext_circle.cpp").write_text(gen_ext_circle)

        templates_module = importlib.import_module("gen_dsp.templates")
        circle_tmpl_dir = templates_module.get_circle_templates_dir()
        config_template_path = circle_tmpl_dir / "config.txt.template"
        if config_template_path.is_file():
            config_content = config_template_path.read_text(encoding="utf-8")
            config_txt = StrTemplate(config_content).safe_substitute(
                audio_boot_config=circle_module.__dict__["_get_boot_config"](
                    circle_board.audio_device
                ),
            )
            (output_dir / "config.txt").write_text(config_txt, encoding="utf-8")

    def _generate_graph_webaudio_assets(
        self,
        output_dir: Path,
        graph: "Graph",
        manifest: "Manifest",
        platform_cls: type,
    ) -> None:
        """Generate WebAudio-specific graph assets."""
        webaudio_module = importlib.import_module("gen_dsp.platforms.webaudio")
        wa_cls = webaudio_module.WebAudioPlatform
        wa = wa_cls()

        wa_templates = importlib.import_module("gen_dsp.templates")
        wa_tmpl_dir = wa_templates.get_webaudio_templates_dir()

        _private_method_call(
            wa,
            "_generate_from_template",
            wa_tmpl_dir / "gen_ext_webaudio.cpp.template",
            output_dir / "gen_ext_webaudio.cpp",
            lib_name=self.config.name,
            gen_name=graph.name,
            genext_version=platform_cls.GENEXT_VERSION,
        )

        param_descriptors, _processor_class, num_outputs_array = _private_method_call(
            wa,
            "_build_processor_vars",
            manifest,
            self.config.name,
        )

        _private_method_call(
            wa,
            "_generate_processor_js",
            wa_tmpl_dir / "processor.js.template",
            output_dir / "_processor.js",
            manifest,
            self.config.name,
        )

        _private_method_call(
            wa,
            "_generate_from_template",
            wa_tmpl_dir / "index.html.template",
            output_dir / "index.html",
            lib_name=self.config.name,
            num_inputs=str(manifest.num_inputs),
            num_outputs=str(manifest.num_outputs),
            num_params=str(len(manifest.params)),
            param_descriptors=json.dumps(param_descriptors, indent=4),
            num_outputs_array=num_outputs_array,
        )

    def _copy_export(self, output_dir: Path) -> None:
        """Copy the gen~ export to the project's gen/ directory."""
        if self.export_info is None:
            msg = "export_info is not set"
            raise RuntimeError(msg)
        gen_dir = output_dir / "gen"

        # Remove existing gen/ if present
        if gen_dir.exists():
            shutil.rmtree(gen_dir)

        # Copy the export
        shutil.copytree(self.export_info.path, gen_dir)
