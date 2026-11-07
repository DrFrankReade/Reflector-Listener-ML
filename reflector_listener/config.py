"""Validated buffer and timing configuration for reflectorListen.

``BufferConfig`` rejects invalid values on construction. File loading is tolerant:
invalid individual settings keep their defaults while valid siblings are retained,
and diagnostics are emitted through this module's logger. An unreadable or malformed
file uses all defaults. The file suffix explicitly selects TOML or INI; a malformed
TOML file is never reinterpreted as INI. Missing files are created when possible,
but an unwritable configuration directory does not prevent application startup.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field, fields, Field
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping


logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# Default buffer configuration template with human-readable comments.
# Tuned for realistic VoIP environments: lossy Wi-Fi links, consumer ISP routing jitter,
# and Windows audio subsystem latency.
DEFAULT_BUFFER_CONFIG_TOML = """# ==============================================================================
# reflectorListen Buffer & Playout Configuration
# ==============================================================================
# This configuration tunes packet jitter buffering, packet loss concealment (PLC),
# and audio device hardware queues.
#
# These settings have been tuned for typical VoIP conditions: lossy Wi-Fi links,
# consumer ISP routing jitter, and Windows scheduler latency. You can adjust
# these values to optimize between audio latency and dropout resilience.
# Changes take effect on the next application launch.
# ==============================================================================

[jitter_buffer]
# Startup delay before initial audio playback begins (seconds).
# Pre-buffers audio frames at the start of a transmission to absorb Wi-Fi jitter.
# Lower values reduce push-to-talk turnaround delay; higher values prevent initial stutter.
# Default: 0.240 (240 ms / 12 frames at 50 fps)
startup_delay_seconds = 0.240

# Re-buffering delay after an audio underrun or gap before resuming playback (seconds).
# If packet loss or network jitter starves the buffer, playout parks and waits for
# this much headroom to accumulate before resuming speech.
# Default: 0.160 (160 ms / 8 frames)
resume_delay_seconds = 0.160

# Maximum capacity of the jitter reorder buffer (in 20 ms frames).
# Wi-Fi often delivers packets in bursts after small contention stalls. If frames
# accumulate beyond this ceiling, the oldest stale frames are dropped to keep
# playback near real time without unbounded latency growth.
# Default: 75 (1.5 seconds); accommodates the 61-frame bursts in reflector captures.
# This is a capacity limit, not an added startup delay.
max_buffered_frames = 75

# Maximum consecutive synthetic concealment frames to decode during packet loss.
# When a 20 ms packet is missing at its playout deadline, the native AMBE vocoder
# synthesizes speech using Packet Loss Concealment (PLC). Playout only parks if
# packet loss exceeds this limit, cleanly bridging brief Wi-Fi dropouts.
# Default: 5 (100 ms / 5 frames)
max_concealment_frames = 5

# Tolerance in 20 ms slots for mapping modulo-21 wire sequence numbers to monotonic frames.
# Disambiguates small sequence wraps after network stalls.
# Default: 8 (160 ms)
sequence_tolerance_frames = 8

# Maximum catch-up frames serviced per 20 ms playout cycle.
# Prevents playout bursts after brief thread stalls while allowing the buffer to catch up.
# Default: 6 (120 ms)
max_playout_catchup_frames = 6

# Inactivity timeout before an un-terminated voice stream is automatically closed (seconds).
# If a transmission stops without an explicit D-Star END packet, the stream finishes
# after this timeout.
# Default: 10.0
voice_timeout_seconds = 10.0

[audio_output]
# Internal queue capacity between the network/decoder thread and the audio player (20 ms frames).
# Buffers decoded PCM before submission to the sound card.
# Default: 100 (2.0 seconds)
max_queued_frames = 100

# Windows waveOut multimedia hardware ring buffer size (in 20 ms frames).
# Number of pre-queued WAVEHDR buffers submitted directly to the sound card driver
# to prevent audio crackle and DAC starvation during Windows CPU spikes.
# Default: 16 (320 ms)
max_device_frames = 16
"""


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Tuned VoIP parameters, validated equally for callers and loaded files.

    Delays must be finite and nonnegative, and the inactivity timeout must be at
    least 0.5 seconds. Frame counts are integers with the minimums declared below;
    booleans and fractional counts are rejected rather than silently coerced.
    """

    startup_delay_seconds: float = field(
        default=0.240, metadata={"section": "jitter_buffer", "minimum": 0.0}
    )
    resume_delay_seconds: float = field(
        default=0.160, metadata={"section": "jitter_buffer", "minimum": 0.0}
    )
    max_buffered_frames: int = field(
        default=75, metadata={"section": "jitter_buffer", "minimum": 2}
    )
    max_concealment_frames: int = field(
        default=5, metadata={"section": "jitter_buffer", "minimum": 1}
    )
    sequence_tolerance_frames: int = field(
        default=8, metadata={"section": "jitter_buffer", "minimum": 1}
    )
    max_playout_catchup_frames: int = field(
        default=6, metadata={"section": "jitter_buffer", "minimum": 1}
    )
    voice_timeout_seconds: float = field(
        default=10.0, metadata={"section": "jitter_buffer", "minimum": 0.5}
    )
    max_queued_frames: int = field(
        default=100, metadata={"section": "audio_output", "minimum": 4}
    )
    max_device_frames: int = field(
        default=16, metadata={"section": "audio_output", "minimum": 2}
    )

    def __post_init__(self) -> None:
        for setting in fields(self):
            _validate_value(setting, getattr(self, setting.name))


def _validate_value(setting: Field[Any], value: Any) -> None:
    """Keep the validation rules shared by construction and tolerant loading."""
    if isinstance(setting.default, int):
        valid_type = isinstance(value, int) and not isinstance(value, bool)
        requirement = "an integer"
    else:
        valid_type = isinstance(value, (float, int)) and not isinstance(value, bool)
        requirement = "a finite number"
        if valid_type:
            try:
                valid_type = math.isfinite(value)
            except OverflowError:
                valid_type = False
    minimum = setting.metadata["minimum"]
    if not valid_type or value < minimum:
        raise ValueError(f"{setting.name} must be {requirement} >= {minimum}")


def _parse_dict(data: Mapping[str, Any]) -> BufferConfig:
    settings = fields(BufferConfig)
    known_sections = {setting.metadata["section"] for setting in settings}
    for section, values in data.items():
        if section not in known_sections:
            logger.warning("Ignoring unknown buffer configuration section %r", section)
        elif not isinstance(values, Mapping):
            logger.warning("Ignoring buffer configuration section %r: expected a table", section)
        else:
            known_keys = {
                setting.name for setting in settings
                if setting.metadata["section"] == section
            }
            for key in values.keys() - known_keys:
                logger.warning("Ignoring unknown buffer configuration setting %s.%s", section, key)

    parsed: dict[str, Any] = {}
    for setting in settings:
        section = setting.metadata["section"]
        values = data.get(section, {})
        if not isinstance(values, Mapping) or setting.name not in values:
            continue
        value = values[setting.name]
        try:
            # INI stores numeric values as strings. Do not truncate TOML floats
            # or accept booleans as frame counts through int(value).
            if isinstance(value, str):
                value = int(value) if isinstance(setting.default, int) else float(value)
            _validate_value(setting, value)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Invalid buffer setting %s.%s=%r: %s; using default %r",
                section, setting.name, values[setting.name], exc, setting.default,
            )
        else:
            parsed[setting.name] = value
    return BufferConfig(**parsed)


def write_default_buffer_config(path: Path | str) -> None:
    """Create the commented defaults without overwriting an existing file.

    The template is valid in both supported formats. I/O errors propagate to the
    caller; ``load_buffer_config`` reports them and continues with defaults.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("x", encoding="utf-8") as output:
            output.write(DEFAULT_BUFFER_CONFIG_TOML)
    except FileExistsError:
        if not file_path.is_file():
            raise


def load_buffer_config(path: Path | str | None = None) -> BufferConfig:
    """Read a .toml or .ini file, reporting errors and retaining safe defaults.

    With no path, prefer ``buffers.toml`` to ``buffers.ini`` in the current
    directory. If neither exists, create ``buffers.toml`` when writable.
    """
    defaults = BufferConfig()
    try:
        target_path = Path(path) if path is not None else next(
            (candidate for candidate in (Path("buffers.toml"), Path("buffers.ini"))
             if candidate.is_file()),
            Path("buffers.toml"),
        )
    except OSError as exc:
        logger.warning("Cannot locate buffer configuration: %s; using defaults", exc)
        return defaults

    suffix = target_path.suffix.lower()
    if suffix not in {".toml", ".ini"}:
        logger.warning("Unsupported buffer configuration format %s; expected .toml or .ini", target_path)
        return defaults

    try:
        content = target_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        try:
            write_default_buffer_config(target_path)
        except OSError as exc:
            logger.warning("Cannot create buffer configuration %s: %s; using defaults", target_path, exc)
        return defaults
    except (OSError, UnicodeError) as exc:
        logger.warning("Cannot read buffer configuration %s: %s; using defaults", target_path, exc)
        return defaults

    if suffix == ".ini":
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read_string(content)
        except configparser.Error as exc:
            logger.warning("Invalid INI buffer configuration %s: %s; using defaults", target_path, exc)
            return defaults
        return _parse_dict({section: dict(cp[section]) for section in cp.sections()})

    if tomllib is None:
        logger.warning("Cannot read TOML buffer configuration %s: a TOML parser is unavailable; using defaults", target_path)
        return defaults
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        logger.warning("Invalid TOML buffer configuration %s: %s; using defaults", target_path, exc)
        return defaults
    return _parse_dict(parsed)
