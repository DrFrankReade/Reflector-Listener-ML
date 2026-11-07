"""UDP reflector connection state machine and realtime receive pipeline."""

from __future__ import annotations

import ctypes
import dataclasses
import re
import secrets
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol as TypingProtocol, Sequence

from .audio import BYTES_PER_FRAME
from .decoder import AmbeDecoder, BitOrder, DecoderError, DEFAULT_GAIN
from .diagnostics import DiagnosticLogger
from .events import ApplicationEvent, SessionEvent
from .protocols import (
    ControlKind,
    ControlPacket,
    MediaKind,
    MediaPacket,
    Protocol,
    RFHeader,
    ReflectorTarget,
    SEQUENCE_MODULUS,
    TargetStreamFilter,
    build_header_packet,
    build_link_packet,
    build_login_packet,
    build_poll_packet,
    build_rf_header,
    build_unlink_packet,
    build_voice_packet,
    parse_control_packet,
    parse_media_packet,
    parse_rf_header,
    parse_target,
    sequence_distance,
)
from .config import BufferConfig, load_buffer_config
from .packet_logger import PacketLogger
from .preferences import Preferences
from .stream_log import BinaryStreamLog, Recording, RecordRef
from .transmit_data import PICTURE_BEEP, application_frames, gpsa_frame, slow_data_cycle, station_message_cycle
from .transmit_diagnostics import TransmitDiagnostics
from .transmit_timing import FramePacer
from .rsms1 import (
    FAST_SYNC_DV_PAYLOAD,
)


_CALLSIGN_PATTERN = re.compile(r"^[A-Z0-9/]{3,8}$")
_AUTO_ORDER_FRAMES = 5
_COMPLETED_STREAM_HOLDOFF_SECONDS = 2.0
_FRAME_INTERVAL_SECONDS = 0.020
_MAX_DATAGRAMS_PER_CYCLE = 64
# Large enough for any IPv4 UDP payload, including unexpected/malformed traffic.
_UDP_RECEIVE_BYTES = 65535
PTT_TIMEOUT_SECONDS = 180.0


class AudioSink(TypingProtocol):
    """Protocol for abstracting the audio playback interface.

    Implementations must manage stream generations to ensure that audio
    queued from a discontinued stream is not played after cancellation.
    """
    generation: int

    def start(self) -> None: ...
    def retry(self) -> None: ...
    def begin_stream(self) -> int: ...
    def cancel_stream(self) -> None: ...
    def submit(self, pcm: bytes, generation: Optional[int] = None) -> bool: ...


class ApplicationReceiver(TypingProtocol):
    def set_logger(self, logger: DiagnosticLogger | None) -> None: ...
    def reset(self) -> None: ...
    def start_stream(self, stream_id: int) -> None: ...
    def end_stream(self, stream_id: int) -> Sequence[ApplicationEvent]: ...
    def feed(self, stream_id: int, sequence: int, dv_payload: bytes) -> tuple[bool, Sequence[ApplicationEvent]]: ...


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    LINKED = "linked"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass(frozen=True)
class ConnectionConfig:
    callsign: str
    reflector: str
    module: str
    host: str
    local_module: str = "D"
    gain: float = DEFAULT_GAIN
    prefer_standard_local_port: bool = True

    def normalized(self) -> "ConnectionConfig":
        callsign = self.callsign.strip().upper()
        if callsign == "N0CALL" or not _CALLSIGN_PATTERN.fullmatch(callsign):
            raise ValueError(
                "enter a valid licensed callsign (3-8 letters, digits, or '/')"
            )
        target = parse_target(self.reflector, self.module)
        host = self.host.strip()
        if not host:
            raise ValueError(f"no host/IP is available for {target.name}")
        local_module = self.local_module.strip().upper()[:1] or "D"
        if not ("A" <= local_module <= "Z"):
            raise ValueError("local module must be A through Z")
        gain = float(self.gain)
        if not 0.0 <= gain <= 10.0:
            raise ValueError("audio gain must be between 0 and 10")
        return dataclasses.replace(
            self,
            callsign=callsign,
            reflector=target.name,
            module=target.module,
            host=host,
            local_module=local_module,
            gain=gain,
        )

    @property
    def target(self) -> ReflectorTarget:
        return parse_target(self.reflector, self.module)


@dataclass(frozen=True)
class _WorkerContext:
    """Immutable ownership token for one connection worker.

    DNS resolution cannot be cancelled portably.  A worker may therefore
    outlive a user disconnect for a short time.  Its generation and private
    stop event make every later state/socket operation conditional on still
    owning the session.
    """

    generation: int
    stop: threading.Event
    config: ConnectionConfig
    target: ReflectorTarget
    stream_filter: TargetStreamFilter


@dataclass
class _VoiceStream:
    """Tracks the state of a single incoming voice transmission.

    This includes the jitter buffer for reordering UDP datagrams,
    AMBE decoder continuity metrics, and sequence indices for playout.
    """
    stream_id: int
    header: RFHeader
    protocol: Protocol
    audio_generation: int
    started_at: float
    last_media_at: float
    order: Optional[BitOrder] = None
    frame_buffer: dict[int, bytes] = field(default_factory=dict)
    highest_sequence: Optional[int] = None
    highest_index: int = 0
    highest_received_at: Optional[float] = None
    playout_index: Optional[int] = None
    next_playout_at: Optional[float] = None
    rebuffer_until: Optional[float] = None
    parked_at: Optional[float] = None
    end_index: Optional[int] = None
    end_reason: str = "end packet"
    msb_cost: int = 0
    lsb_cost: int = 0
    scored_frames: int = 0
    received_frames: int = 0
    decoded_frames: int = 0
    lost_frames: int = 0
    concealed_frames: int = 0
    skipped_frames: int = 0
    late_frames: int = 0
    reordered_frames: int = 0
    duplicate_frames: int = 0
    sequence_conflicts: int = 0
    resync_events: int = 0
    delayed_resumes: int = 0
    buffer_dropped_frames: int = 0
    peak_buffered_frames: int = 0
    buffer_overflow_active: bool = False
    decoder_reset_pending: bool = False
    concealment_run: int = 0
    concealment_reset: bool = False
    max_concealment_run: int = 0
    corrected_frames: int = 0
    clipped_samples: int = 0
    application_data: bool = False
    recording: Recording | None = None


@dataclass
class _TransmitStream:
    stream_id: int
    header: bytes
    started_at: float
    sequence: int = 0
    packet_counter: int = 0
    frames: int = 0
    timer: Optional[threading.Timer] = None
    mode: str = "voice"
    label: str = "voice"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    picture_beep: bool = False
    slow_cycle: tuple[bytes, ...] = ()
    slow_index: int = 0
    station_message: str = ""
    pacer: FramePacer | None = None
    diagnostics: TransmitDiagnostics | None = None


class ReflectorSession:
    """Manages the UDP connection to a D-Star reflector.

    This class handles the background thread for socket communication,
    parses incoming datagrams, maintains the sequence jitter buffer, and
    coordinates with the AMBE decoder and audio player. It is designed to
    own one reflector socket and one non-reentrant decoder stream.
    """

    def __init__(
        self,
        decoder: AmbeDecoder,
        audio: AudioSink,
        on_event: Optional[Callable[[SessionEvent], None]] = None,
        *,
        config: Optional[BufferConfig] = None,
        app_receiver: Optional[ApplicationReceiver] = None,
        stream_log: BinaryStreamLog | None = None,
    ) -> None:
        self.decoder = decoder
        self.audio = audio
        self._on_event = on_event or (lambda _event: None)
        self.buffer_config = config or load_buffer_config()
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._generation = 0
        self._worker: Optional[_WorkerContext] = None
        self._disconnecting = False
        self._sock: Optional[socket.socket] = None
        self._socket_generation: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[_VoiceStream] = None
        self._interrupted_streams: dict[tuple[Protocol, int], float] = {}
        self._completed_stream: Optional[tuple[Protocol, int, float]] = None
        self._tx: Optional[_TransmitStream] = None
        self._app_receiver = app_receiver
        self.packet_logger = PacketLogger()
        self.stream_log = stream_log
        self.preferences = Preferences()
        self.log_directory = Path("logs")
        self._last_archive_error = ""
        if self._app_receiver is not None:
            self._app_receiver.set_logger(self.packet_logger)
        self._gain = DEFAULT_GAIN

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def transmitting(self) -> bool:
        with self._lifecycle_lock:
            return self._tx is not None

    @property
    def receiving(self) -> bool:
        """Keep the channel busy across gaps and overlapping remote streams."""
        with self._lifecycle_lock:
            now = time.monotonic()
            self._interrupted_streams = {
                key: deadline for key, deadline in self._interrupted_streams.items()
                if deadline > now
            }
            return self._stream is not None or bool(self._interrupted_streams)

    def _require_idle_receiver(self) -> None:
        # Call under the lifecycle lock, before modifying audio or sending headers.
        if self.receiving:
            raise RuntimeError("Reflector is busy receiving. Wait for active streams to end before transmitting.")

    def cancel_application_transmit(
        self, cancel_event: Optional[threading.Event] = None
    ) -> bool:
        """Cancel an application transmission, optionally matching its token.

        A supplied token is signalled before acquiring the session lock so a
        caller can also cancel preparation or header setup in progress. Tokens
        belong to one operation and are never reset by the session.
        """
        if cancel_event is not None:
            cancel_event.set()
        with self._lifecycle_lock:
            if (
                self._tx is not None
                and self._tx.mode == "application"
                and (cancel_event is None or self._tx.cancel_event is cancel_event)
            ):
                self._tx.cancel_event.set()
                return True
        return False

    def enable_packet_logging(self, directory: Path | str | None = None) -> Path:
        """Enable parsed packet and raw UDP logging, returning the parsed path."""
        return self.packet_logger.start(self.log_directory if directory is None else directory)

    def disable_packet_logging(self) -> Path | None:
        """Disable forensic packet logging and close both log files."""
        return self.packet_logger.stop()

    @property
    def raw_packet_log_path(self) -> Path | None:
        """Path of the raw UDP receive log, if logging is enabled."""
        return self.packet_logger.current_raw_path

    def is_packet_logging_enabled(self) -> bool:
        """Return whether forensic packet logging is currently active."""
        return self.packet_logger.is_enabled

    def set_preferences(self, preferences: Preferences) -> None:
        """Snapshot immutable TX options at the start of each transmission."""
        with self._lifecycle_lock:
            self.preferences = preferences

    def connect(self, config: ConnectionConfig) -> None:
        """Start the background thread and connect to the specified reflector.

        This method is non-blocking. It initializes the UDP socket loop,
        sends the initial link datagrams, and sets the connection state.
        """
        normalized = config.normalized()
        target = normalized.target
        stream_filter = TargetStreamFilter(target)
        with self._lifecycle_lock:
            if self._disconnecting:
                raise RuntimeError("the previous reflector session is still stopping")
            if self._worker is not None and not self._worker.stop.is_set():
                raise RuntimeError("a reflector session is already active")
            self.decoder.set_gain(normalized.gain)
            self._gain = normalized.gain
            self._generation += 1
            worker = _WorkerContext(
                generation=self._generation,
                stop=threading.Event(),
                config=normalized,
                target=target,
                stream_filter=stream_filter,
            )
            self._worker = worker
            self._stream = None
            self._interrupted_streams.clear()
            self._completed_stream = None
            if self._app_receiver is not None:
                self._app_receiver.reset()
            thread = threading.Thread(
                target=self._run,
                args=(worker,),
                name="dstar-reflector",
                daemon=True,
            )
            self._thread = thread
            # Keep publication, audio preparation, state, and thread start in
            # one lifecycle transaction.  A simultaneous disconnect cannot
            # invalidate this worker and then have connect revive it afterward.
            try:
                self.audio.retry()
                self.audio.start()
                self._set_state(
                    ConnectionState.CONNECTING,
                    f"Connecting to {target.name} {target.module} "
                    f"via {target.protocol.value}…",
                )
                thread.start()
            except Exception:
                if self._worker is worker:
                    self._worker = None
                if self._thread is thread:
                    self._thread = None
                self._set_state(
                    ConnectionState.ERROR, "Unable to start network worker"
                )
                raise

    def disconnect(self) -> None:
        """Signal the background thread to unlink and close the socket.

        If a transmission is currently active, it is cleanly stopped first.
        This method blocks until the background thread acknowledges the
        disconnect and releases network resources.
        """
        self.stop_transmit(reason="disconnect")
        with self._lifecycle_lock:
            if self._disconnecting:
                return
            self._disconnecting = True
            worker = self._worker
            thread = self._thread
            sock = (
                self._sock
                if worker is not None
                and self._socket_generation == worker.generation
                else None
            )
            if worker is not None:
                worker.stop.set()
            # Invalidate ownership before releasing the lock. Even a worker
            # still blocked in DNS can no longer publish/close a later socket
            # or mutate the new connection's decoder and stream state.
            self._generation += 1
            self._worker = None
            self._interrupted_streams.clear()
            if sock is not None:
                self._sock = None
                self._socket_generation = None

        try:
            if worker is not None or (thread is not None and thread.is_alive()):
                self._set_state(ConnectionState.STOPPING, "Disconnecting…")
            if sock is not None and worker is not None:
                try:
                    self._send_unlink(sock, worker.config, worker.target)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
            with self._lifecycle_lock:
                if self._thread is thread:
                    self._thread = None

            try:
                self._finish_stream("disconnect")
            finally:
                # Flush the device even on the no-worker/early-disconnect path:
                # waveOut may already own blocks no longer in the Python queue.
                self.audio.cancel_stream()
        finally:
            self._set_state(ConnectionState.DISCONNECTED, "Disconnected")
            with self._lifecycle_lock:
                self._disconnecting = False

    def set_gain(self, gain: float) -> None:
        value = float(gain)
        if not 0.0 <= value <= 10.0:
            raise ValueError("audio gain must be between 0 and 10")
        self.decoder.set_gain(value)
        self._gain = value

    def start_transmit(self) -> None:
        """Begin a local microphone stream; callers must keep PTT held."""

        with self._lifecycle_lock:
            if self._tx is not None:
                return
            worker = self._worker
            sock = self._sock
            if (
                self.state is not ConnectionState.LINKED
                or worker is None
                or sock is None
                or self._socket_generation != worker.generation
                or not self._worker_is_current_locked(worker)
            ):
                raise RuntimeError("connect to a reflector before transmitting")
            self._require_idle_receiver()
            self.audio.cancel_stream()
            self.decoder.begin_encode_stream()
            stream_id = secrets.randbelow(0xFFFF) + 1
            header = build_rf_header(
                worker.target,
                worker.config.callsign,
                local_module=worker.config.local_module,
                suffix=self.preferences.voice_suffix,
            )
            header_packet = build_header_packet(worker.target, header, stream_id)
            if header_packet is not None:
                for _ in range(5):
                    self._send(sock, header_packet)
            tx = _TransmitStream(stream_id, header, time.monotonic())
            tx.slow_cycle = station_message_cycle(self.preferences.station_message)
            if self.preferences.send_dprs:
                tx.slow_cycle += slow_data_cycle(gpsa_frame(worker.config.callsign, self.preferences.position))

            def stop_on_timeout() -> None:
                # Timer.cancel() cannot stop a callback already waiting on this
                # lock. Such a callback must not terminate a replacement stream.
                with self._lifecycle_lock:
                    if self._tx is tx:
                        self.stop_transmit(reason="180-second timeout")

            timer = threading.Timer(
                PTT_TIMEOUT_SECONDS,
                stop_on_timeout,
            )
            timer.daemon = True
            tx.timer = timer
            self._tx = tx
            timer.start()
            self._emit_for_worker(
                worker,
                "tx_start",
                f"Transmitting {worker.config.callsign} on "
                f"{worker.target.name} {worker.target.module}",
                stream_id=stream_id,
                timeout_seconds=int(PTT_TIMEOUT_SECONDS),
            )

    def transmit_pcm(self, pcm: bytes) -> bool:
        """Encode and send one 20 ms microphone block while PTT is active."""

        with self._lifecycle_lock:
            tx = self._tx
            worker = self._worker
            sock = self._sock
            if tx is None:
                return False
            if (
                worker is None
                or sock is None
                or self._socket_generation != worker.generation
                or not self._worker_is_current_locked(worker)
            ):
                self.stop_transmit(reason="connection closed")
                return False
            try:
                ambe = self.decoder.encode_frame(pcm)
                slow_data = None
                if tx.slow_cycle and tx.sequence != 0:
                    slow_data = tx.slow_cycle[tx.slow_index % len(tx.slow_cycle)]
                    tx.slow_index += 1
                packet = build_voice_packet(
                    worker.target,
                    tx.header,
                    ambe,
                    tx.sequence,
                    tx.stream_id,
                    packet_counter=tx.packet_counter,
                    slow_data=slow_data,
                )
                self._send(sock, packet)
            except (DecoderError, OSError, ValueError) as exc:
                self._emit_for_worker(worker, "error", f"Transmit failed: {exc}")
                self.stop_transmit(reason="transmit error")
                return False
            tx.sequence = (tx.sequence + 1) % SEQUENCE_MODULUS
            tx.packet_counter = (tx.packet_counter + 1) & 0xFFFFFF
            tx.frames += 1
            if tx.frames % 50 == 0 or tx.frames == 1:
                self._emit_for_worker(
                    worker,
                    "tx_progress",
                    f"Transmitting {tx.frames} frames",
                    callsign=worker.config.callsign,
                    frames=tx.frames,
                )
            return True

    def transmit_application(
        self,
        payload: bytes,
        *,
        label: str = "application data",
        metadata: Optional[Mapping[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Send an RS-MS1 payload, sharing cancellation with its preparation."""

        app_data = bytes(payload)
        if not app_data:
            raise ValueError("application payload is empty")
        details = dict(metadata or {})
        operation_cancel = cancel_event if cancel_event is not None else threading.Event()
        with self._lifecycle_lock:
            if self.preferences.send_dprs and self._worker is not None:
                app_data = gpsa_frame(self._worker.config.callsign, self.preferences.position) + app_data
            worker, sock, tx, payload_frame_count, sync_frame_count = self._prepare_transmit_application(
                app_data, label, details, operation_cancel
            )
        if not worker or not tx:
            return False

        return self._pump_transmit_application_frames(
            worker, tx, app_data, label, details, payload_frame_count
        )

    def _prepare_transmit_application(
        self,
        app_data: bytes,
        label: str,
        details: dict[str, Any],
        cancel_event: threading.Event,
    ) -> tuple[Optional['_WorkerContext'], Optional['socket.socket'], Optional['_TransmitStream'], int, int]:
        """Prepare the application transmit context and emit start events."""
        with self._lifecycle_lock:
            if cancel_event.is_set():
                return None, None, None, 0, 0
            if self._tx is not None:
                raise RuntimeError("another transmission is already active")
            worker = self._worker
            sock = self._sock
            if (
                self.state is not ConnectionState.LINKED
                or worker is None
                or sock is None
                or self._socket_generation != worker.generation
                or not self._worker_is_current_locked(worker)
            ):
                raise RuntimeError("connect to a reflector before transmitting")
            self._require_idle_receiver()
            self.audio.cancel_stream()
            stream_id = secrets.randbelow(0xFFFF) + 1
            is_picture = (
                "picture" in label.lower()
                or ("number" in details or details.get("kind") == "picture")
            )
            rf_suffix = self.preferences.picture_suffix if is_picture else self.preferences.text_suffix
            header = build_rf_header(
                worker.target,
                worker.config.callsign,
                local_module=worker.config.local_module,
                suffix=rf_suffix,
            )
            if cancel_event.is_set():
                return None, None, None, 0, 0
            tx = _TransmitStream(
                stream_id,
                header,
                time.monotonic(),
                mode="application",
                label=label,
                metadata=details,
                cancel_event=cancel_event,
                picture_beep=is_picture and self.preferences.picture_beep,
                station_message=self.preferences.station_message,
            )
            self._tx = tx
            try:
                if not self._send_application_headers(
                    worker, sock, stream_id, header, label, rf_suffix, cancel_event
                ) or cancel_event.is_set():
                    self.stop_transmit(reason="cancelled")
                    return None, None, None, 0, 0
                fast_frame_count = ((len(app_data) + 19) // 20) * 2
                payload_frame_count = fast_frame_count + 8
                sync_frame_count = (payload_frame_count + 19) // 20
                estimated_seconds = (payload_frame_count + sync_frame_count) * _FRAME_INTERVAL_SECONDS
                if tx.station_message:
                    estimated_seconds += 21 * _FRAME_INTERVAL_SECONDS
                if tx.picture_beep:
                    estimated_seconds /= 0.88
                self._emit_for_worker(
                    worker,
                    "app_tx_start",
                    f"Sending {label} ({len(app_data):,} bytes, about {estimated_seconds:.1f}s)",
                    stream_id=stream_id,
                    label=label,
                    bytes=len(app_data),
                    estimated_seconds=estimated_seconds,
                    metadata=details,
                )
                return worker, sock, tx, payload_frame_count, sync_frame_count
            except Exception:
                self.stop_transmit(reason="transmit failed")
                raise

    def _send_application_headers(
        self,
        worker: _WorkerContext,
        sock: socket.socket,
        stream_id: int,
        header: bytes,
        label: str,
        rf_suffix: str,
        cancel_event: threading.Event,
    ) -> bool:
        """Send the D-STAR RF headers for a new application stream."""
        header_packet = build_header_packet(worker.target, header, stream_id)
        if header_packet is not None:
            for _ in range(5):
                if cancel_event.is_set():
                    return False
                self._send(sock, header_packet)
            if self.packet_logger.is_enabled:
                self.packet_logger.log_tx(
                    worker.target.protocol.value,
                    stream_id,
                    0,
                    0,
                    "HEADER",
                    details=f"sent 5 RF headers for {label} (suffix={rf_suffix})",
                )
        return not cancel_event.is_set()

    def _pump_transmit_application_frames(
        self, worker: '_WorkerContext', tx: '_TransmitStream', app_data: bytes, label: str, details: dict[str, Any], payload_frame_count: int
    ) -> bool:
        """Pump fast-data chunks onto the socket with accurate timing."""
        payloads = application_frames(app_data, tx.picture_beep, tx.station_message)
        sent_payload_frames = 0
        completed = False
        windows_timer_set = False
        if sys.platform == "win32":
            try:
                windows_timer_set = ctypes.windll.winmm.timeBeginPeriod(1) == 0
            except Exception:
                pass
        try:
            tx.diagnostics = TransmitDiagnostics(self.packet_logger)
            tx.pacer = FramePacer(clock=time.perf_counter, sleep=time.sleep)
            for dv_payload, counts_progress in payloads:
                if tx.cancel_event.is_set():
                    break
                if not self._pump_single_application_frame(tx, dv_payload):
                    return False
                if counts_progress:
                    sent_payload_frames += 1
                    if sent_payload_frames == payload_frame_count or sent_payload_frames % 20 == 0:
                        self._emit_for_worker(
                            worker,
                            "app_tx_progress",
                            f"Sending {label}: {sent_payload_frames}/{payload_frame_count} data frames",
                            label=label,
                            sent=sent_payload_frames,
                            total=payload_frame_count,
                            percent=round(sent_payload_frames * 100.0 / payload_frame_count, 1),
                            metadata=details,
                        )
            if not tx.cancel_event.is_set():
                tx.pacer.wait_end()
            with self._lifecycle_lock:
                completed = self._tx is tx and not tx.cancel_event.is_set()
            return completed
        except (OSError, ValueError) as exc:
            self._emit_for_worker(worker, "error", f"{label} transmit failed: {exc}")
            return False
        finally:
            try:
                if tx.cancel_event.is_set():
                    self._emit_for_worker(
                        worker, "app_tx_cancel", f"Cancelled {label}",
                        label=label, metadata=details,
                    )
            finally:
                self._cleanup_transmit_application(windows_timer_set, tx, completed)

    def _pump_single_application_frame(
        self, tx: _TransmitStream, dv_payload: bytes
    ) -> bool:
        """Send a single application frame, handling sync sequences and sleep pacing."""
        if tx.sequence == 0:
            sync_payload = (dv_payload[:9] + bytes.fromhex("55 2d 16")) if dv_payload in PICTURE_BEEP else FAST_SYNC_DV_PAYLOAD
            if not self._send_application_slot(tx, sync_payload, sync=True):
                return False
        return self._send_application_slot(tx, dv_payload)

    def _cleanup_transmit_application(
        self, windows_timer_set: bool, tx: '_TransmitStream', completed: bool
    ) -> None:
        """End the stream before restoring timers or draining diagnostics."""
        try:
            with self._lifecycle_lock:
                if self._tx is tx:
                    reason = "cancelled" if tx.cancel_event.is_set() else ("complete" if completed else "transmit stopped")
                    self.stop_transmit(reason=reason)
        finally:
            if windows_timer_set:
                try:
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
            if tx.diagnostics is not None:
                # Disk flushes must never run under the session lifecycle lock.
                tx.diagnostics.close()
                if tx.pacer is not None:
                    # Write after draining so queue overflow cannot hide the
                    # timing summary needed to diagnose an overloaded machine.
                    self.packet_logger.log_event(
                        "app_tx_timing", "Application socket-send timing (not remote delivery)",
                        stream_id=tx.stream_id, **tx.pacer.summary(),
                    )

    def _send_application_slot(
        self,
        tx: _TransmitStream,
        dv_payload: bytes,
        *,
        sync: bool = False,
    ) -> bool:
        sleep_ms = tx.pacer.wait()
        with self._lifecycle_lock:
            worker = self._worker
            sock = self._sock
            if (
                self._tx is not tx
                or tx.cancel_event.is_set()
                or worker is None
                or sock is None
                or self._socket_generation != worker.generation
                or not self._worker_is_current_locked(worker)
            ):
                return False
            pkt = build_voice_packet(
                worker.target,
                tx.header,
                dv_payload[:9],
                tx.sequence,
                tx.stream_id,
                packet_counter=tx.packet_counter,
                slow_data=None if sync else dv_payload[9:],
            )
            if tx.cancel_event.is_set():
                return False
            self._send(sock, pkt)
            sent_at = time.perf_counter()
            tx.pacer.sent(sent_at)
            if tx.diagnostics is not None:
                tx.diagnostics.log_tx(
                    worker.target.protocol.value,
                    tx.stream_id,
                    tx.sequence,
                    tx.packet_counter,
                    "VOICE",
                    dv_payload=dv_payload,
                    is_sync=sync,
                    sleep_ms=sleep_ms,
                    sent_at=sent_at,
                )
            tx.sequence = (tx.sequence + 1) % SEQUENCE_MODULUS
            tx.packet_counter = (tx.packet_counter + 1) & 0xFFFFFF
            tx.frames += 1
            return True

    def stop_transmit(self, *, reason: str = "PTT released") -> None:
        """End the current stream immediately and send its protocol terminator."""

        with self._lifecycle_lock:
            tx = self._tx
            if tx is None:
                return
            self._tx = None
            if tx.timer is not None:
                tx.timer.cancel()
            worker = self._worker
            sock = self._sock
            if (
                worker is not None
                and sock is not None
                and self._socket_generation == worker.generation
                and self._worker_is_current_locked(worker)
            ):
                try:
                    end_packet = build_voice_packet(
                        worker.target,
                        tx.header,
                        None,
                        tx.sequence,
                        tx.stream_id,
                        end=True,
                        packet_counter=tx.packet_counter,
                    )
                    for _ in range(3):
                        self._send(sock, end_packet)
                    logger = tx.diagnostics if tx.diagnostics is not None else self.packet_logger
                    logger.log_tx(
                        worker.target.protocol.value,
                        tx.stream_id,
                        tx.sequence,
                        tx.packet_counter,
                        "END",
                        details=f"reason={reason}",
                    )
                except OSError:
                    pass
                duration = max(0.0, time.monotonic() - tx.started_at)
                if tx.mode == "application":
                    kind = "app_tx_end"
                    message = (
                        f"{tx.label.capitalize()} transmission ended after "
                        f"{duration:.1f}s / {tx.frames} DV frames ({reason})"
                    )
                else:
                    kind = "tx_timeout" if reason == "180-second timeout" else "tx_end"
                    message = (
                        f"Transmission ended after {duration:.1f}s / "
                        f"{tx.frames} frames ({reason})"
                    )
                self._emit_for_worker(
                    worker,
                    kind,
                    message,
                    stream_id=tx.stream_id,
                    frames=tx.frames,
                    duration_seconds=duration,
                    reason=reason,
                    label=tx.label,
                    metadata=dict(tx.metadata),
                )

    def _run(self, worker: _WorkerContext) -> None:
        reconnect_attempts = 0
        max_reconnects = 5
        was_linked = False

        while self._worker_is_current(worker) and not worker.stop.is_set():
            if reconnect_attempts > 0:
                self._set_state_for_worker(
                    worker,
                    ConnectionState.CONNECTING,
                    f"Reconnecting to {worker.target.name} {worker.target.module} "
                    f"via {worker.target.protocol.value}… (Attempt {reconnect_attempts}/{max_reconnects})",
                )

            try:
                self._run_attempt(worker)
            except Exception as exc:
                if worker.stop.is_set() or not self._worker_is_current(worker):
                    break

                if self.state == ConnectionState.LINKED:
                    was_linked = True

                if was_linked and reconnect_attempts < max_reconnects:
                    reconnect_attempts += 1
                    self._emit_for_worker(
                        worker, "warning",
                        f"Connection dropped: {exc}. Reconnecting in 3 seconds..."
                    )

                    for _ in range(30):
                        if worker.stop.is_set():
                            break
                        time.sleep(0.1)

                    continue
                else:
                    self._set_state_for_worker(
                        worker, ConnectionState.ERROR, f"Connection failed: {exc}"
                    )
                    break
            else:
                break

        with self._lifecycle_lock:
            if self._worker is worker:
                self._worker = None
            if self._thread is threading.current_thread():
                self._thread = None

    def _run_attempt(self, worker: _WorkerContext) -> None:
        config = worker.config
        target = worker.target
        sock: Optional[socket.socket] = None
        try:
            remote_ip = socket.gethostbyname(config.host)
            if not self._worker_is_current(worker):
                return
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            local_port = target.port if config.prefer_standard_local_port else 0
            try:
                sock.bind(("", local_port))
            except OSError as exc:
                if not local_port:
                    raise
                sock.bind(("", 0))
                self._emit_for_worker(
                    worker,
                    "warning",
                    f"UDP {local_port} was unavailable ({exc}); using local port "
                    f"{sock.getsockname()[1]}",
                )
            sock.connect((remote_ip, target.port))
            sock.setblocking(False)
            with self._lifecycle_lock:
                if not self._worker_is_current_locked(worker):
                    sock.close()
                    return
                self._sock = sock
                self._socket_generation = worker.generation
            self._emit_for_worker(
                worker,
                "network",
                f"UDP {sock.getsockname()[1]} → {remote_ip}:{target.port}",
                host=remote_ip,
                port=target.port,
            )
            self._connection_loop(sock, worker)
        finally:
            # A disconnected worker may still be unwinding a blocked DNS or
            # socket call after a replacement connection has started sending.
            with self._lifecycle_lock:
                if self._worker_is_current_locked(worker):
                    self.stop_transmit(reason="connection closed")
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            with self._lifecycle_lock:
                current = self._worker_is_current_locked(worker)
                if (
                    self._sock is sock
                    and self._socket_generation == worker.generation
                ):
                    self._sock = None
                    self._socket_generation = None
            if current:
                self._finish_stream(
                    "connection closed", generation=worker.generation
                )

    def _connection_loop(
        self, sock: socket.socket, worker: _WorkerContext
    ) -> None:
        protocol = worker.target.protocol
        now = time.monotonic()
        link_started = now
        next_link = now
        retry_delay = 1.0
        next_poll = now + self._poll_interval(protocol)
        last_valid_packet = now

        while self._worker_is_current(worker):
            now = time.monotonic()
            state = self.state
            is_tx = self.transmitting

            next_link, retry_delay, next_poll = self._service_link_state(
                sock, worker, protocol, state, is_tx, now, link_started, next_link, retry_delay, next_poll, last_valid_packet
            )

            if is_tx:
                last_valid_packet = now

            last_valid_packet = self._wait_and_drain_socket(sock, worker, now, last_valid_packet)

            now = time.monotonic()
            self._service_stream_timeouts(worker, now)

    def _service_link_state(
        self, sock: socket.socket, worker: _WorkerContext, protocol: Protocol, state: ConnectionState,
        is_tx: bool, now: float, link_started: float, next_link: float, retry_delay: float, next_poll: float, last_valid_packet: float
    ) -> tuple[float, float, float]:
        """Handle protocol keep-alives and link establishment timeouts."""
        if state == ConnectionState.CONNECTING and now >= next_link:
            self._send_link(sock, worker)
            next_link = now + retry_delay
            if protocol is not Protocol.DPLUS:
                retry_delay = min(retry_delay * 2.0, 8.0)
        if state == ConnectionState.CONNECTING and now - link_started > 15.0:
            raise TimeoutError("reflector did not acknowledge the link")
        if state == ConnectionState.LINKED and now >= next_poll:
            if not is_tx and protocol is not Protocol.DCS:
                self._send_poll(sock, worker)
            next_poll = now + self._poll_interval(protocol)
        timeout = 30.0 if protocol is Protocol.DPLUS else 60.0
        if not is_tx and state == ConnectionState.LINKED and now - last_valid_packet > timeout:
            raise TimeoutError("reflector stopped responding")
        return next_link, retry_delay, next_poll

    def _wait_and_drain_socket(
        self, sock: socket.socket, worker: _WorkerContext, now: float, last_valid_packet: float
    ) -> float:
        """Poll the UDP socket and drain all waiting datagrams."""
        try:
            readable, _, _ = select.select([sock], [], [], self._receive_wait(now))
        except (OSError, ValueError):
            if worker.stop.is_set() or not self._worker_is_current(worker):
                return last_valid_packet
            raise

        if readable:
            for _ in range(_MAX_DATAGRAMS_PER_CYCLE):
                try:
                    data, source = sock.recvfrom(_UDP_RECEIVE_BYTES)
                except BlockingIOError:
                    break
                except OSError:
                    if worker.stop.is_set() or not self._worker_is_current(worker):
                        return last_valid_packet
                    raise
                # Capture before parsing, filtering, or skipping empty datagrams.
                self.packet_logger.log_raw_rx(
                    worker.target.protocol.value,
                    data,
                    source=source,
                    reflector=worker.target.name,
                    module=worker.target.module,
                    connection=worker.generation,
                )
                if data:
                    last_valid_packet = time.monotonic()
                    self._handle_datagram(data, sock, worker)
                elif self.stream_log is not None:
                    self.stream_log.append(data, worker.target.protocol)
        return last_valid_packet

    def _service_stream_timeouts(self, worker: _WorkerContext, now: float) -> None:
        """Check for stream starvation and service active audio playout."""
        with self._lifecycle_lock:
            stream = self._stream
            timeout_limit = (
                30.0
                if (stream is not None and stream.application_data)
                else self.buffer_config.voice_timeout_seconds
            )
            timed_out = (
                stream is not None
                and now - stream.last_media_at > timeout_limit
            )
        if timed_out:
            self._finish_stream(
                "data timeout" if (stream and stream.application_data) else "voice timeout",
                generation=worker.generation,
            )
        else:
            self._service_playout(now, generation=worker.generation)
    def _receive_wait(self, now: float) -> float:
        """Bound select() by the next 20 ms playout or rebuffer deadline."""

        with self._lifecycle_lock:
            stream = self._stream
            if stream is None or stream.application_data:
                return 0.1
            deadline = (
                stream.next_playout_at
                if stream.next_playout_at is not None
                else stream.rebuffer_until
            )
        if deadline is None:
            return 0.1
        return max(0.0, min(0.1, deadline - now))

    @staticmethod
    def _poll_interval(protocol: Protocol) -> float:
        if protocol is Protocol.DPLUS:
            return 1.0
        if protocol is Protocol.DEXTRA:
            return 10.0
        return 60.0

    def _handle_datagram(
        self, data: bytes, sock: socket.socket, worker: _WorkerContext
    ) -> bool:
        if not self._worker_is_current(worker):
            return False
        protocol = worker.target.protocol
        record = self._record_datagram(data, worker)
        control = parse_control_packet(data, protocol)
        if control is not None:
            return self._handle_control_packet(control, sock, worker)

        media = parse_media_packet(data, protocol)
        if media is None:
            return False
        if self.state is not ConnectionState.LINKED:
            return True
        if not worker.stream_filter.accept(media):
            if (
                media.kind is not MediaKind.VOICE
                or not worker.stream_filter.admit_stream(media.stream_id)
            ):
                return True
        self._handle_media(media, worker, record)
        return True

    def _record_datagram(self, data: bytes, worker: _WorkerContext) -> RecordRef | None:
        """Archive every received datagram and report each recorder failure once."""
        if self.stream_log is None:
            return None
        record = self.stream_log.append(data, worker.target.protocol)
        if self.stream_log.error != self._last_archive_error:
            self._last_archive_error = self.stream_log.error
            if self._last_archive_error:
                self._emit_for_worker(
                    worker, "warning", f"Binary recording stopped: {self._last_archive_error}"
                )
        return record

    def _handle_control_packet(
        self, control: ControlPacket, sock: socket.socket, worker: _WorkerContext
    ) -> bool:
        """Apply a correlated link reply or keepalive to the connection handshake."""
        protocol = worker.target.protocol
        if not self._control_matches_worker(control, worker):
            self._emit_for_worker(
                worker,
                "warning",
                f"Ignored {protocol.value} {control.kind.value} for another client",
            )
            return False
        if control.kind is ControlKind.LINK and protocol is Protocol.DPLUS:
            self._send(sock, build_login_packet(worker.config.callsign))
            self._emit_for_worker(worker, "network", "D-Plus login requested")
        elif control.kind is ControlKind.ACK:
            self._link_complete(worker)
        elif control.kind is ControlKind.NAK:
            reason = control.reply or "link rejected"
            raise ConnectionError(str(reason))
        elif control.kind is ControlKind.UNLINK:
            raise ConnectionError("reflector unlinked this client")
        elif control.kind is ControlKind.POLL:
            # Only the 22-byte server form solicits a DCS client response;
            # replying to an echoed 17-byte client poll can create a loop.
            if protocol is Protocol.DCS and len(control.raw) == 22:
                self._send_poll(sock, worker)
            if self.state == ConnectionState.CONNECTING and protocol is not Protocol.DPLUS:
                self._link_complete(worker)
        return True

    def _link_complete(self, worker: _WorkerContext) -> None:
        if (
            not self._worker_is_current(worker)
            or self.state is not ConnectionState.CONNECTING
        ):
            return
        note = ""
        if worker.target.protocol is Protocol.DPLUS:
            note = " (D-Plus login is reflector-wide; incoming headers identify traffic)"
        self._set_state_for_worker(
            worker,
            ConnectionState.LINKED,
            f"Linked to {worker.target.name} {worker.target.module}{note}",
        )

    @staticmethod
    def _control_matches_worker(
        control: ControlPacket, worker: _WorkerContext
    ) -> bool:
        """Correlate identity-bearing replies with the attempted link.

        Several clients can share a standard UDP port on a host, and stale
        datagrams can survive a rapid reconnect.  Treating an unrelated ACK or
        poll as ours would make the UI say "Linked" without a valid session.
        D-Plus control replies contain no station identity, so source filtering
        by the connected UDP socket is the strongest correlation available.
        """

        target = worker.target
        if control.protocol is not target.protocol:
            return False
        if target.protocol is Protocol.DPLUS:
            return True

        local_call = worker.config.callsign[:7].rstrip()
        local_module = worker.config.local_module

        if target.protocol is Protocol.DEXTRA and control.kind is ControlKind.POLL:
            # Reflectors in the field use either the client base or their own
            # base in the otherwise identity-light nine-byte keepalive.
            return control.local_callsign in {local_call, target.wire_name}

        if target.protocol is Protocol.DCS and control.kind is ControlKind.POLL:
            # A 17-byte packet is the client form (and can be our own echo).
            # Only a fully correlated 22-byte server poll solicits a response.
            return (
                len(control.raw) == 22
                and control.local_callsign == local_call
                and control.local_module == local_module
                and control.reflector == target.wire_name
                and control.reflector_module == target.module
            )

        if control.local_callsign != local_call:
            return False
        if (
            control.local_module is not None
            and control.local_module != local_module
        ):
            return False
        if (
            control.reflector is not None
            and control.reflector != target.wire_name
        ):
            return False
        if control.kind in {ControlKind.ACK, ControlKind.LINK}:
            return control.reflector_module == target.module
        return control.reflector_module in {None, target.module}

    def _handle_media(
        self, packet: MediaPacket, worker: _WorkerContext, record: RecordRef | None = None
    ) -> None:
        with self._lifecycle_lock:
            if not self._worker_is_current_locked(worker):
                return
            if self._tx is not None:
                return
            now = time.monotonic()

            if packet.kind is MediaKind.END:
                self._interrupted_streams.pop((packet.protocol, packet.stream_id), None)
                if self.stream_log is not None and self._stream is not None and self._stream.stream_id == packet.stream_id:
                    self.stream_log.include(self._stream.recording, record)
                self._handle_media_end(packet, worker, now)
                return

            self._sync_stream_header(packet, worker, now)
            if self.stream_log is not None and self._stream is not None and self._stream.stream_id == packet.stream_id:
                self.stream_log.include(self._stream.recording, record, voice=packet.kind is MediaKind.VOICE)

            if packet.kind is MediaKind.HEADER:
                return
            if self._stream is None or self._stream.stream_id != packet.stream_id:
                return
            if packet.ambe is None or packet.sequence is None:
                return

            stream = self._stream
            if self.packet_logger.is_enabled and packet.dv_payload is not None:
                self.packet_logger.log_rx(
                    packet.protocol.value,
                    packet.stream_id,
                    packet.sequence,
                    packet.kind.value,
                    packet.dv_payload,
                )

            self._process_application_data(stream, packet, worker, now)

    def _handle_media_end(self, packet: MediaPacket, worker: _WorkerContext, now: float) -> None:
        if self.packet_logger.is_enabled:
            self.packet_logger.log_rx(
                packet.protocol.value,
                packet.stream_id,
                packet.sequence,
                "END",
                packet.dv_payload,
            )
        if self._app_receiver is not None:
            end_events = self._app_receiver.end_stream(packet.stream_id)
            for app_event in end_events:
                self._emit_for_worker(worker, app_event.kind, app_event.message, **app_event.fields)

        if self._stream and self._stream.stream_id == packet.stream_id:
            if self._stream.application_data:
                self._finish_stream("end packet", generation=worker.generation)
            else:
                self._mark_stream_end(packet.sequence, now)

    def _sync_stream_header(self, packet: MediaPacket, worker: _WorkerContext, now: float) -> None:
        incoming_header = packet.header
        if incoming_header is None and packet.kind is MediaKind.VOICE:
            synth_raw = build_rf_header(worker.target, "UNKNOWN", local_module=" ")
            incoming_header = parse_rf_header(synth_raw)

        # Buffer starvation and repeated headers do not create transmissions.
        # Keep the ID's parser, decoder, and queued tail through playout/END.
        if incoming_header is not None and (
            self._stream is None
            or self._stream.stream_id != packet.stream_id
        ):
            if self._recently_completed(packet, now):
                return
            self._start_stream(packet.stream_id, incoming_header, packet.protocol, now, worker)
        elif (
            packet.header is not None
            and self._stream is not None
            and self._stream.stream_id == packet.stream_id
            and self._stream.header.my_call == "UNKNOWN"
        ):
            self._stream.header = packet.header

    def _process_application_data(self, stream: _VoiceStream, packet: MediaPacket, worker: _WorkerContext, now: float) -> None:
        claimed, app_events = False, []
        if self._app_receiver is not None:
            claimed, app_events = self._app_receiver.feed(
                packet.stream_id, packet.sequence, packet.dv_payload or b""
            )

        for app_event in app_events:
            self._emit_for_worker(worker, app_event.kind, app_event.message,
                                  **{"callsign": stream.header.formatted_my_call,
                                     "stream_id": stream.stream_id, **app_event.fields})
            if stream.header.my_call == "UNKNOWN" and "source" in app_event.fields:
                source_call = str(app_event.fields["source"]).strip().upper()[:8]
                if source_call:
                    try:
                        synth_raw = build_rf_header(worker.target, source_call, local_module=" ")
                        stream.header = parse_rf_header(synth_raw)
                    except (ValueError, TypeError):
                        pass

        if claimed:
            if not stream.application_data:
                stream.application_data = True
                stream.parked_at = None
                self.audio.cancel_stream()
                self._emit_for_worker(
                    worker,
                    "app_stream",
                    f"Receiving RS-MS1 data from {stream.header.my_call or 'Unknown'}",
                    callsign=stream.header.formatted_my_call,
                    stream_id=stream.stream_id,
                )
            stream.last_media_at = now
            return

        if stream.application_data:
            stream.last_media_at = now
            return

        self._accept_voice_frame(packet.sequence, packet.ambe, now)
        if self._stream is stream:
            stream.last_media_at = now

    def _recently_completed(self, packet: MediaPacket, now: float) -> bool:
        completed = self._completed_stream
        if completed is None:
            return False
        protocol, stream_id, completed_at = completed
        if now - completed_at > _COMPLETED_STREAM_HOLDOFF_SECONDS:
            self._completed_stream = None
            return False
        return protocol is packet.protocol and stream_id == packet.stream_id

    def _start_stream(
        self,
        stream_id: int,
        header: RFHeader,
        protocol: Protocol,
        now: float,
        worker: _WorkerContext,
    ) -> None:
        if not self._worker_is_current_locked(worker):
            return
        if self._stream is not None:
            self._finish_stream("superseded", generation=worker.generation)
        self._interrupted_streams.pop((protocol, stream_id), None)
        seed = secrets.randbits(32)
        self.decoder.begin_stream(seed)
        generation = self.audio.begin_stream()
        if self._app_receiver is not None:
            self._app_receiver.start_stream(stream_id)
        self._stream = _VoiceStream(
            stream_id=stream_id,
            header=header,
            protocol=protocol,
            audio_generation=generation,
            started_at=now,
            last_media_at=now,
            recording=self.stream_log.begin(stream_id, protocol) if self.stream_log is not None else None,
        )
        suffix = f"/{header.my_suffix}" if header.my_suffix else ""
        self._emit_for_worker(
            worker,
            "stream_start",
            f"{header.formatted_my_call} → {header.your_call or 'CQCQCQ'} "
            f"via {header.rpt2 or header.rpt1}  [stream {stream_id:04X}]",
            callsign=header.formatted_my_call,
            suffix=header.my_suffix,
            destination=header.your_call,
            rpt1=header.rpt1,
            rpt2=header.rpt2,
            stream_id=stream_id,
            protocol=protocol.value,
            recording=self._stream.recording,
        )

    def _accept_voice_frame(
        self, sequence: int, ambe: bytes, now: Optional[float] = None
    ) -> bool:
        """Place one frame in the bounded-delay sequence reorder buffer.

        This handles sequence numbers arriving out of order from the UDP socket
        by unwrapping the 0-20 sequence modulus into an absolute stream index,
        discarding hopelessly late frames.
        """

        stream = self._stream
        if stream is None:
            return False
        received_at = time.monotonic() if now is None else now
        position = self._locate_voice_frame(stream, sequence, ambe, received_at)
        if position is None:
            return False
        index, reordered = position
        self._resume_delayed_voice(stream, index, sequence)

        if stream.playout_index is not None and index < stream.playout_index:
            stream.late_frames += 1
            return False

        if stream.end_index is not None and index >= stream.end_index:
            stream.late_frames += 1
            return False
        if stream.highest_sequence is None or index > stream.highest_index:
            stream.highest_sequence = sequence
            stream.highest_index = index
            stream.highest_received_at = received_at
        stream.frame_buffer[index] = ambe
        stream.received_frames += 1
        if stream.received_frames % 50 == 0 or stream.received_frames == 1:
            self._emit(
                "voice_progress",
                f"{stream.header.my_call}: {stream.received_frames} frames",
                callsign=stream.header.formatted_my_call,
                frames=stream.received_frames,
                stream_id=stream.stream_id,
            )
        if reordered:
            stream.reordered_frames += 1
        self._score_voice_order(stream, ambe)
        self._trim_voice_buffer(stream)
        if self._stream is not stream:
            return False

        if stream.next_playout_at is None and stream.rebuffer_until is None:
            delay = (
                self.buffer_config.startup_delay_seconds
                if stream.playout_index is None
                else self.buffer_config.resume_delay_seconds
            )
            stream.rebuffer_until = received_at + delay
            stream.parked_at = None
        return True

    def _locate_voice_frame(
        self, stream: _VoiceStream, sequence: int, ambe: bytes, received_at: float
    ) -> tuple[int, bool] | None:
        """Resolve a wire sequence and retain distinct audio in an occupied slot.

        A repeated payload is a duplicate. A different payload can indicate a
        sender restarting its short counter, so anchor it after buffered speech
        while keeping the same transmission and decoder history.
        """
        index, reordered = self._extend_sequence(stream, sequence, received_at)
        buffered = stream.frame_buffer.get(index)
        if buffered is None:
            return index, reordered
        if buffered == ambe:
            stream.duplicate_frames += 1
            return None
        if stream.end_index is not None:
            stream.late_frames += 1
            return None

        index = stream.highest_index + 1
        stream.sequence_conflicts += 1
        self._emit(
            "stream_recovery",
            f"{stream.header.my_call or 'Unknown'}: retained different audio with a repeated sequence",
            stream_id=stream.stream_id,
            sequence=sequence,
            sequence_conflicts=stream.sequence_conflicts,
        )
        return index, False

    def _resume_delayed_voice(self, stream: _VoiceStream, index: int, sequence: int) -> None:
        """Rewind only concealed silence when delayed speech arrives after a gap."""
        if (
            (stream.parked_at is not None or stream.concealment_run > 0)
            and not stream.frame_buffer
            and stream.end_index is None
            and stream.playout_index is not None
            and stream.highest_index < index < stream.playout_index
        ):
            # Never replay received audio: only the concealed portion after
            # highest_index can be replaced by this delayed source timeline.
            stream.playout_index = index
            stream.next_playout_at = None
            stream.rebuffer_until = None
            stream.delayed_resumes += 1
            self._emit(
                "stream_recovery",
                f"{stream.header.my_call or 'Unknown'}: resumed delayed packets in the same stream",
                stream_id=stream.stream_id,
                sequence=sequence,
                delayed_resumes=stream.delayed_resumes,
            )

    def _score_voice_order(self, stream: _VoiceStream, ambe: bytes) -> None:
        """Accumulate the initial FEC evidence before selecting the AMBE bit order."""
        if stream.order is not None or stream.scored_frames >= _AUTO_ORDER_FRAMES:
            return
        msb = self.decoder.score_frame(ambe, BitOrder.MSB_FIRST)
        lsb = self.decoder.score_frame(ambe, BitOrder.LSB_FIRST)
        stream.msb_cost += msb.cost
        stream.lsb_cost += lsb.cost
        stream.scored_frames += 1
        if stream.scored_frames >= _AUTO_ORDER_FRAMES:
            self._select_bit_order()

    def _trim_voice_buffer(self, stream: _VoiceStream) -> None:
        """Bound suspend/burst latency by retaining only the newest configured frames."""

        overflow = len(stream.frame_buffer) - self.buffer_config.max_buffered_frames
        if overflow > 0:
            stale = sorted(stream.frame_buffer)[:overflow]
            for index in stale:
                del stream.frame_buffer[index]
            stream.buffer_dropped_frames += len(stale)
            first_retained = min(stream.frame_buffer)
            if (
                stream.playout_index is not None
                and stream.playout_index < first_retained
            ):
                stream.playout_index = first_retained
                stream.concealment_run = 0
                stream.concealment_reset = False
                stream.decoder_reset_pending = True
                if not stream.buffer_overflow_active:
                    stream.buffer_overflow_active = True
                    stream.resync_events += 1
                    self._emit(
                        "stream_recovery",
                        f"{stream.header.my_call or 'Unknown'}: trimmed stale "
                        "buffered audio to stay live",
                        callsign=stream.header.formatted_my_call,
                        stream_id=stream.stream_id,
                        buffer_dropped=len(stale),
                    )
        stream.peak_buffered_frames = max(
            stream.peak_buffered_frames, len(stream.frame_buffer)
        )

    def _extend_sequence(
        self,
        stream: _VoiceStream,
        sequence: int,
        received_at: float,
        *,
        tolerance_frames: Optional[int] = None,
    ) -> tuple[int, bool]:
        """Map a modulo-21 sequence onto a monotonic stream-frame index.

        The nearest elapsed-time-consistent cycle disambiguates the tiny wire
        sequence space after long outages and complete 21-frame wraps. If no
        cycle is plausible within tolerance frame times, the modulo half-space rule
        wins: distances 1..10 are forward and 11..20 are reordered/late.
        """

        if stream.highest_sequence is None:
            return 0, False

        forward = sequence_distance(stream.highest_sequence, sequence)
        if (stream.parked_at is not None or stream.concealment_run > 0) and forward == 1:
            # Arrival time cannot distinguish queued UDP from a whole lost
            # cycle. Preserve consecutive source sequence after a pause;
            # inferring extra cycles here manufactures loss from network delay.
            return stream.highest_index + forward, False
        anchor_time = (
            received_at
            if stream.highest_received_at is None
            else stream.highest_received_at
        )
        elapsed_slots = int(
            max(
                0.0,
                received_at - anchor_time,
            )
            / _FRAME_INTERVAL_SECONDS
        )
        allow_new_cycle = (
            stream.playout_index is not None
            and stream.playout_index > stream.highest_index
        )
        tolerance = (
            tolerance_frames
            if tolerance_frames is not None
            else self.buffer_config.sequence_tolerance_frames
        )
        delta = self._sequence_delta(forward, elapsed_slots, tolerance, allow_new_cycle)
        return stream.highest_index + delta, delta < 0

    @staticmethod
    def _sequence_delta(
        forward: int, elapsed_slots: int, tolerance: int, allow_new_cycle: bool
    ) -> int:
        """Choose the wire-sequence cycle closest to elapsed source-frame time.

        Advancing a whole cycle is permitted only after playout passes the last
        received frame; otherwise the same counter can still be a queued copy.
        """
        max_cycle = max(1, (elapsed_slots // SEQUENCE_MODULUS) + 1)
        candidates = [
            forward + cycle * SEQUENCE_MODULUS
            for cycle in range(max_cycle + 1)
        ]
        if forward:
            candidates.append(forward - SEQUENCE_MODULUS)
        plausible = [
            delta
            for delta in candidates
            if abs(delta - elapsed_slots) <= tolerance
            and (delta < SEQUENCE_MODULUS or allow_new_cycle)
        ]
        if plausible:
            return min(
                plausible,
                key=lambda value: (abs(value - elapsed_slots), value < 0),
            )
        if forward <= SEQUENCE_MODULUS // 2 or (
            forward > 0 and elapsed_slots >= forward - tolerance
        ):
            # A long pause followed by a short burst need not consume the
            # entire elapsed wall-clock interval in source frames. The first
            # forward cycle may be late even when no cycle is close in time.
            return forward
        return forward - SEQUENCE_MODULUS

    def _mark_stream_end(self, sequence: Optional[int], now: float) -> None:
        stream = self._stream
        if stream is None or stream.end_index is not None:
            return

        floor = stream.playout_index
        if stream.frame_buffer:
            buffered_floor = max(stream.frame_buffer) + 1
            floor = buffered_floor if floor is None else max(floor, buffered_floor)

        end_index = floor
        if sequence is not None and stream.highest_sequence is not None:
            forward = sequence_distance(stream.highest_sequence, sequence)
            if 1 <= forward <= SEQUENCE_MODULUS // 2:
                candidate = stream.highest_index + forward
            else:
                # A repeated or late terminator is authoritative but cannot
                # justify a speculative extra 21-frame tail.
                candidate = stream.highest_index + 1
            end_index = candidate if end_index is None else max(end_index, candidate)
        stream.end_index = end_index
        stream.end_reason = "end packet"
        stream.last_media_at = now

        if not stream.frame_buffer and stream.playout_index is None:
            self._finish_stream(stream.end_reason)

    def _select_bit_order(self) -> None:
        stream = self._stream
        if stream is None or stream.order is not None:
            return
        # Match the native decoder CLI's deterministic D-Star default.  Real
        # speech FEC usually separates the candidates within five frames; a
        # tie (notably null/silence frames) must still resolve predictably.
        stream.order = (
            BitOrder.LSB_FIRST
            if stream.lsb_cost <= stream.msb_cost
            else BitOrder.MSB_FIRST
        )
        self._emit(
            "decoder",
            f"AMBE order {stream.order.label} "
            f"(FEC cost MSB {stream.msb_cost}, LSB {stream.lsb_cost})",
            bit_order=stream.order.name,
            msb_cost=stream.msb_cost,
            lsb_cost=stream.lsb_cost,
        )

    def _service_playout(
        self, now: Optional[float] = None, *, generation: Optional[int] = None
    ) -> None:
        """Emit due frames at 50 fps, reordering first and concealing at deadline.

        This loop continuously checks if the playout time for the next sequence
        index has arrived. If a frame hasn't arrived by its scheduled playout
        deadline, the decoder conceals it by generating a silence frame to maintain
        continuous audio feeding.
        """

        service_at = time.monotonic() if now is None else now
        with self._lifecycle_lock:
            if generation is not None and (
                self._worker is None
                or self._worker.generation != generation
                or self._generation != generation
            ):
                return

            serviced = 0
            while serviced < self.buffer_config.max_playout_catchup_frames:
                stream = self._stream
                if stream is None or stream.application_data:
                    return

                if stream.next_playout_at is None:
                    if not self._try_rebuffer(stream, service_at, generation):
                        return

                if not self._playout_single_frame(stream, service_at, generation):
                    return
                serviced += 1

            # Avoid a decode burst after suspend/resume. Keep buffered voice,
            # but re-anchor the next device submission to real time.
            stream = self._stream
            if (
                stream is not None
                and stream.next_playout_at is not None
                and stream.next_playout_at < service_at
            ):
                stream.next_playout_at = service_at + _FRAME_INTERVAL_SECONDS

    def _try_rebuffer(self, stream: _VoiceStream, service_at: float, generation: Optional[int]) -> bool:
        if stream.rebuffer_until is None:
            if stream.end_index is not None:
                self._record_terminal_gap(stream)
                self._finish_stream(stream.end_reason, generation=generation)
            return False
        if service_at < stream.rebuffer_until:
            return False

        floor = stream.playout_index
        candidates = [
            index for index in stream.frame_buffer if floor is None or index >= floor
        ]
        if not candidates:
            stream.rebuffer_until = None
            stream.parked_at = service_at
            if stream.end_index is not None:
                self._record_terminal_gap(stream)
                self._finish_stream(stream.end_reason, generation=generation)
            return False

        first = min(candidates)
        if floor is None:
            stream.playout_index = first
        elif first > floor:
            skipped = first - floor
            self._record_resync(stream, skipped)
            stream.playout_index = first

        if stream.order is None:
            self._select_bit_order()
            if self._stream is not stream:
                return False

        stream.next_playout_at = stream.rebuffer_until
        stream.rebuffer_until = None
        stream.parked_at = None
        return True

    def _playout_single_frame(
        self, stream: _VoiceStream, service_at: float, generation: Optional[int]
    ) -> bool:
        """Service one due slot, recovering the timeline after bounded concealment."""
        deadline = stream.next_playout_at
        if deadline is None or service_at < deadline:
            return False
        if stream.playout_index is None:
            return False

        if stream.end_index is not None and stream.playout_index >= stream.end_index:
            self._finish_stream(stream.end_reason, generation=generation)
            return False

        index = stream.playout_index
        ambe = stream.frame_buffer.pop(index, None)
        if ambe is not None or stream.concealment_run < self.buffer_config.max_concealment_frames:
            return self._decode_playout_frame(stream, ambe, deadline)
        return self._recover_playout_gap(stream, index, service_at, generation)

    def _decode_playout_frame(
        self, stream: _VoiceStream, ambe: bytes | None, deadline: float
    ) -> bool:
        """Decode received or concealed audio and advance only the surviving stream."""
        if ambe is None:
            stream.lost_frames += 1
            stream.concealed_frames += 1
            stream.concealment_run += 1
            stream.max_concealment_run = max(stream.max_concealment_run, stream.concealment_run)
        elif stream.decoder_reset_pending:
            self._reset_decoder_after_discontinuity(stream)
            if self._stream is not stream:
                return False
        self._decode_and_play(ambe, concealed=ambe is None)
        if self._stream is not stream:
            return False
        stream.playout_index += 1
        stream.next_playout_at = deadline + _FRAME_INTERVAL_SECONDS
        return True

    def _recover_playout_gap(
        self,
        stream: _VoiceStream,
        index: int,
        service_at: float,
        generation: Optional[int],
    ) -> bool:
        """After concealment is exhausted, skip to queued speech, finish, or park."""
        future = [
            candidate for candidate in stream.frame_buffer
            if candidate > index and (stream.end_index is None or candidate < stream.end_index)
        ]
        if not stream.concealment_reset:
            self._reset_decoder_after_concealment(stream)
            if self._stream is not stream:
                return False

        if future:
            first = min(future)
            self._record_resync(stream, first - index)
            stream.playout_index = first
            return True

        if stream.end_index is not None:
            self._record_terminal_gap(stream)
            self._finish_stream(stream.end_reason, generation=generation)
            return False

        stream.next_playout_at = None
        stream.rebuffer_until = None
        stream.parked_at = service_at
        return False

    def _reset_decoder_after_concealment(self, stream: _VoiceStream) -> None:
        """Run mbelib's fourth repeat/mute transition without queuing silence."""

        try:
            result = self.decoder.conceal_frame()
        except DecoderError as exc:
            self._emit("error", str(exc))
            self._finish_stream("decoder error")
            return
        if len(result.pcm) != BYTES_PER_FRAME:
            self._emit("error", f"decoder returned {len(result.pcm)} PCM bytes")
            self._finish_stream("decoder error")
            return
        stream.clipped_samples += result.clipped_samples
        stream.concealment_reset = True
        if reset := getattr(self.audio, "reset_decoder_state", None):
            reset(stream.audio_generation)

    def _reset_decoder_after_discontinuity(self, stream: _VoiceStream) -> None:
        """Force mbelib through its bounded repeat path to a known mute/reset."""

        for _ in range(4):
            try:
                result = self.decoder.conceal_frame()
            except DecoderError as exc:
                self._emit("error", str(exc))
                self._finish_stream("decoder error")
                return
            if len(result.pcm) != BYTES_PER_FRAME:
                self._emit("error", f"decoder returned {len(result.pcm)} PCM bytes")
                self._finish_stream("decoder error")
                return
            stream.clipped_samples += result.clipped_samples
            if "M" in result.markers:
                break
        stream.decoder_reset_pending = False
        stream.concealment_reset = True
        if reset := getattr(self.audio, "reset_decoder_state", None):
            reset(stream.audio_generation)

    @staticmethod
    def _record_terminal_gap(stream: _VoiceStream) -> None:
        if stream.end_index is None or stream.playout_index is None:
            return
        skipped = max(0, stream.end_index - stream.playout_index)
        stream.lost_frames += skipped
        stream.skipped_frames += skipped

    def _record_resync(self, stream: _VoiceStream, skipped: int) -> None:
        if skipped <= 0:
            return
        stream.lost_frames += skipped
        stream.skipped_frames += skipped
        stream.resync_events += 1
        self._emit(
            "stream_recovery",
            f"{stream.header.my_call or 'Unknown'}: recovered after "
            f"{skipped} unavailable voice frame{'s' if skipped != 1 else ''}",
            callsign=stream.header.formatted_my_call,
            stream_id=stream.stream_id,
            skipped=skipped,
        )

    def _decode_and_play(
        self, ambe: Optional[bytes], *, concealed: bool = False
    ) -> None:
        stream = self._stream
        if stream is None or stream.order is None:
            return
        try:
            result = (
                self.decoder.conceal_frame()
                if concealed
                else self.decoder.decode_frame(ambe, stream.order)  # type: ignore[arg-type]
            )
        except DecoderError as exc:
            self._emit("error", str(exc))
            self._finish_stream("decoder error")
            return
        if len(result.pcm) != BYTES_PER_FRAME:
            self._emit("error", f"decoder returned {len(result.pcm)} PCM bytes")
            self._finish_stream("decoder error")
            return
        if result.total_corrections and not concealed:
            stream.corrected_frames += 1
        stream.clipped_samples += result.clipped_samples
        stream.decoded_frames += 1
        if not concealed:
            stream.concealment_run = 0
            stream.concealment_reset = False
            if len(stream.frame_buffer) < self.buffer_config.max_buffered_frames:
                stream.buffer_overflow_active = False
        submit_decoded = getattr(self.audio, "submit_decoded", None)
        accepted = (submit_decoded(result, stream.audio_generation) if submit_decoded
                    else self.audio.submit(result.pcm, stream.audio_generation))
        if not accepted:
            self._emit("warning", "Audio queue overrun; dropped one decoded frame")

    def _finish_stream(
        self, reason: str, *, generation: Optional[int] = None
    ) -> None:
        with self._lifecycle_lock:
            if generation is not None and (
                self._generation != generation
                or self._worker is None
                or self._worker.generation != generation
            ):
                return
            stream = self._stream
            if stream is None:
                return
            if finish := getattr(self.audio, "finish_stream", None):
                if reason in ("disconnect", "superseded", "decoder error"):
                    self.audio.cancel_stream()
                else:
                    finish(stream.audio_generation)
            if reason == "superseded" and stream.end_index is None:
                timeout = 30.0 if stream.application_data else self.buffer_config.voice_timeout_seconds
                self._interrupted_streams[(stream.protocol, stream.stream_id)] = stream.last_media_at + timeout
            else:
                self._interrupted_streams.pop((stream.protocol, stream.stream_id), None)
            duration = max(0.0, time.monotonic() - stream.started_at)
            if self.stream_log is not None:
                self.stream_log.finish(stream.recording)
            self._completed_stream = (
                stream.protocol,
                stream.stream_id,
                time.monotonic(),
            )
            if self._app_receiver is not None:
                drained_events = self._app_receiver.end_stream(stream.stream_id)
                for app_event in drained_events:
                    self._emit(app_event.kind, app_event.message, **app_event.fields)
            if stream.application_data:
                self._emit(
                    "app_stream_end",
                    f"RS-MS1 data stream from {stream.header.my_call or 'Unknown'} ended ({reason})",
                    callsign=stream.header.formatted_my_call,
                    stream_id=stream.stream_id,
                    duration_seconds=duration,
                    reason=reason,
                )
                self._stream = None
                return
            self._emit(
                "stream_end",
                f"{stream.header.my_call or 'Unknown'} ended: "
                f"{stream.decoded_frames} played, {stream.lost_frames} lost "
                f"({stream.concealed_frames} concealed), "
                f"{stream.reordered_frames} reordered, "
                f"{stream.late_frames} late, "
                f"{stream.duplicate_frames} duplicate, "
                f"{stream.resync_events} resync, "
                f"{stream.buffer_dropped_frames} stale-buffer drop ({reason})",
                callsign=stream.header.formatted_my_call,
                stream_id=stream.stream_id,
                frames=stream.decoded_frames,
                received=stream.received_frames,
                lost=stream.lost_frames,
                concealed=stream.concealed_frames,
                skipped=stream.skipped_frames,
                late=stream.late_frames,
                reordered=stream.reordered_frames,
                duplicates=stream.duplicate_frames,
                sequence_conflicts=stream.sequence_conflicts,
                resyncs=stream.resync_events,
                delayed_resumes=stream.delayed_resumes,
                buffer_dropped=stream.buffer_dropped_frames,
                peak_buffered=stream.peak_buffered_frames,
                max_concealment_run=stream.max_concealment_run,
                corrected=stream.corrected_frames,
                clipped=stream.clipped_samples,
                duration_seconds=duration,
                reason=reason,
            )
            self._stream = None

    def _send_link(
        self, sock: socket.socket, worker: _WorkerContext
    ) -> None:
        with self._lifecycle_lock:
            if not self._worker_is_current_locked(worker):
                return
            packet = build_link_packet(
                worker.target,
                worker.config.callsign,
                local_module=worker.config.local_module,
                client_description="reflectorListen",
            )
            self._send(sock, packet)
            if worker.target.protocol is Protocol.DEXTRA:
                self._send(sock, packet)

    def _send_unlink(
        self,
        sock: socket.socket,
        config: ConnectionConfig,
        target: ReflectorTarget,
    ) -> None:
        packet = build_unlink_packet(
            target,
            config.callsign,
            local_module=config.local_module,
        )
        self._send(sock, packet)
        if target.protocol is not Protocol.DCS:
            self._send(sock, packet)

    def _send_poll(
        self, sock: socket.socket, worker: _WorkerContext
    ) -> None:
        with self._lifecycle_lock:
            if not self._worker_is_current_locked(worker):
                return
            packet = build_poll_packet(
                worker.target,
                worker.config.callsign,
                local_module=worker.config.local_module,
            )
            self._send(sock, packet)

    def _worker_is_current_locked(self, worker: _WorkerContext) -> bool:
        return (
            self._worker is worker
            and self._generation == worker.generation
            and not worker.stop.is_set()
        )

    def _worker_is_current(self, worker: _WorkerContext) -> bool:
        with self._lifecycle_lock:
            return self._worker_is_current_locked(worker)

    def _set_state_for_worker(
        self,
        worker: _WorkerContext,
        state: ConnectionState,
        message: str,
    ) -> bool:
        with self._lifecycle_lock:
            if not self._worker_is_current_locked(worker):
                return False
            self._set_state(state, message)
            return True

    def _emit_for_worker(
        self, worker: _WorkerContext, kind: str, message: str, **fields: Any
    ) -> bool:
        with self._lifecycle_lock:
            if not self._worker_is_current_locked(worker):
                return False
            self._emit(kind, message, **fields)
            return True

    @staticmethod
    def _send(sock: socket.socket, packet: bytes) -> None:
        if sock.send(packet) != len(packet):
            raise OSError("Incomplete UDP datagram send")

    def _set_state(self, state: ConnectionState, message: str) -> None:
        with self._state_lock:
            self._state = state
        self._emit("state", message, state=state.value)

    def _emit(self, kind: str, message: str, **fields: Any) -> None:
        tx = self._tx
        if tx is not None and tx.diagnostics is not None:
            tx.diagnostics.log_event(kind, message, **fields)
        elif self.packet_logger.is_enabled:
            self.packet_logger.log_event(kind, message, **fields)
        try:
            self._on_event(SessionEvent(kind, message, fields=fields))
        except Exception:
            # UI callbacks must not be able to terminate the network thread.
            pass
