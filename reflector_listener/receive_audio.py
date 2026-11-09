"""Bounded receive audio worker with optional experimental ML reconstruction.

Native AMBE decoding for diagnostics/transmission stays in the existing codec.
Only this worker imports the optional ML runtime, and only when selected.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from types import SimpleNamespace
from typing import Callable

from .audio import WaveOutPlayer
from .config import BufferConfig


def enhanced_pcm16(output, gain: float) -> bytes:
    """Convert a complete enhanced block for playback or WAV export."""
    import numpy as np

    audio = np.asarray(output, dtype=np.float32)
    if audio.ndim != 1 or len(audio) % 960 or not np.isfinite(audio).all():
        raise ValueError("ML decoder returned invalid audio")
    return np.clip(np.rint(audio.astype(np.float64) * (32768.0 * gain)),
                   -32768, 32767).astype("<i2").tobytes()


@dataclass(frozen=True)
class _Command:
    kind: str
    generation: int
    epoch: int
    value: object = None


class ReceiveAudioPlayer:
    """Keep inference and device waits off Tk and the network receive thread.

    A setting change applies to the next stream. Cancellation invalidates work
    already in inference as well as queued sound. Muting resets a private epoch
    while preserving the receiver's public stream generation.
    """

    def __init__(self, on_error: Callable[[str], None] | None = None,
                 max_queued_frames: int | None = None,
                 max_device_frames: int | None = None, *,
                 config: BufferConfig | None = None,
                 on_warning: Callable[[str], None] | None = None,
                 device_id: str = "",
                 on_level: Callable[[float], None] | None = None,
                 player_factory: Callable = WaveOutPlayer,
                 runtime_factory: Callable | None = None) -> None:
        cfg = config or BufferConfig()
        self._on_error = on_error or (lambda message: None)
        self._on_warning = on_warning or (lambda message: None)
        self._device_id = device_id
        self._on_level = on_level or (lambda level: None)
        self._player_factory = player_factory
        self._runtime_factory = runtime_factory
        self._queue_limit = max(32, max_queued_frames or cfg.max_queued_frames)
        self._device_frames = max_device_frames or cfg.max_device_frames
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[_Command] = deque()
        self._generation = self._epoch = 0
        self._enabled = self._stream_enabled = self._muted = self._closed = False
        self._overflow: tuple[int, int] | None = None
        self._thread: threading.Thread | None = None
        self._player = None
        self._player_generation = 0
        self._runtime = None
        self._runtime_error: str | None = None
        self._runtime_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._finish_events: dict[int, threading.Event] = {}

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def set_ml_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("ML decoder selection must be on or off")
        with self._condition:
            if enabled != self._enabled:
                self._runtime_error = None
            self._enabled = enabled
            if (enabled and not self._closed and self._runtime is None
                    and not (self._preload_thread and self._preload_thread.is_alive())):
                # Importing PyTorch can take seconds. Keep that first load off
                # the audio worker so an active standard stream keeps playing.
                self._preload_thread = threading.Thread(
                    target=self._prewarm, name="ml-decoder-preload", daemon=True)
                self._preload_thread.start()

    def set_device(self, device_id: str) -> None:
        """Route live or replay output without resetting decoder state."""
        with self._condition:
            self._device_id = device_id
            if self._player is not None:
                self._player.set_device(device_id)

    def start(self) -> None:
        with self._condition:
            if self._closed or (self._thread and self._thread.is_alive()):
                return
            self._thread = threading.Thread(target=self._run, name="receive-audio", daemon=True)
            self._thread.start()

    def _invalidate(self) -> None:
        self._epoch += 1
        self._queue.clear()
        self._overflow = None
        for event in self._finish_events.values():
            event.set()
        self._finish_events.clear()
        if self._player is not None:
            self._player.cancel_stream()
        self._on_level(0.0)

    def begin_stream(self) -> int:
        with self._condition:
            self._generation += 1
            self._invalidate()
            self._stream_enabled = self._enabled
            self._queue.append(_Command("begin", self._generation, self._epoch, self._stream_enabled))
            self.start()
            self._condition.notify_all()
            return self._generation

    def cancel_stream(self) -> None:
        with self._condition:
            self._generation += 1
            self._invalidate()
            self._queue.append(_Command("cancel", self._generation, self._epoch))
            self._condition.notify_all()

    def set_muted(self, muted: bool) -> None:
        with self._condition:
            if self._muted == bool(muted):
                return
            self._muted = bool(muted)
            self._invalidate()
            self._queue.append(_Command("begin", self._generation, self._epoch, self._stream_enabled))
            self._condition.notify_all()

    def retry(self) -> None:
        with self._condition:
            self._runtime_error = None
            if self._player is not None:
                self._player.retry()

    def submit(self, pcm: bytes, generation: int | None = None) -> bool:
        """Compatibility PCM path; enhanced decoding requires submit_decoded."""
        return self._submit(SimpleNamespace(pcm=pcm, ambe_frame=None, gain=1.0), generation, "pcm")

    def submit_decoded(self, result, generation: int | None = None) -> bool:
        return self._submit(result, generation, "frame")

    def _submit(self, result, generation, kind) -> bool:
        if len(result.pcm) != 320:
            raise ValueError("Receive input must be one 20 ms PCM16/8 kHz frame")
        with self._condition:
            generation = self._generation if generation is None else generation
            if self._closed or generation != self._generation:
                return False
            if self._muted:
                return True
            dropped = False
            if sum(item.kind in ("frame", "pcm") for item in self._queue) >= self._queue_limit:
                for item in self._queue:
                    if item.kind in ("frame", "pcm"):
                        self._queue.remove(item)
                        break
                self._overflow = (generation, self._epoch)
                dropped = True
            self._queue.append(_Command(kind, generation, self._epoch, result))
            self.start()
            self._condition.notify_all()
            return not dropped

    def reset_decoder_state(self, generation: int) -> None:
        with self._condition:
            if not self._closed and generation == self._generation:
                self._queue.append(_Command("reset_decoder", generation, self._epoch))
                self._condition.notify_all()

    def finish_stream(self, generation: int) -> threading.Event:
        with self._condition:
            if generation in self._finish_events:
                return self._finish_events[generation]
            event = threading.Event()
            if self._closed or generation != self._generation or self._muted:
                event.set()
            else:
                self._finish_events[generation] = event
                self._queue.append(_Command("finish", generation, self._epoch, event))
                self.start()
                self._condition.notify_all()
            return event

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._invalidate()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def _current(self, command: _Command) -> bool:
        with self._condition:
            return (not self._closed and command.generation == self._generation
                    and command.epoch == self._epoch and not self._muted)

    def _select_player(self, rate: int, command: _Command) -> bool:
        if not self._current(command):
            return False
        if self._player is None or self._player.sample_rate != rate:
            old_player = self._player
            if old_player is not None:
                # A fallback preserves enhanced audio already sent to the
                # device before playing the remaining ordinary decoder frames.
                while self._current(command) and not old_player.wait_drained(timeout=0.05):
                    pass
                if not self._current(command):
                    return False
                old_player.close()
            device_id = self._device_id
            player = self._player_factory(sample_rate=rate, on_error=self._on_error,
                                          device_id=device_id, on_level=self._on_level,
                                          max_queued_frames=max(64, self._queue_limit),
                                          max_device_frames=self._device_frames)
            with self._condition:
                if not self._current(command):
                    player.close()
                    return False
                self._player = player
                if self._device_id != device_id:
                    player.set_device(self._device_id)
                self._player_generation = player.begin_stream()
        return True

    def _write_pcm(self, pcm: bytes, rate: int, command: _Command) -> None:
        if not self._select_player(rate, command):
            return
        size = rate // 50 * 2
        if len(pcm) % size:
            raise ValueError("ML output must contain whole 20 ms audio frames")
        for offset in range(0, len(pcm), size):
            with self._condition:
                if not self._current(command):
                    return
                if not self._player.submit(pcm[offset:offset + size], self._player_generation):
                    self._on_warning("Audio queue overrun; dropped one decoded frame")

    def _write_outputs(self, outputs, gain: float, pending: deque, command: _Command) -> None:
        for output in outputs:
            pcm = enhanced_pcm16(output, gain)
            self._write_pcm(pcm, 48_000, command)
            for _ in range(len(pcm) // 1920):
                if pending:
                    pending.popleft()

    def _fallback(self, stream, pending: deque, command: _Command, error: Exception | str):
        if stream is not None:
            stream.close()
        if self._current(command):
            self._on_warning(f"ML Enhanced 48kHz Decoder unavailable; using standard decoder: {error}")
            for result in pending:
                self._write_pcm(result.pcm, 8_000, command)
        pending.clear()
        return None

    def _prewarm(self) -> None:
        with self._condition:
            if self._closed or not self._enabled:
                return
        try:
            self._load_runtime()
        except Exception:
            # A selected stream reports the cached failure through fallback.
            # Background loading itself never publishes errors during shutdown.
            pass

    def _load_runtime(self):
        # A new enabled stream may arrive while preload is still running.
        # Only the enabled audio worker waits; settings and standard audio do not.
        with self._runtime_lock:
            if self._runtime_error is not None:
                raise RuntimeError(self._runtime_error)
            if self._runtime is None:
                try:
                    factory = self._runtime_factory
                    if factory is None:
                        from .ml_runtime import Runtime
                        factory = Runtime
                    self._runtime = factory()
                except Exception as exc:
                    self._runtime_error = str(exc)
                    raise
            return self._runtime

    def _run(self) -> None:
        stream = None
        pending: deque = deque()
        gain = 1.0
        try:
            while True:
                with self._condition:
                    while not self._closed and not self._queue:
                        self._condition.wait()
                    if self._closed:
                        return
                    command = self._queue.popleft()
                if command.kind in ("begin", "cancel"):
                    if stream is not None:
                        stream.close()
                    stream = None
                    pending.clear()
                    if self._current(command):
                        if self._player is not None:
                            self._player_generation = self._player.begin_stream()
                        if command.kind == "begin" and command.value:
                            try:
                                stream = self._load_runtime().new_stream(seed=6110704)
                            except Exception as exc:
                                self._fallback(None, pending, command, exc)
                    continue
                if not self._current(command):
                    if command.kind == "finish":
                        command.value.set()
                    continue
                if self._overflow == (command.generation, command.epoch) and stream is not None:
                    stream = self._fallback(stream, pending, command, "inference queue exceeded its bound")
                if command.kind in ("frame", "pcm"):
                    result = command.value
                    gain = result.gain
                    if stream is not None and command.kind == "frame":
                        pending.append(result)
                        try:
                            options = {} if getattr(result, "packet_loss_concealment", True) else {"packet_loss_concealment": False}
                            self._write_outputs(stream.decode(result.ambe_frame, **options), gain, pending, command)
                        except Exception as exc:
                            stream = self._fallback(stream, pending, command, exc)
                    else:
                        if stream is not None:
                            stream = self._fallback(stream, pending, command, "AMBE observations unavailable")
                        self._write_pcm(result.pcm, 8_000, command)
                elif command.kind == "reset_decoder" and stream is not None:
                    try:
                        stream.reset_decoder_state()
                    except Exception as exc:
                        stream = self._fallback(stream, pending, command, exc)
                elif command.kind == "finish":
                    if stream is not None:
                        try:
                            self._write_outputs(stream.finish(), gain, pending, command)
                        except Exception as exc:
                            self._fallback(stream, pending, command, exc)
                        finally:
                            stream.close()
                            stream = None
                    while self._current(command) and self._player is not None:
                        if self._player.wait_drained(timeout=0.05):
                            break
                    command.value.set()
        except Exception as exc:
            self._on_error(str(exc))
        finally:
            if stream is not None:
                stream.close()
            if self._player is not None:
                self._player.close()
            with self._condition:
                self._player = None
                for event in self._finish_events.values():
                    event.set()
                self._thread = None
