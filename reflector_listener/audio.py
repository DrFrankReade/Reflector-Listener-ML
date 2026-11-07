"""Dependency-free Windows PCM playback and microphone capture using WinMM.

The session's jitter buffer produces one 20 ms, 8 kHz, mono PCM16 block at a
time. Keeping waveOut calls on one worker thread prevents Tkinter or the UDP
receive loop from waiting for the sound device.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .config import BufferConfig


SAMPLE_RATE = 8_000
SAMPLES_PER_FRAME = 160
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2


class AudioError(RuntimeError):
    """Raised when Windows cannot open or write to the selected audio device."""


if os.name == "nt":
    from ctypes import wintypes

    DWORD_PTR = ctypes.c_size_t

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", wintypes.WORD),
            ("nChannels", wintypes.WORD),
            ("nSamplesPerSec", wintypes.DWORD),
            ("nAvgBytesPerSec", wintypes.DWORD),
            ("nBlockAlign", wintypes.WORD),
            ("wBitsPerSample", wintypes.WORD),
            ("cbSize", wintypes.WORD),
        ]

    class WAVEHDR(ctypes.Structure):
        pass

    WAVEHDR._fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", wintypes.DWORD),
        ("dwBytesRecorded", wintypes.DWORD),
        ("dwUser", DWORD_PTR),
        ("dwFlags", wintypes.DWORD),
        ("dwLoops", wintypes.DWORD),
        ("lpNext", ctypes.POINTER(WAVEHDR)),
        ("reserved", DWORD_PTR),
    ]


@dataclass(frozen=True)
class _AudioItem:
    generation: int
    pcm: bytes


@dataclass(frozen=True)
class _ResetItem:
    generation: int


@dataclass
class _WaveBlock:
    storage: ctypes.Array
    header: "WAVEHDR"


class WaveOutPlayer:
    """Asynchronously play 20 ms mono PCM16 frames through Windows waveOut.

    This class manages a bounded device queue for audio playback.
    `begin_stream` invalidates queued audio from an older D-Star stream.
    Network reordering and timing are handled before decoded PCM reaches this
    bounded device queue.
    """

    _WAVE_MAPPER = 0xFFFFFFFF
    _WAVE_FORMAT_PCM = 1
    _CALLBACK_NULL = 0
    _WHDR_DONE = 0x00000001
    _MMSYSERR_NOERROR = 0

    def __init__(
        self,
        on_error: Optional[Callable[[str], None]] = None,
        max_queued_frames: Optional[int] = None,
        max_device_frames: Optional[int] = None,
        *,
        config: Optional[BufferConfig] = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if os.name != "nt":
            raise AudioError("waveOut audio is available only on Windows")
        self._on_error = on_error or (lambda _message: None)
        if sample_rate not in (8_000, 48_000):
            raise ValueError("Playback sample rate must be 8000 or 48000 Hz")
        self.sample_rate = sample_rate
        self.bytes_per_frame = sample_rate // 50 * 2
        default_cfg = config or BufferConfig()
        queued_cap = (
            max_queued_frames
            if max_queued_frames is not None
            else default_cfg.max_queued_frames
        )
        device_cap = (
            max_device_frames
            if max_device_frames is not None
            else default_cfg.max_device_frames
        )
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queued_cap)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self._muted = False
        self._max_device_frames = max(2, int(device_cap))
        self._thread: Optional[threading.Thread] = None
        self._sentinel = object()
        self._failure_latched = False
        self._drained = threading.Event()
        self._drained.set()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._failure_latched:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="dstar-waveout", daemon=True
        )
        self._thread.start()

    def begin_stream(self) -> int:
        """Start a logical stream and discard queued audio from older streams."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        self._replace_queue_with(_ResetItem(generation))
        return generation

    def cancel_stream(self) -> None:
        """Immediately invalidate queued and device-buffered speech."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        self._replace_queue_with(_ResetItem(generation))

    def retry(self) -> None:
        """Allow one fresh device-open attempt after an explicit reconnect."""
        if self._thread is None or not self._thread.is_alive():
            self._failure_latched = False

    def set_muted(self, muted: bool) -> None:
        """Silence output without changing the receiver's stream generation."""
        with self._lock:
            if self._muted == muted:
                return
            self._muted = muted
            self._replace_queue_with(_ResetItem(self._generation))

    def submit(self, pcm: bytes, generation: Optional[int] = None) -> bool:
        if len(pcm) != self.bytes_per_frame:
            raise ValueError(
                f"PCM frame must be {self.bytes_per_frame} bytes, got {len(pcm)}"
            )
        with self._lock:
            if self._muted:
                return True
        if self._failure_latched:
            # The error callback already reported the device failure. Keep
            # receiving/logging without retrying fifty times per second.
            return True
        if self._thread is None or not self._thread.is_alive():
            self.start()
        with self._lock:
            item = _AudioItem(self._generation if generation is None else generation, pcm)
            self._drained.clear()
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                # Prefer current speech over stale latency. Report the drop.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return False
                try:
                    self._queue.put_nowait(item)
                    return False
                except queue.Full:
                    return False

    def wait_drained(self, timeout: float | None = None) -> bool:
        """Wait for both the software queue and submitted device blocks."""
        return self._drained.wait(timeout)

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(self._sentinel)
            except (queue.Empty, queue.Full):
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _replace_queue_with(self, control: object) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(control)
        except queue.Full:
            pass

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != WaveOutPlayer._MMSYSERR_NOERROR:
            raise AudioError(f"{operation} failed (Windows multimedia error {result})")

    def _run(self) -> None:
        winmm = ctypes.WinDLL("winmm.dll")
        handle = ctypes.c_void_p()
        wave_format = WAVEFORMATEX(
            self._WAVE_FORMAT_PCM,
            1,
            self.sample_rate,
            self.sample_rate * 2,
            2,
            16,
            0,
        )

        winmm.waveOutOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.POINTER(WAVEFORMATEX),
            DWORD_PTR,
            DWORD_PTR,
            ctypes.c_uint,
        ]
        winmm.waveOutOpen.restype = ctypes.c_uint
        winmm.waveOutPrepareHeader.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(WAVEHDR),
            ctypes.c_uint,
        ]
        winmm.waveOutPrepareHeader.restype = ctypes.c_uint
        winmm.waveOutWrite.argtypes = winmm.waveOutPrepareHeader.argtypes
        winmm.waveOutWrite.restype = ctypes.c_uint
        winmm.waveOutUnprepareHeader.argtypes = winmm.waveOutPrepareHeader.argtypes
        winmm.waveOutUnprepareHeader.restype = ctypes.c_uint
        winmm.waveOutReset.argtypes = [ctypes.c_void_p]
        winmm.waveOutReset.restype = ctypes.c_uint
        winmm.waveOutClose.argtypes = [ctypes.c_void_p]
        winmm.waveOutClose.restype = ctypes.c_uint

        pending: list[_WaveBlock] = []
        device_generation = self.generation
        header_size = ctypes.sizeof(WAVEHDR)

        try:
            self._check(
                winmm.waveOutOpen(
                    ctypes.byref(handle),
                    self._WAVE_MAPPER,
                    ctypes.byref(wave_format),
                    0,
                    0,
                    self._CALLBACK_NULL,
                ),
                "waveOutOpen",
            )

            while not self._stop.is_set():
                with self._lock:
                    muted = self._muted
                if muted:
                    if pending:
                        self._reset(winmm, handle, pending, header_size)
                    self._stop.wait(0.005)
                    continue
                self._reap(winmm, handle, pending, header_size)
                with self._lock:
                    if not pending and self._queue.empty():
                        self._drained.set()
                if len(pending) >= self._max_device_frames:
                    self._stop.wait(0.005)
                    continue
                try:
                    item = self._queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                if item is self._sentinel:
                    break
                if isinstance(item, _ResetItem):
                    self._reset(winmm, handle, pending, header_size)
                    device_generation = item.generation
                    continue
                if not isinstance(item, _AudioItem):
                    continue
                current_generation = self.generation
                if item.generation != current_generation:
                    continue
                if device_generation != item.generation:
                    self._reset(winmm, handle, pending, header_size)
                    device_generation = item.generation
                self._write(winmm, handle, pending, item.pcm, header_size)
        except Exception as exc:  # audio failure should not kill UDP reception
            self._failure_latched = True
            self._on_error(str(exc))
        finally:
            if handle.value:
                try:
                    self._reset(winmm, handle, pending, header_size)
                    winmm.waveOutClose(handle)
                except Exception:
                    pass
            self._drained.set()

    def _write(
        self,
        winmm: ctypes.WinDLL,
        handle: ctypes.c_void_p,
        pending: list[_WaveBlock],
        pcm: bytes,
        header_size: int,
    ) -> None:
        storage = ctypes.create_string_buffer(pcm, len(pcm))
        header = WAVEHDR()
        header.lpData = ctypes.addressof(storage)
        header.dwBufferLength = len(pcm)
        block = _WaveBlock(storage, header)
        self._check(
            winmm.waveOutPrepareHeader(handle, ctypes.byref(header), header_size),
            "waveOutPrepareHeader",
        )
        try:
            self._check(
                winmm.waveOutWrite(handle, ctypes.byref(header), header_size),
                "waveOutWrite",
            )
        except Exception:
            winmm.waveOutUnprepareHeader(handle, ctypes.byref(header), header_size)
            raise
        pending.append(block)

    def _reap(
        self,
        winmm: ctypes.WinDLL,
        handle: ctypes.c_void_p,
        pending: list[_WaveBlock],
        header_size: int,
    ) -> None:
        remaining: list[_WaveBlock] = []
        for block in pending:
            if block.header.dwFlags & self._WHDR_DONE:
                winmm.waveOutUnprepareHeader(
                    handle, ctypes.byref(block.header), header_size
                )
            else:
                remaining.append(block)
        pending[:] = remaining

    def _reset(
        self,
        winmm: ctypes.WinDLL,
        handle: ctypes.c_void_p,
        pending: list[_WaveBlock],
        header_size: int,
    ) -> None:
        winmm.waveOutReset(handle)
        for block in pending:
            winmm.waveOutUnprepareHeader(
                handle, ctypes.byref(block.header), header_size
            )
        pending.clear()


class WaveInRecorder:
    """Capture exact 20 ms PCM16/8 kHz microphone frames on a worker thread."""

    _WAVE_MAPPER = 0xFFFFFFFF
    _WAVE_FORMAT_PCM = 1
    _CALLBACK_NULL = 0
    _WHDR_DONE = 0x00000001
    _MMSYSERR_NOERROR = 0

    def __init__(
        self,
        on_frame: Callable[[bytes], None],
        on_error: Optional[Callable[[str], None]] = None,
        buffer_count: int = 6,
    ) -> None:
        if os.name != "nt":
            raise AudioError("waveIn audio is available only on Windows")
        self._on_frame = on_frame
        self._on_error = on_error or (lambda _message: None)
        self._buffer_count = max(3, int(buffer_count))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="dstar-wavein", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != WaveInRecorder._MMSYSERR_NOERROR:
            raise AudioError(f"{operation} failed (Windows multimedia error {result})")

    def _run(self) -> None:
        winmm = ctypes.WinDLL("winmm.dll")
        handle = ctypes.c_void_p()
        wave_format = WAVEFORMATEX(
            self._WAVE_FORMAT_PCM, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, 0
        )
        header_size = ctypes.sizeof(WAVEHDR)
        blocks = [
            _WaveBlock(ctypes.create_string_buffer(BYTES_PER_FRAME), WAVEHDR())
            for _ in range(self._buffer_count)
        ]
        for block in blocks:
            block.header.lpData = ctypes.addressof(block.storage)
            block.header.dwBufferLength = BYTES_PER_FRAME

        winmm.waveInOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
            ctypes.POINTER(WAVEFORMATEX), DWORD_PTR, DWORD_PTR, ctypes.c_uint,
        ]
        winmm.waveInOpen.restype = ctypes.c_uint
        header_args = [ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveInPrepareHeader.argtypes = header_args
        winmm.waveInPrepareHeader.restype = ctypes.c_uint
        winmm.waveInAddBuffer.argtypes = header_args
        winmm.waveInAddBuffer.restype = ctypes.c_uint
        winmm.waveInUnprepareHeader.argtypes = header_args
        winmm.waveInUnprepareHeader.restype = ctypes.c_uint
        winmm.waveInStart.argtypes = [ctypes.c_void_p]
        winmm.waveInStart.restype = ctypes.c_uint
        winmm.waveInReset.argtypes = [ctypes.c_void_p]
        winmm.waveInReset.restype = ctypes.c_uint
        winmm.waveInClose.argtypes = [ctypes.c_void_p]
        winmm.waveInClose.restype = ctypes.c_uint

        try:
            self._check(
                winmm.waveInOpen(
                    ctypes.byref(handle), self._WAVE_MAPPER,
                    ctypes.byref(wave_format), 0, 0, self._CALLBACK_NULL,
                ),
                "waveInOpen",
            )
            for block in blocks:
                self._queue_input_block(winmm, handle, block, header_size)
            self._check(winmm.waveInStart(handle), "waveInStart")

            while not self._stop.wait(0.002):
                for block in blocks:
                    if not block.header.dwFlags & self._WHDR_DONE:
                        continue
                    recorded = int(block.header.dwBytesRecorded)
                    if recorded == BYTES_PER_FRAME:
                        self._on_frame(bytes(block.storage.raw[:recorded]))
                    self._check(
                        winmm.waveInUnprepareHeader(
                            handle, ctypes.byref(block.header), header_size
                        ),
                        "waveInUnprepareHeader",
                    )
                    block.header = WAVEHDR()
                    block.header.lpData = ctypes.addressof(block.storage)
                    block.header.dwBufferLength = BYTES_PER_FRAME
                    self._queue_input_block(winmm, handle, block, header_size)
        except Exception as exc:
            self._on_error(str(exc))
        finally:
            if handle.value:
                winmm.waveInReset(handle)
                for block in blocks:
                    if block.header.dwFlags:
                        winmm.waveInUnprepareHeader(
                            handle, ctypes.byref(block.header), header_size
                        )
                winmm.waveInClose(handle)
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def _queue_input_block(
        self,
        winmm: ctypes.WinDLL,
        handle: ctypes.c_void_p,
        block: _WaveBlock,
        header_size: int,
    ) -> None:
        self._check(
            winmm.waveInPrepareHeader(handle, ctypes.byref(block.header), header_size),
            "waveInPrepareHeader",
        )
        self._check(
            winmm.waveInAddBuffer(handle, ctypes.byref(block.header), header_size),
            "waveInAddBuffer",
        )


class NullAudioPlayer:
    """Test/headless audio sink with the same small interface."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def start(self) -> None:
        return

    def begin_stream(self) -> int:
        self._generation += 1
        self.frames.clear()
        return self._generation

    def cancel_stream(self) -> None:
        self._generation += 1
        self.frames.clear()

    def retry(self) -> None:
        return

    def submit(self, pcm: bytes, generation: Optional[int] = None) -> bool:
        if generation is None or generation == self._generation:
            self.frames.append(pcm)
            return True
        return False

    def close(self) -> None:
        return
