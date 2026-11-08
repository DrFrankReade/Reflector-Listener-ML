"""Persistent, thread-isolated stock mbelib-neo2.6.5 decoder. No Torch import."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from ._native_abi import Params, Result, load_decoder

STOCK_SHA256 = '698cb4ef2ef78ec0528e5ebddee3b5ba6ef964ba7f847ec99bfeb0459bcf11ea'
DEFAULT_LIBRARY = Path(__file__).resolve().parent.parent / 'vendor/stock265-Release.dll'

# Stock2.6.5 caches an FFT allocation per native thread without a public cleanup
# API. Reuse exclusively leased workers across successive talkspurts so those
# caches grow with peak simultaneous streams, not with every stream ever opened.
_workers_lock = Lock()
_idle_workers = []


def _lease_worker():
    with _workers_lock:
        return _idle_workers.pop() if _idle_workers else ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='ambe-stock265')


def _release_worker(worker):
    with _workers_lock:
        _idle_workers.append(worker)


@dataclass
class DecodedFrames:
    # Exactly the floats returned by stock mbe_processAmbe3600x2400Framef.
    pcm8_native: np.ndarray
    flags: np.ndarray
    errors: np.ndarray
    return_codes: np.ndarray
    compact: np.ndarray | None = None
    state: np.ndarray | None = None


class StockDecoder:
    """One ordered AMBE2400/D-STAR stream, one private native worker thread.

    The stock library has thread-local RNGs. Each session initializes once on
    its own thread; neither caller thread switches nor interleaving sessions
    reset/cross-contaminate those RNGs. Calls are synchronous, not queued ahead.
    Workers are exclusively leased and reused after close to bound native caches.
    ``decode`` accepts complete9-byte frames. ``close`` discards native state.
    """

    def __init__(self, library=DEFAULT_LIBRARY, *, observations=False):
        self._gate = Lock()
        self._closed = False
        self._observations = bool(observations)
        self._pool = _lease_worker()
        try:
            self._pool.submit(self._initialize, library).result()
        except BaseException:
            _release_worker(self._pool)
            raise

    def _initialize(self, library):
        self._lib = load_decoder(library, STOCK_SHA256)
        self._lib.mbe_versionString.argtypes = []
        self._lib.mbe_versionString.restype = C.c_char_p
        if self._lib.mbe_versionString() != b'2.6.5':
            raise ValueError('Expected qualified stock mbelib-neo2.6.5')
        self._parameters = (Params(), Params(), Params())
        self._lib.mbe_setThreadRngSeed(12345)
        self._lib.mbe_initMbeParms(*(C.byref(p) for p in self._parameters))

    def decode(self, packets):
        if not isinstance(packets, (bytes, bytearray, memoryview)):
            raise TypeError('Expected packed D-STAR bytes')
        raw = bytes(packets)
        if len(raw) % 9:
            raise ValueError('Expected complete9-byte D-STAR frames')
        with self._gate:
            if self._closed:
                raise RuntimeError('Native decoder is closed')
            try:
                return self._pool.submit(self._decode, raw).result()
            except BaseException:
                # A partially advanced native state cannot be retried safely.
                self._closed = True
                _release_worker(self._pool)
                raise

    def _decode(self, raw):
        if self._observations:
            from .observations import model_features, _snapshot, frame_conditioning
        count = len(raw) // 9
        pcm8 = np.empty((count, 160), np.float32)
        flags = np.empty(count, np.uint32)
        errors = np.empty(count, np.int32)
        codes = np.empty(count, np.int32)
        compact = np.empty((count, 74), np.float32) if self._observations else None
        state = np.empty((count, 371), np.float32) if self._observations else None
        for index in range(count):
            packet = (C.c_ubyte * 9).from_buffer_copy(raw[index*9:index*9+9])
            plane, data = (C.c_char * 96)(), (C.c_char * 49)()
            pcm, result = (C.c_float * 160)(), Result()
            if self._lib.mbe_decodeDStarDVData(packet, plane):
                raise ValueError('D-STAR channel mapping failed')
            rc = self._lib.mbe_processAmbe3600x2400Framef(
                pcm, C.byref(result), plane, data, *(C.byref(p) for p in self._parameters))
            if rc < 0:
                raise ValueError('Native decoder failed: ' + str(rc))
            waveform = np.frombuffer(bytes(pcm), dtype='<f4').copy()
            if not np.isfinite(waveform).all():
                raise ValueError('Nonfinite native PCM')
            bits = bytes(data)
            b0 = sum(bits[pos] << (6-j) for j, pos in enumerate((0, 1, 2, 3, 4, 5, 48)))
            flag = int(result.flags) | (0x80000000 if b0 == 124 else 0)
            pcm8[index], flags[index], errors[index], codes[index] = waveform, flag, result.total_errors, rc
            if self._observations:
                compact[index] = model_features(self._parameters[2], waveform)[:74]
                current, pre, post = (_snapshot(p) for p in self._parameters)
                state[index] = frame_conditioning(pre, post, current, waveform, result, flag)
        return DecodedFrames(pcm8, flags, errors, codes, compact, state)

    def close(self):
        with self._gate:
            if not self._closed:
                self._closed = True
                _release_worker(self._pool)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
