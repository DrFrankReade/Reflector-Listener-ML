"""Opt-in ML beside an unchanged native8 kHz output timeline."""
from dataclasses import dataclass
from threading import Lock
import time

import numpy as np

from .native import DEFAULT_LIBRARY, StockDecoder


@dataclass
class AudioChunk:
    pcm8_native: np.ndarray       # Unscaled stock float samples, always available.
    enhanced48: np.ndarray | None # Normalized ML floats; may be empty during lookahead.
    flags: np.ndarray
    errors: np.ndarray
    return_codes: np.ndarray


class DecoderStream:
    """Feed packed9-byte AMBE2400/D-STAR frames, possibly split between calls.

    Default: only stock8 kHz; no model/Torch import. Pass ``enhance=True`` to
    load the accepted model, or provide a shared ModelBundle. Output48 is a
    separate delayed timeline; never replace/interpolate native8 in place.
    Call finish once at the actual stream end, then close (or use with).
    After failure the stream is unusable; create a fresh stream at a known
    decoder reset boundary. No frame dropping, invented packets or silent retry.
    """

    def __init__(self, *, enhance=False, bundle=None, library=DEFAULT_LIBRARY,
                 device='cpu', seed=6110704):
        if bundle is not None and not enhance:
            raise ValueError('A model bundle requires explicit enhance=True')
        self._gate = Lock()
        self._state = 'open'
        self._pending = b''
        self._frames = 0
        self._native_seconds = self._max_native_batch_seconds = 0.
        self._elapsed = self._max_feed_seconds = 0.
        self._neural = None
        self.bundle = None
        if enhance:
            if bundle is None:
                from .model import ModelBundle
                bundle = ModelBundle(device=device)
            self.bundle = bundle
            self._neural = bundle.stream(seed=seed)
        self._native = StockDecoder(library, observations=enhance)

    def _require_open(self):
        if self._state != 'open':
            raise RuntimeError('Decoder stream is ' + self._state)

    def feed(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('Expected packed D-STAR bytes')
        with self._gate:
            self._require_open()
            started = time.perf_counter()
            raw = self._pending + bytes(data)
            complete = len(raw) - len(raw) % 9
            self._pending = raw[complete:]
            pcm8, outputs, flags, errors, codes = [], [], [], [], []
            try:
                # Bound native observation batches regardless of input chunk size.
                for pos in range(0, complete, 8*9):
                    native_start = time.perf_counter()
                    decoded = self._native.decode(raw[pos:min(pos+8*9, complete)])
                    native_elapsed = time.perf_counter() - native_start
                    self._native_seconds += native_elapsed
                    self._max_native_batch_seconds = max(self._max_native_batch_seconds, native_elapsed)
                    pcm8.append(decoded.pcm8_native.reshape(-1))
                    flags.append(decoded.flags); errors.append(decoded.errors); codes.append(decoded.return_codes)
                    self._frames += len(decoded.flags)
                    if self._neural is not None:
                        inputs = dict(pcm8=decoded.pcm8_native * np.float32(7/32768),
                                      features=decoded.compact, flags=decoded.flags, errors=decoded.errors)
                        outputs.extend(self._neural.feed(inputs, decoded.state))
                elapsed = time.perf_counter() - started
                self._elapsed += elapsed
                self._max_feed_seconds = max(self._max_feed_seconds, elapsed)
                return AudioChunk(_join(pcm8), _join(outputs) if self._neural is not None else None,
                                  _join(flags, np.uint32), _join(errors, np.int32), _join(codes, np.int32))
            except BaseException:
                self._state = 'failed'
                self._native.close()
                raise

    def finish(self):
        with self._gate:
            self._require_open()
            try:
                if self._pending:
                    raise ValueError('Truncated final D-STAR frame: ' + str(len(self._pending)) + ' trailing bytes')
                started = time.perf_counter()
                tail = _join(self._neural.finish()) if self._neural is not None else None
                self._elapsed += time.perf_counter() - started
                self._state = 'finished'
                return AudioChunk(np.empty(0, np.float32), tail, np.empty(0, np.uint32),
                                  np.empty(0, np.int32), np.empty(0, np.int32))
            except BaseException:
                self._state = 'failed'
                raise
            finally:
                self._native.close()

    def report(self):
        with self._gate:
            report = dict(state=self._state, input_frames=self._frames, pcm8_samples=self._frames*160,
                          incomplete_packet_bytes=len(self._pending), native8_modified=False,
                          native_thread_isolated=True, native_seconds=self._native_seconds,
                          maximum_native_batch_ms=1000*self._max_native_batch_seconds,
                          processing_seconds=self._elapsed, maximum_feed_ms=1000*self._max_feed_seconds,
                          realtime_factor=self._elapsed/(self._frames*.020) if self._frames else None,
                          model_id=self.bundle.manifest['model_id'] if self.bundle else None,
                          audio_device_qualified=False)
            if self._neural is not None:
                report['enhancement'] = self._neural.report()
                report['enhancement']['native_decoder_integrated'] = True
            return report

    def close(self):
        with self._gate:
            self._native.close()
            if self._state == 'open':
                self._state = 'closed_without_flush'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _join(values, dtype=np.float32):
    return np.concatenate(values) if values else np.empty(0, dtype)
