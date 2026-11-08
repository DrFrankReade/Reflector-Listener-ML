"""Optional, inference-only full360 step473130 decoder with 48 kHz float output.

Nothing imports NumPy or PyTorch until Runtime is constructed. The ordinary
8 kHz decoder and encoder remain independent of this experimental path.
"""
from __future__ import annotations

from collections import deque
import hashlib
import os
from pathlib import Path
import sys
import threading


def default_asset_path() -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / "native" / "ml_decoder"


class Runtime:
    """Load the frozen CPU inference model once, then create independent streams.

    Each stream's native state runs on an exclusively leased worker, preserving
    thread-local RNG isolation even when live/replay sessions are interleaved.
    """

    sample_rate = 48000
    future_frames = 18
    block_frames = 8

    def __init__(self, asset_path: str | os.PathLike | None = None):
        try:
            import numpy as np
            import torch
            from ._ml.model import ModelBundle
            from ._ml.native import STOCK_SHA256
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "ML decoder runtime is unavailable. Install requirements-ml.txt "
                "with this Python interpreter. " + str(exc)) from exc
        self._np = np
        self.path = Path(asset_path) if asset_path is not None else default_asset_path()
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.bundle = ModelBundle(self.path, device='cpu')
            self.library = self.path / 'stock265.dll'
            if hashlib.sha256(self.library.read_bytes()).hexdigest() != STOCK_SHA256:
                raise ValueError('ML decoder asset differs: stock265.dll')
        except (OSError, KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
            raise RuntimeError("Unable to initialize ML Enhanced 48kHz Decoder: " + str(exc)) from exc

    def new_stream(self, seed: int = 6110704):
        return _Stream(self, seed)


class _Stream:
    def __init__(self, runtime: Runtime, seed: int):
        from .ml_transport import TransportDecoder

        self._runtime = runtime
        self._stream = runtime.bundle.stream(seed=seed)
        self._native = TransportDecoder(runtime.library, observations=True)
        self._owner = None
        self._closed = False
        self.last_errors = 0
        self.last_flags = 0
        self._silent_frames = deque()

    def _require_worker(self):
        if self._closed:
            raise RuntimeError("ML decoder stream is closed")
        ident = threading.get_ident()
        if self._owner is None:
            self._owner = ident
        elif self._owner != ident:
            raise RuntimeError("ML decoder stream must stay on its original worker thread")

    def _observe(self, frame: bytes | None):
        self._require_worker()
        if frame is not None and len(frame) != 9:
            raise ValueError("ML decoder requires a 9-byte LSB-first AMBE frame")
        decoded = self._native.erase() if frame is None else self._native.decode(frame)
        self.last_errors, self.last_flags = int(decoded.errors[0]), int(decoded.flags[0])
        inputs = dict(pcm8=decoded.pcm8_native * self._runtime._np.float32(7 / 32768),
                      features=decoded.compact, flags=decoded.flags, errors=decoded.errors)
        return inputs, decoded.state

    def decode(self, frame: bytes | None, *, packet_loss_concealment: bool = True):
        """Advance one 20 ms frame, optionally silence its delayed missing audio."""
        self._require_worker()
        if type(packet_loss_concealment) is not bool:
            raise ValueError("Packet loss concealment must be on or off")
        if self._stream.finished:
            raise RuntimeError("ML decoder stream has already finished")
        inputs, state = self._observe(frame)
        self._silent_frames.append(frame is None and not packet_loss_concealment)
        return self._mask_output(self._stream.feed(inputs, state))

    def _mask_output(self, outputs):
        # The native predictor and neural context always advance through loss.
        # Apply the choice on the output timeline, after future-context buffering,
        # so an unchecked PLC setting produces exactly one silent missing slot.
        for output in outputs:
            for frame in output.reshape(-1, 960):
                if self._silent_frames.popleft():
                    frame.fill(0)
        return outputs

    def reset_decoder_state(self):
        """Mirror native resynchronization without restarting the neural timeline."""
        self._require_worker()
        decoded = self._native.resynchronize()
        self.last_errors, self.last_flags = int(decoded.errors[0]), int(decoded.flags[0])

    def finish(self):
        """Flush every final sample with unavailable-future padding exactly once."""
        self._require_worker()
        return self._mask_output(self._stream.finish())

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._native.close()
        self._native = None
        self._stream = None
        self._silent_frames.clear()
