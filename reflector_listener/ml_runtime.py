"""Optional, inference-only Pilot5C decoder with 48 kHz float output.

Nothing imports NumPy or PyTorch until Runtime is constructed. The ordinary
8 kHz decoder and encoder remain independent of this experimental path.
"""
from __future__ import annotations

import ctypes as C
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def default_asset_path() -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / "native" / "ml_decoder"


class Runtime:
    """Load the frozen CPU inference model once, then create independent streams.

    Each stream belongs to one worker thread. The stock decoder's noise RNG is
    thread local; do not interleave two native streams on the same thread.
    """

    sample_rate = 48000
    future_frames = 18
    block_frames = 8

    def __init__(self, asset_path: str | os.PathLike | None = None):
        try:
            import numpy as np
            import torch
            from ._ml.model import SpeechReconstructorV2P5C
            from ._ml.observations import load_decoder, normalize_features
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "ML decoder runtime is unavailable. Install requirements-ml.txt "
                "with this Python interpreter. " + str(exc)
            ) from exc

        self._np = np
        self.path = Path(asset_path) if asset_path is not None else default_asset_path()
        try:
            manifest = json.loads((self.path / "MANIFEST.json").read_text(encoding="utf-8"))
            if (manifest["model"] != "Pilot5C" or manifest["future_frames"] != 18
                    or manifest["decoder_version"] != "2.6.5"):
                raise ValueError("Unsupported ML decoder manifest")
            for name in ("candidate.pt", "normalization.json", "state_normalization.json", "stock265.dll"):
                payload = (self.path / name).read_bytes()
                expected = manifest["files"][name]
                if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError("ML decoder asset differs: " + name)
            self.normalization = json.loads((self.path / "normalization.json").read_text(encoding="utf-8"))
            self.state_normalization = json.loads((self.path / "state_normalization.json").read_text(encoding="utf-8"))
            normalize_features(np.zeros((1, 371), np.float32), self.state_normalization)
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            # Construction initializes parameters before the frozen weights load;
            # preserve unrelated PyTorch users' RNG state in the same process.
            with torch.random.fork_rng(devices=[]):
                self.model = SpeechReconstructorV2P5C().cpu().eval()
            self.model.load_state_dict(torch.load(self.path / "candidate.pt", map_location="cpu", weights_only=True), strict=True)
            self.model.requires_grad_(False)
            self._lib = load_decoder(self.path / "stock265.dll", manifest["files"]["stock265.dll"]["sha256"])
            self._lib.mbe_versionString.argtypes = []
            self._lib.mbe_versionString.restype = C.c_char_p
            if self._lib.mbe_versionString() != b"2.6.5":
                raise ValueError("ML decoder requires its pinned stock2.6.5 library")
            self._lib.mbe_processAmbe2400Dataf.argtypes = [C.c_void_p] * 6
            self._lib.mbe_processAmbe2400Dataf.restype = C.c_int
        except (OSError, KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
            raise RuntimeError("Unable to initialize ML Enhanced 48kHz Decoder: " + str(exc)) from exc

    def new_stream(self, seed: int = 6110704):
        return _Stream(self, seed)


class _Stream:
    def __init__(self, runtime: Runtime, seed: int):
        from ._ml.observations import Params
        from ._ml.stream import FutureContextStream

        self._runtime = runtime
        self._native = (Params(), Params(), Params())
        self._stream = FutureContextStream(runtime.model, runtime.normalization,
                                          runtime.state_normalization, seed=seed, future_frames=18)
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
            # Keep the training/inference decoder's fixed native seed. The
            # caller's seed controls only the continuous neural noise timeline.
            self._runtime._lib.mbe_setThreadRngSeed(12345)
            self._runtime._lib.mbe_initMbeParms(*(C.byref(p) for p in self._native))
        elif self._owner != ident:
            raise RuntimeError("ML decoder stream must stay on its original worker thread")

    def _observe(self, frame: bytes | None):
        from ._ml.observations import Result, _snapshot, frame_conditioning, model_features

        self._require_worker()
        np = self._runtime._np
        lib = self._runtime._lib
        pcm, bits = (C.c_float * 160)(), (C.c_char * 49)()
        if frame is None:
            # v2.6.5 uses the explicit uncorrectable-C0 flag to distinguish a
            # lost frame from merely counted, correctable channel errors.
            result = Result(c0_errors=4, total_errors=4, flags=0x0402)
            code = lib.mbe_processAmbe2400Dataf(pcm, C.byref(result), bits,
                                             *(C.byref(p) for p in self._native))
        else:
            if len(frame) != 9:
                raise ValueError("ML decoder requires a 9-byte LSB-first AMBE frame")
            packet, plane = (C.c_ubyte * 9).from_buffer_copy(frame), (C.c_char * 96)()
            if lib.mbe_decodeDStarDVData(packet, plane):
                raise RuntimeError("ML decoder D-STAR channel mapping failed")
            result = Result()
            # The full-frame API corrects protected AMBE channel bits before
            # synthesis; the ML observations use that corrected decoder state.
            code = lib.mbe_processAmbe3600x2400Framef(pcm, C.byref(result), plane, bits,
                                                   *(C.byref(p) for p in self._native))
        if code < 0:
            raise RuntimeError(f"ML native decode failed with status {code}")
        waveform = np.frombuffer(bytes(pcm), dtype="<f4").copy()
        data = bytes(bits)
        b0 = sum(data[pos] << (6 - j) for j, pos in enumerate((0, 1, 2, 3, 4, 5, 48)))
        flags = int(result.flags) | (0x80000000 if b0 == 124 else 0)
        features = model_features(self._native[2], waveform)[:74]
        current, pre, post = (_snapshot(p) for p in self._native)
        state = frame_conditioning(pre, post, current, waveform, result, flags)
        self.last_errors, self.last_flags = int(result.total_errors), flags
        # The model was trained using stock float PCM scaled by 7/32768.
        # This is its fixed input calibration, independent of listening gain.
        inputs = {"pcm8": (waveform * np.float32(7 / 32768)).reshape(1, 160),
                  "features": features.reshape(1, 74),
                  "flags": np.asarray([flags], np.int64),
                  "errors": np.asarray([self.last_errors], np.int32)}
        return inputs, state.reshape(1, 371)

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
        self._runtime._lib.mbe_initMbeParms(*(C.byref(p) for p in self._native))
        for _ in range(4):
            self._observe(None)

    def finish(self):
        """Flush every final sample with unavailable-future padding exactly once."""
        self._require_worker()
        return self._mask_output(self._stream.finish())

    def close(self):
        self._closed = True
        self._native = None
        self._stream = None
        self._silent_frames.clear()
