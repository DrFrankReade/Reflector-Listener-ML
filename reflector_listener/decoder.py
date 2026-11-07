"""Thread-safe D-STAR encoder/decoder using the pinned mbelib-neo C API."""
from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
import threading
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

FRAME_BYTES = 9
PCM_SAMPLES = 160
CODEC_VERSION = "2.6.2"
DEFAULT_GAIN = 1.0
GAIN_PROFILE = "mbelib-neo-pcm16-v1"


class DecoderError(RuntimeError):
    """The native codec could not be loaded or rejected an operation."""


class BitOrder(IntEnum):
    MSB_FIRST = 0
    LSB_FIRST = 1

    @property
    def label(self) -> str:
        return "MSB-first" if self is BitOrder.MSB_FIRST else "LSB-first"


class _MbeParms(ctypes.Structure):
    """mbe_parms from native/vendor/mbelib-neo/include/mbelib-neo/mbelib.h."""
    _fields_ = [
        ("w0", ctypes.c_float), ("L", ctypes.c_int), ("K", ctypes.c_int),
        ("Vl", ctypes.c_int * 57), ("Ml", ctypes.c_float * 57),
        ("log2Ml", ctypes.c_float * 57), ("PHIl", ctypes.c_float * 57),
        ("PSIl", ctypes.c_float * 57), ("gamma", ctypes.c_float),
        ("tonePhase", ctypes.c_uint32), ("swn", ctypes.c_int),
        ("localEnergy", ctypes.c_float), ("amplitudeThreshold", ctypes.c_int),
        ("errorRate", ctypes.c_float), ("errorCountTotal", ctypes.c_int),
        ("errorCount4", ctypes.c_int), ("repeatCount", ctypes.c_int),
        ("mutingThreshold", ctypes.c_float), ("previousUw", ctypes.c_float * 256),
        ("noiseSeed", ctypes.c_float), ("noiseOverlap", ctypes.c_float * 96),
    ]


class _ProcessResult(ctypes.Structure):
    _fields_ = [
        ("c0_errors", ctypes.c_int), ("protected_errors", ctypes.c_int),
        ("c4_errors", ctypes.c_int), ("total_errors", ctypes.c_int),
        ("flags", ctypes.c_uint),
    ]


_Frame = (ctypes.c_char * 24) * 4
_Bits = ctypes.c_char * 49
_Packed = ctypes.c_uint8 * FRAME_BYTES
_Pcm = ctypes.c_int16 * PCM_SAMPLES
_REVERSE_BITS = bytes(int(f"{byte:08b}"[::-1], 2) for byte in range(256))


@dataclass(frozen=True)
class FecScore:
    c0: int
    c1: int
    total: int
    parity: int
    cost: int


@dataclass(frozen=True)
class DecodeResult:
    pcm: bytes
    c0: int = 0
    c1: int = 0
    total_corrections: int = 0
    clipped_samples: int = 0
    nonfinite_samples: int = 0
    markers: str = ""
    # Retain the wire input for the optional asynchronous ML receive path.
    # None represents a concealed frame. The normal PCM path is unchanged.
    ambe_frame: bytes | None = None
    gain: float = DEFAULT_GAIN


def default_dll_path() -> Path:
    # PyInstaller one-file and onedir builds put bundled data in _MEIPASS.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / "native" / "bin" / "mbe-neo.dll"


class AmbeDecoder:
    """Independent transmit/receive state for 8 kHz PCM16 and 9-byte AMBE.

    The PCM16 entry point performs mbelib-neo's native output conversion.
    User gain is applied once afterward; no Soft-AMBE compensation is needed.
    """

    def __init__(self, dll_path: os.PathLike[str] | str | None = None, *,
                 seed: int = 1, gain: float = DEFAULT_GAIN) -> None:
        self._lock = threading.RLock()
        self._encoder = None
        self._closed = True
        self._gain = self._validated_gain(gain)
        if struct.calcsize("P") != 8:
            raise DecoderError("mbe-neo.dll requires 64-bit Python")
        self.path = Path(dll_path).resolve() if dll_path else default_dll_path()
        if not self.path.is_file():
            raise DecoderError(f"bundled mbelib-neo codec DLL not found at {self.path}")
        try:
            self._dll = ctypes.CDLL(str(self.path))
            self._bind()
        except (OSError, AttributeError) as exc:
            raise DecoderError(f"unable to load mbelib-neo from {self.path}: {exc}") from exc
        self._decode_state = (_MbeParms(), _MbeParms(), _MbeParms())
        self._encode_state = (_MbeParms(), _MbeParms(), _MbeParms())
        self._encoder = self._dll.mbe_ambe2400EncoderAlloc()
        if not self._encoder:
            raise DecoderError("mbelib-neo encoder initialization failed")
        self._closed = False
        try:
            self.begin_stream(seed)
            self.begin_encode_stream()
        except Exception:
            self.close()
            raise

    def _bind(self) -> None:
        dll = self._dll
        dll.mbe_versionString.argtypes = []
        dll.mbe_versionString.restype = ctypes.c_char_p
        self._version = (dll.mbe_versionString() or b"unknown").decode("ascii", "replace")
        # ctypes owns these public structs. Reject unverified versions before
        # passing buffers into a library whose layout might have changed.
        if self._version != CODEC_VERSION:
            raise DecoderError(f"unsupported mbelib-neo {self._version}; expected {CODEC_VERSION}")
        parms = ctypes.POINTER(_MbeParms)
        bits = ctypes.POINTER(ctypes.c_char)
        frame = ctypes.POINTER(ctypes.c_char * 24)
        packed = ctypes.POINTER(ctypes.c_uint8)
        pcm = ctypes.POINTER(ctypes.c_int16)
        status = ctypes.POINTER(_ProcessResult)
        signatures = {
            "mbe_initMbeParms": ([parms, parms, parms], None),
            "mbe_moveMbeParms": ([parms, parms], None),
            "mbe_setThreadRngSeed": ([ctypes.c_uint32], None),
            "mbe_ambe2400EncoderAlloc": ([], ctypes.c_void_p),
            "mbe_ambe2400EncoderReset": ([ctypes.c_void_p], None),
            "mbe_ambe2400EncoderFree": ([ctypes.c_void_p], None),
            "mbe_encodeAmbe2400ParmsShort": ([ctypes.c_void_p, pcm, bits, parms, parms], ctypes.c_int),
            "mbe_encodeAmbe3600x2400Frame": ([bits, frame], ctypes.c_int),
            "mbe_encodeDStarDVData": ([frame, packed], ctypes.c_int),
            "mbe_decodeDStarDVData": ([packed, frame], ctypes.c_int),
            "mbe_decodeAmbe3600x2400Frame": ([frame, bits, status], ctypes.c_int),
            "mbe_processAmbe3600x2400Frame": ([pcm, status, frame, bits, parms, parms, parms], ctypes.c_int),
            "mbe_processAmbe2400Data": ([pcm, status, bits, parms, parms, parms], ctypes.c_int),
            "mbe_formatProcessResult": ([bits, ctypes.c_size_t, status], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(dll, name)
            function.argtypes = arguments
            function.restype = result

    @property
    def version(self) -> str:
        return f"mbelib-neo {self._version}"

    @property
    def profile(self) -> str:
        return f"{self.version}; D-STAR AMBE 3600x2400; native PCM16"

    @property
    def bridge_version(self) -> str:
        return self.version

    @property
    def mbelib_version(self) -> str:
        return self._version

    def _require_open(self) -> None:
        if self._closed:
            raise DecoderError("mbelib-neo codec is closed")

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status < 0:
            raise DecoderError(f"mbelib-neo {operation} failed with error {status}")

    def begin_stream(self, seed: int) -> None:
        with self._lock:
            self._require_open()
            self._rng_seed = (int(seed) & 0xFFFFFFFF) or 1
            self._dll.mbe_initMbeParms(*(ctypes.byref(p) for p in self._decode_state))

    def begin_encode_stream(self) -> None:
        with self._lock:
            self._require_open()
            self._dll.mbe_ambe2400EncoderReset(self._encoder)
            self._dll.mbe_initMbeParms(*(ctypes.byref(p) for p in self._encode_state))

    def set_gain(self, gain: float) -> None:
        with self._lock:
            self._gain = self._validated_gain(gain)

    def _unpack(self, frame: bytes, order: BitOrder) -> _Frame:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"AMBE frame must be {FRAME_BYTES} bytes, got {len(frame)}")
        order = BitOrder(order)
        if order == BitOrder.MSB_FIRST:
            frame = frame.translate(_REVERSE_BITS)
        native_frame = _Frame()
        self._check(self._dll.mbe_decodeDStarDVData(_Packed.from_buffer_copy(frame), native_frame), "unpack")
        return native_frame

    def score_frame(self, frame: bytes, order: BitOrder) -> FecScore:
        """Measure FEC without changing receive prediction or synthesis state."""
        with self._lock:
            self._require_open()
            native_frame = self._unpack(frame, order)
            bits, result = _Bits(), _ProcessResult()
            self._check(self._dll.mbe_decodeAmbe3600x2400Frame(native_frame, bits, ctypes.byref(result)), "FEC")
            # Re-encoding includes parity errors omitted from data correction
            # counts, so automatic bit-order selection has real evidence.
            repaired, packed = _Frame(), _Packed()
            self._check(self._dll.mbe_encodeAmbe3600x2400Frame(bits, repaired), "FEC re-encode")
            self._check(self._dll.mbe_encodeDStarDVData(repaired, packed), "pack")
            received = frame.translate(_REVERSE_BITS) if order == BitOrder.MSB_FIRST else frame
            distance = sum((a ^ b).bit_count() for a, b in zip(received, packed))
            parity = max(0, distance - result.total_errors)
            return FecScore(result.c0_errors, result.protected_errors, result.total_errors, parity, distance)

    def _seed_synthesis(self) -> None:
        # Seed on the synthesizing thread. Per-frame seeds keep comfort noise
        # independent across instances sharing a thread. Continuous unvoiced
        # noise state lives in each instance's mbe_parms triplet.
        self._dll.mbe_setThreadRngSeed(self._rng_seed)
        self._rng_seed = (1664525 * self._rng_seed + 1013904223) & 0xFFFFFFFF

    def _result(self, pcm: _Pcm, result: _ProcessResult, ambe_frame: bytes | None = None) -> DecodeResult:
        output, clipped = self._scale_pcm(pcm, self._gain)
        markers = ctypes.create_string_buffer(64)
        self._dll.mbe_formatProcessResult(markers, len(markers), ctypes.byref(result))
        return DecodeResult(output, result.c0_errors, result.protected_errors, result.total_errors,
                            clipped, markers=markers.value.decode("ascii"),
                            ambe_frame=ambe_frame, gain=self._gain)

    def decode_frame(self, frame: bytes, order: BitOrder) -> DecodeResult:
        with self._lock:
            self._require_open()
            native_frame = self._unpack(frame, order)
            pcm, bits, result = _Pcm(), _Bits(), _ProcessResult()
            self._seed_synthesis()
            self._check(self._dll.mbe_processAmbe3600x2400Frame(
                pcm, ctypes.byref(result), native_frame, bits,
                *(ctypes.byref(p) for p in self._decode_state)), "decode")
            canonical = frame.translate(_REVERSE_BITS) if order == BitOrder.MSB_FIRST else frame
            return self._result(pcm, result, canonical)

    def encode_frame(self, pcm: bytes) -> bytes:
        if len(pcm) != PCM_SAMPLES * 2:
            raise ValueError(f"PCM frame must be {PCM_SAMPLES * 2} bytes, got {len(pcm)}")
        with self._lock:
            self._require_open()
            bits, frame, packed = _Bits(), _Frame(), _Packed()
            current, previous, _ = self._encode_state
            self._check(self._dll.mbe_encodeAmbe2400ParmsShort(
                self._encoder, _Pcm.from_buffer_copy(pcm), bits,
                ctypes.byref(current), ctypes.byref(previous)), "encode")
            self._check(self._dll.mbe_encodeAmbe3600x2400Frame(bits, frame), "frame encode")
            self._check(self._dll.mbe_encodeDStarDVData(frame, packed), "pack")
            self._dll.mbe_moveMbeParms(ctypes.byref(current), ctypes.byref(previous))
            return bytes(packed)

    def conceal_frame(self) -> DecodeResult:
        """Advance mbelib-neo's three-repeat/fourth-frame mute path on loss."""
        with self._lock:
            self._require_open()
            pcm, bits = _Pcm(), _Bits()  # b0=0 is voice, never a tone/silence sentinel.
            # Uncorrectable data triggers the native bounded repeat policy.
            # These counts describe loss, not received FEC errors.
            result = _ProcessResult(c0_errors=4, total_errors=4, flags=0x0002)
            self._seed_synthesis()
            self._check(self._dll.mbe_processAmbe2400Data(
                pcm, ctypes.byref(result), bits,
                *(ctypes.byref(p) for p in self._decode_state)), "conceal")
            return self._result(pcm, result)

    def close(self) -> None:
        with self._lock:
            if self._encoder:
                self._dll.mbe_ambe2400EncoderFree(self._encoder)
                self._encoder = None
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _validated_gain(gain: float) -> float:
        value = float(gain)
        if not math.isfinite(value) or not 0.0 <= value <= 10.0:
            raise ValueError("audio gain must be finite and between 0 and 10")
        return value

    @staticmethod
    def _scale_pcm(samples: ctypes.Array, gain: float) -> tuple[bytes, int]:
        if gain == 1.0:
            return bytes(samples), 0
        output = _Pcm()
        clipped = 0
        for index, sample in enumerate(samples):
            value = round(sample * gain)
            if value > 32767:
                value = 32767
                clipped += 1
            elif value < -32768:
                value = -32768
                clipped += 1
            output[index] = value
        return bytes(output), clipped
