"""Small PCM16 helpers shared by microphone gain and the live meters."""
from array import array
import math


def validated_gain(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Microphone gain must be a number between 0 and 10")
    gain = float(value)
    if not math.isfinite(gain) or not 0 <= gain <= 10:
        raise ValueError("Microphone gain must be a number between 0 and 10")
    return gain


def scale_pcm16(pcm: bytes, gain: float) -> bytes:
    """Apply gain once, saturating rather than wrapping at PCM16 limits."""
    if gain == 1.0:
        return pcm
    if len(pcm) % 2:
        raise ValueError("PCM16 data must contain complete samples")
    samples = array('h', pcm)
    return array('h', (max(-32768, min(32767, round(x * gain))) for x in samples)).tobytes()


def pcm16_level(pcm: bytes) -> float:
    """Return normalized RMS amplitude; callers can display this on a dB scale."""
    samples = array('h', pcm)
    if not samples:
        return 0.0
    return min(1.0, math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768.0)
