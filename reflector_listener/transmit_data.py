"""Wire scheduling for picture notification tones and optional GPS-A positions."""

from __future__ import annotations

from itertools import chain, repeat
import re
from typing import Iterator

from .location import coordinates
from .protocols import NULL_AMBE_FRAME
from .rsms1 import FAST_FILL_DV_PAYLOAD, iter_fast_data_frames


# Captured ID-52 stream 0DC3, raw_rx_20261008_100140_004430.log:54877.
# Two tone frames, two silence frames, two tone frames. These are AMBE tone
# codewords (b0=126), not microphone-encoder approximations of a sine wave.
_TONE = bytes.fromhex("f3 cf 38 e2 c7 3c 11 45 0c")
_SILENCE = (bytes.fromhex("aa cc 2a 78 e1 13 3c 67 c0"),
            bytes.fromhex("ce 8e 2e 39 66 90 70 31 b0"))
PICTURE_BEEP = tuple(frame + bytes.fromhex("16 29 f5")
                     for frame in (_TONE, _TONE, *_SILENCE, _TONE, _TONE))


def application_frames(data: bytes, beep: bool = False, station_message: str = "") -> Iterator[tuple[bytes, bool]]:
    """Yield payload and progress flag; insert tones only between complete pairs.

    The session inserts sequence-zero sync frames. Count them here too, so a
    burst begins after about 50 wire frames, independent of data length.
    Six even-numbered slots preserve A/B alignment across modulo-21 wraps.
    """
    wire_count = 0
    next_beep = 50
    sequence = 0
    # Send a complete ordinary slow-data cycle before Fast Data takes over.
    for slow in station_message_cycle(station_message):
        if sequence == 0:
            wire_count += 1
            sequence = 1
        yield NULL_AMBE_FRAME + slow, False
        wire_count += 1
        sequence = (sequence + 1) % 21
    payloads = chain(repeat(FAST_FILL_DV_PAYLOAD, 4), iter_fast_data_frames(data),
                     repeat(FAST_FILL_DV_PAYLOAD, 4))
    for index, payload in enumerate(payloads):
        if beep and index >= 4 and index % 2 == 0 and wire_count >= next_beep:
            for tone in PICTURE_BEEP:
                if sequence == 0:
                    wire_count += 1
                    sequence = 1
                yield tone, False
                wire_count += 1
                sequence = (sequence + 1) % 21
            next_beep = wire_count - len(PICTURE_BEEP) + 50
        if sequence == 0:
            wire_count += 1
            sequence = 1
        yield payload, True
        wire_count += 1
        sequence = (sequence + 1) % 21


def gpsa_crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return crc ^ 0xFFFF


def _aprs_coordinate(value: float, latitude: bool) -> str:
    # Work in hundredths of a minute to carry rounding through 60 minutes.
    total = round(abs(value) * 6000)
    degrees, hundredths = divmod(total, 6000)
    direction = ("N" if value >= 0 else "S") if latitude else ("E" if value >= 0 else "W")
    return f"{degrees:0{2 if latitude else 3}d}{hundredths / 100:05.2f}{direction}"


def gpsa_frame(callsign: str, position: tuple[float, float]) -> bytes:
    """GPS-A/D-PRS CRC covers the complete APRS line, including its CR.

    Format reference: https://www.aprs-is.net/DPRS%20Parser.aspx
    """
    lat, lon = coordinates(*position)
    source = callsign.strip().upper()
    if not re.fullmatch(r"[A-Z0-9/]{3,8}", source):
        raise ValueError("A valid call sign is required for D-PRS.")
    body = (f"{source}>APRS,DSTAR*:!{_aprs_coordinate(lat, True)}/"
            f"{_aprs_coordinate(lon, False)}-reflectorListen\r").encode("ascii")
    return f"$$CRC{gpsa_crc(body):04X},".encode("ascii") + body


def slow_data_cycle(data: bytes) -> tuple[bytes, ...]:
    """Frame serial bytes in five-byte low-speed records, aligned to 20 slots."""
    result = []
    for start in range(0, len(data), 5):
        chunk = data[start:start + 5]
        record = bytes((0x30 | len(chunk),)) + chunk.ljust(5, b"f")
        for part in (record[:3], record[3:]):
            result.append(bytes(value ^ mask for value, mask in zip(part, b"\x70\x4f\x93")))
    while len(result) % 20:
        result.append(bytes.fromhex("16 29 f5"))
    return tuple(result)


def station_message_cycle(message: str) -> tuple[bytes, ...]:
    """Encode the four D-STAR 0x40–0x43 text blocks in one 20-slot cycle."""
    if not message:
        return ()
    if not re.fullmatch(r"[\x20-\x7e]{1,20}", message):
        raise ValueError("Station message must be at most 20 printable ASCII characters.")
    text = message.encode("ascii").ljust(20, b" ")
    slots = []
    for index in range(4):
        block = bytes((0x40 + index,)) + text[index * 5:index * 5 + 5]
        for half in (block[:3], block[3:]):
            slots.append(bytes(value ^ mask for value, mask in zip(half, b"\x70\x4f\x93")))
    return tuple(slots + [bytes.fromhex("16 29 f5")] * 12)
