"""RS-MS1 text/picture framing and D-Star application-data transport.

The frame formats and Fast Data byte map mirror the T-Beam hotspot and the
RS-MS1A reverse-lab captures.  This module deliberately stays independent of
reflector socket framing so it can also be exercised with captured DV payloads.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageOps

from reflector_listener.diagnostics import DiagnosticLogger, SafeDiagnostics


_SCRAMBLER = bytes.fromhex("70 4f 93 40 64 74 6d 30 2b 70 4f 93")
_RESERVED = frozenset((0x00, 0x0D, 0x11, 0x13, 0x1A, 0x24, 0x2C,
                       0xE7, 0xEF, 0xFD, 0xFE, 0xFF))
_NULL_AMBE = bytes.fromhex("4d b2 44 12 03 68 14 64 13")
_SLOW_IDLE = bytes.fromhex("16 29 f5")
FAST_FILL_DV_PAYLOAD = _NULL_AMBE + _SLOW_IDLE
FAST_SYNC_DV_PAYLOAD = bytes.fromhex("70 4f 93 40 66 74 6d 30 2b 55 2d 16")
_FRAME_MARKERS = (b"$$Msg,", b"$$Pic,", b"$GPRMC,", b"$GPGGA,", b"$$CRC")
_SEQUENCE_SLOTS = 21
_FAST_REORDER_SLOTS = 4

PICTURE_SIZE_OPTIONS = ("160x120", "320x240", "640x480")
PICTURE_QUALITY_OPTIONS = (25, 50, 75)
CROP_OPTIONS = (
    "Fit / letterbox",
    "Fill / center crop",
    "Crop from top",
    "Crop from bottom",
    "Crop from left",
    "Crop from right",
    "Stretch",
)


@dataclass(frozen=True, slots=True)
class PicturePreset:
    label: str
    width: int
    height: int
    tiles_x: int
    tiles_y: int
    quality: int
    record: int

    @property
    def tile_count(self) -> int:
        return self.tiles_x * self.tiles_y


def picture_preset(size: str, quality: int) -> PicturePreset:
    label = str(size).strip()
    q = int(quality)
    if label not in PICTURE_SIZE_OPTIONS:
        raise ValueError(f"unsupported picture size: {label}")
    if q not in PICTURE_QUALITY_OPTIONS:
        raise ValueError("picture quality must be 25, 50, or 75")
    quality_offset = PICTURE_QUALITY_OPTIONS.index(q)
    if label == "160x120":
        # RS-MS1 labels this 160x120, but transports an 8-row 160x128 canvas.
        return PicturePreset(label, 160, 128, 10, 8, q, quality_offset)
    if label == "320x240":
        return PicturePreset(label, 320, 240, 20, 15, q, 0x08 + quality_offset)
    return PicturePreset(label, 640, 480, 40, 30, q, 0x10 + quality_offset)


def preset_for_record(record: int) -> PicturePreset | None:
    value = int(record) & 0xFF
    for size in PICTURE_SIZE_OPTIONS:
        for quality in PICTURE_QUALITY_OPTIONS:
            preset = picture_preset(size, quality)
            if preset.record == value:
                return preset
    return None


def normalize_payload_callsign(value: str, fallback: str = "") -> str:
    text = str(value or fallback).strip().upper()[:8]
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("call signs must contain ASCII characters") from exc
    if any(ord(ch) < 0x20 or ord(ch) > 0x7E or ch in ",\r\n" for ch in text):
        raise ValueError("call signs contain an unsupported character")
    return text


def _ascii_sum(value: str) -> int:
    return sum(value.encode("ascii")) & 0xFF


def text_control(source: str, destination: str) -> str:
    check = (0x1A + _ascii_sum(source) + _ascii_sum(destination)) & 0xFF
    return f"0011{check:02X}"


def picture_control(source: str, destination: str, number: int) -> str:
    picture_hex = f"{number & 0xFF:02X}"
    check = (
        0x58
        + _ascii_sum(source)
        + _ascii_sum(destination)
        + _ascii_sum(picture_hex)
    ) & 0xFF
    return picture_hex + f"{check:02X}"


def _picture_address_matches(source: str, destination: str, control: str) -> bool:
    # IC-9700 uses uppercase picture numbers; RS-MS1A uses lowercase. The
    # checksum covers the actual ASCII number bytes, not its numeric value.
    check = (0x58 + _ascii_sum(source) + _ascii_sum(destination)
             + _ascii_sum(control[:2])) & 0xFF
    return int(control[2:], 16) == check


def escape_bytes(data: bytes) -> bytes:
    output = bytearray()
    for value in data:
        if value in _RESERVED:
            output.extend((0xEF, value ^ 0x80))
        else:
            output.append(value)
    return bytes(output)


def unescape_bytes(data: bytes) -> bytes:
    output = bytearray()
    index = 0
    while index < len(data):
        value = data[index]
        if value == 0xEF:
            index += 1
            if index >= len(data):
                raise ValueError("truncated RS-MS1 escape")
            value = data[index] ^ 0x80
        output.append(value)
        index += 1
    return bytes(output)


def build_text_frame(
    source: str,
    destination: str,
    text: str,
    *,
    picture_number: int | None = None,
) -> bytes:
    source = normalize_payload_callsign(source, "MYCALL")
    destination = normalize_payload_callsign(destination)
    clean_text = str(text).replace("\r", " ").replace("\n", " ").strip()
    if not clean_text:
        raise ValueError("message is empty")
    if len(clean_text) > 128:
        raise ValueError("message is longer than 128 characters")
    body = clean_text.encode("utf-8")
    payload = body + bytes((sum(body) & 0xFF,))
    suffix = b""
    if picture_number is not None:
        number_hex = f"{picture_number & 0xFF:02X}".encode("ascii")
        # The hotspot leaves the picture-binding suffix unescaped.
        suffix = b"," + number_hex + bytes((sum(number_hex) & 0xFF,))
    prefix = f"$$Msg,{source},{destination},{text_control(source, destination)}".encode("ascii")
    return prefix + escape_bytes(payload) + suffix + b"\r\0"


def build_picture_tile_frame(
    source: str,
    destination: str,
    number: int,
    preset: PicturePreset,
    x: int,
    y: int,
    jpeg_scan: bytes,
) -> bytes:
    source = normalize_payload_callsign(source, "MYCALL")
    destination = normalize_payload_callsign(destination, "CQCQCQ")
    if not 0 <= x < preset.tiles_x or not 0 <= y < preset.tiles_y:
        raise ValueError("picture tile coordinate is outside the selected size")
    if not jpeg_scan:
        raise ValueError("picture tile JPEG data is empty")
    raw = bytes((preset.record, x, y)) + bytes(jpeg_scan)
    raw += bytes((sum(raw) & 0xFF,))
    prefix = f"$$Pic,{source},{destination},{picture_control(source, destination, number)}".encode("ascii")
    return prefix + escape_bytes(raw) + b"\r"


def _jpeg_scan_offset(jpeg: bytes) -> int:
    marker = jpeg.find(b"\xff\xda")
    if marker < 0 or marker + 4 > len(jpeg):
        raise ValueError("JPEG does not contain a start-of-scan marker")
    segment_length = int.from_bytes(jpeg[marker + 2 : marker + 4], "big")
    offset = marker + 2 + segment_length
    if offset > len(jpeg):
        raise ValueError("JPEG start-of-scan segment is truncated")
    return offset


def _encode_tile(tile: Image.Image, quality: int) -> bytes:
    # Headerless RS-MS1 tiles require the IJG/libjpeg table profile selected
    # by the record's 25/50/75 quality class. Do not use input-file tables or
    # encoder presets: the radio never receives DQT/DHT markers for a tile.
    output = io.BytesIO()
    tile.convert("RGB").save(
        output,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    jpeg = output.getvalue()
    start = _jpeg_scan_offset(jpeg)
    end = jpeg.rfind(b"\xff\xd9")
    if end < start:
        raise ValueError("JPEG does not contain an end marker")
    return jpeg[start:end]


@lru_cache(maxsize=3)
def _jpeg_prefix(quality: int) -> bytes:
    # Must match _encode_tile's table and sampling profile. An IC-9700's
    # saved whole-image JPEG can use Q75 even after receiving Q50 tiles;
    # choose these tables from the on-air record, not that saved JPEG.
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(
        output,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    jpeg = output.getvalue()
    return jpeg[: _jpeg_scan_offset(jpeg)]


def _decode_tile(scan: bytes, quality: int) -> Image.Image:
    with Image.open(io.BytesIO(_jpeg_prefix(quality) + scan + b"\xff\xd9")) as image:
        return image.convert("RGB").copy()


def resize_picture(image: Image.Image, preset: PicturePreset, crop: str) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    nominal_target = (preset.width, 120) if preset.label == "160x120" else (preset.width, preset.height)
    mode = str(crop).strip()
    if mode == "Stretch":
        resized = source.resize(nominal_target, Image.Resampling.LANCZOS)
    elif mode == "Fit / letterbox":
        fitted = ImageOps.contain(source, nominal_target, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", nominal_target, "white")
        canvas.paste(fitted, ((nominal_target[0] - fitted.width) // 2, (nominal_target[1] - fitted.height) // 2))
        resized = canvas
    else:
        centering = {
            "Fill / center crop": (0.5, 0.5),
            "Crop from top": (0.5, 0.0),
            "Crop from bottom": (0.5, 1.0),
            "Crop from left": (0.0, 0.5),
            "Crop from right": (1.0, 0.5),
        }.get(mode)
        if centering is None:
            raise ValueError(f"unsupported crop mode: {mode}")
        resized = ImageOps.fit(source, nominal_target, Image.Resampling.LANCZOS, centering=centering)

    if preset.label == "160x120":
        # Pad bottom 8 rows to form the full 160x128 radio tile canvas (matching RS-MS1 standard)
        canvas = Image.new("RGB", (preset.width, preset.height), "black")
        canvas.paste(resized, (0, 0))
        return canvas
    return resized


@dataclass(frozen=True, slots=True)
class PreparedPicture:
    image: Image.Image
    payload: bytes
    preset: PicturePreset
    number: int
    caption: str


def prepare_picture(
    path: str | Path,
    source: str,
    destination: str,
    size: str,
    quality: int,
    crop: str,
    number: int,
    caption: str = "",
) -> PreparedPicture:
    preset = picture_preset(size, quality)
    with Image.open(Path(path)) as opened:
        image = resize_picture(opened, preset, crop)
    frames: list[bytes] = []
    clean_caption = str(caption).replace("\r", " ").replace("\n", " ").strip()
    if clean_caption:
        frames.append(
            build_text_frame(
                source,
                destination,
                clean_caption,
                picture_number=number,
            )
        )
    for y in range(preset.tiles_y):
        for x in range(preset.tiles_x):
            box = (x * 16, y * 16, x * 16 + 16, y * 16 + 16)
            frames.append(
                build_picture_tile_frame(
                    source,
                    destination,
                    number,
                    preset,
                    x,
                    y,
                    _encode_tile(image.crop(box), preset.quality),
                )
            )
    return PreparedPicture(image.copy(), b"".join(frames), preset, number & 0xFF, clean_caption)


def _xor_scrambler(payload: bytes) -> bytes:
    if len(payload) != 12:
        raise ValueError("a DV payload must contain exactly 12 bytes")
    return bytes(value ^ mask for value, mask in zip(payload, _SCRAMBLER))


def iter_fast_data_frames(data: bytes | bytearray | memoryview) -> Iterator[bytes]:
    """Yield hotspot-compatible scrambled DV payloads for application bytes."""

    source = bytes(data)
    if not source:
        return
    for offset in range(0, len(source), 20):
        chunk = source[offset : offset + 20]
        length = len(chunk)
        # 0x83 denotes the second half, so a three-byte final A record would
        # be ambiguous. One idle byte keeps the application terminator intact.
        if length == 3:
            length = 4
        message = chunk.ljust(20, b"\x66")
        even = bytearray(12)
        odd = bytearray(12)
        even[0:4] = message[4:8]
        even[4] = 0x02
        even[5:9] = message[8:12]
        even[9] = (0x90 | (length - 16)) if length > 15 else (0x80 | length)
        even[10:12] = message[0:2]
        odd[0:4] = message[12:16]
        odd[4] = 0x02
        odd[5:9] = message[16:20]
        odd[9] = 0x83
        odd[10:12] = message[2:4]
        yield _xor_scrambler(bytes(even))
        yield _xor_scrambler(bytes(odd))


def is_fast_data_payload(payload: bytes | None) -> bool:
    """Return whether a 12-byte DV payload matches RS-MS1 Fast Data framing."""
    if payload is None or len(payload) != 12:
        return False
    plain = _xor_scrambler(payload)
    if plain[4] != 0x02:
        return False
    if (plain[9] & 0xF0) in (0x80, 0x90) or plain[9] == 0x83:
        return True
    # A sync trailer belongs to ordinary voice too; a single 0x02 byte in
    # AMBE audio is not evidence of Fast Data. Only this canonical all-zero
    # fast sync is recognizable without neighboring frames.
    return payload == FAST_SYNC_DV_PAYLOAD


@dataclass(frozen=True, slots=True)
class RsMs1Event:
    kind: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PictureState:
    source: str
    destination: str
    number: int
    preset: PicturePreset
    image: Image.Image
    received: set[tuple[int, int]] = field(default_factory=set)


@dataclass(slots=True)
class _TransportState:
    buffer: bytearray = field(default_factory=bytearray)
    pending_a: dict[tuple[int, int], bytes] = field(default_factory=dict)
    pending_b: dict[tuple[int, int], bytes] = field(default_factory=dict)
    pending_seq00: dict[int, bytes] = field(default_factory=dict)
    last_seq: int | None = None
    superframe_cycle: int = 0
    ready_chunks: dict[int, bytes | None] = field(default_factory=dict)
    idle_slots: set[tuple[int, int]] = field(default_factory=set)
    idle_chunks: set[int] = field(default_factory=set)
    marker_candidates: set[tuple[int, int]] = field(default_factory=set)
    last_strict_pair_slot: int | None = None
    picture_active: bool = False
    strict_text_checksum: bool = False
    next_chunk_id: int | None = None
    slow_remaining: int = 0
    slow_even_sequence: int | None = None
    slow_header: int | None = None
    last_sequence: int | None = None
    active: bool = False
    text_chunks: dict[int, bytearray] = field(default_factory=dict)
    last_text_emitted: str = ""


class RsMs1Receiver:
    """Incrementally decode normal and Fast Data RS-MS1 streams.

    This class reassembles fragmented picture and text frames from
    incoming D-Star packets, handling sequence tracking, unscrambling,
    and picture tile assembly over time.
    """

    def __init__(self, logger: DiagnosticLogger | None = None) -> None:
        self._streams: dict[int, _TransportState] = {}
        self._pictures: dict[str, _PictureState] = {}
        self._captions: dict[tuple[str, str, int], str] = {}
        self._logger = SafeDiagnostics(logger)

    def set_logger(self, logger: DiagnosticLogger | None) -> None:
        self._logger = SafeDiagnostics(logger)

    def reset(self) -> None:
        """Clear all active transport streams and picture assembly buffers."""
        self._streams.clear()
        self._pictures.clear()
        self._captions.clear()

    def start_stream(self, stream_id: int) -> None:
        """Initialize state for a newly begun D-Star stream."""
        self._streams[stream_id] = _TransportState()

    def end_stream(self, stream_id: int) -> list[RsMs1Event]:
        """Finalize a stream, flushing any remaining buffered data."""
        state = self._streams.pop(stream_id, None)
        events: list[RsMs1Event] = []
        if state is not None:
            self._match_pending_fast_frames(state, stream_id, final=True)
            for chunk_id in sorted(state.ready_chunks):
                chunk = state.ready_chunks[chunk_id]
                if chunk is None:
                    state.buffer.clear()
                else:
                    events.extend(self._feed_bytes(state, chunk))
            state.ready_chunks.clear()
            events.extend(self._feed_bytes(state, b""))
        return events

    def _update_sequence_cycle(self, state: _TransportState, sequence: int) -> int:
        """Map a packet to the nearest cycle without moving progress backward.

        Sequence numbers alone distinguish reordering only within half of the
        21-slot superframe. Late packets near a wrap belong to the prior cycle.
        """
        if state.last_seq is None:
            state.last_seq = sequence
            return state.superframe_cycle

        forward = (sequence - state.last_seq) % _SEQUENCE_SLOTS
        if 0 < forward <= _SEQUENCE_SLOTS // 2:
            if sequence < state.last_seq:
                state.superframe_cycle += 1
            state.last_seq = sequence
        elif forward > _SEQUENCE_SLOTS // 2 and sequence > state.last_seq:
            return state.superframe_cycle - 1
        return state.superframe_cycle

    def _route_fast_data_frame(self, state: _TransportState, plain: bytes, sequence: int, pkt_cycle: int) -> tuple[bool, bool]:
        is_fast_a = False
        is_fast_b = False
        if plain[4] == 0x02:
            if (plain[9] & 0xF0) in (0x80, 0x90) and plain[9] != 0x83:
                is_fast_a = True
            elif plain[9] == 0x83:
                expected_for_waiting_a = False
                for (ca, sa) in state.pending_a:
                    exp_b = (ca + 1, 1) if sa == 20 else (ca, sa + 1)
                    if exp_b == (pkt_cycle, sequence):
                        expected_for_waiting_a = True
                        break
                if expected_for_waiting_a:
                    is_fast_b = True
                elif sequence % 2 == 1:
                    is_fast_a = True
                else:
                    is_fast_b = True
        return is_fast_a, is_fast_b

    def _damaged_fast_role(
        self, state: _TransportState, plain: bytes, sequence: int, cycle: int
    ) -> tuple[bool, bool]:
        """Recognize one damaged nonpayload marker only inside a known picture.

        This is a candidate, never evidence to start or mute a Fast Data stream.
        Its partner must be strict, and neither payload nor length is repaired.
        """
        slot = cycle * _SEQUENCE_SLOTS + sequence
        if (not state.picture_active or state.last_strict_pair_slot is None
                or not 0 <= slot - state.last_strict_pair_slot <= _SEQUENCE_SLOTS):
            return False, False
        flag_damage = (plain[4] ^ 0x02).bit_count() == 1
        if sequence & 1:
            return flag_damage and plain[9] == 0x94, False
        # Only the observed lost high bit of the B tag is unambiguous;
        # other one-bit changes can also be legitimate A length tags.
        second_half = ((flag_damage and plain[9] == 0x83)
                       or (plain[4] == 0x02 and plain[9] == 0x03))
        return False, second_half

    def _assemble_fast_chunks(self, state: _TransportState, stream_id: int) -> list[RsMs1Event]:
        """Combine pending A/B DV frames into application data chunks."""
        self._match_pending_fast_frames(state, stream_id)
        events = self._process_ready_chunks(state, stream_id)
        self._prune_old_fast_frames(state)
        return events

    def _match_pending_fast_frames(
        self, state: _TransportState, stream_id: int, *, final: bool = False
    ) -> None:
        """Find matching pairs, waiting briefly for a reordered sync extension."""
        for (cycle_a, seq_a) in list(state.pending_a):
            first = (cycle_a, seq_a)
            expected_b = (cycle_a + 1, 1) if seq_a == 20 else (cycle_a, seq_a + 1)
            if expected_b not in state.pending_b:
                continue
            if not self._accept_fast_pair_markers(state, stream_id, first, expected_b):
                continue
            ready, data = self._decode_fast_pair(state, stream_id, first, expected_b, final=final)
            if not ready:
                continue
            del state.pending_a[first]
            del state.pending_b[expected_b]
            state.marker_candidates.discard(first)
            state.marker_candidates.discard(expected_b)
            slot = max(0, min(9, (seq_a - 1) // 2))
            chunk_id = cycle_a * 10 + slot
            if state.next_chunk_id is None or chunk_id >= state.next_chunk_id:
                state.ready_chunks[chunk_id] = data

    def _accept_fast_pair_markers(
        self, state: _TransportState, stream_id: int,
        first: tuple[int, int], second: tuple[int, int],
    ) -> bool:
        """Apply marker-damage policy before consuming a complete A/B pair."""
        cycle_a, seq_a = first
        candidate_a = first in state.marker_candidates
        candidate_b = second in state.marker_candidates
        if not candidate_a and not candidate_b:
            state.last_strict_pair_slot = cycle_a * _SEQUENCE_SLOTS + seq_a
            return True
        # Require ordinary adjacent A/B slots and one unchanged partner. Full
        # 20-byte A lengths avoid guessing a short or sync-extended payload.
        if candidate_a == candidate_b or not seq_a & 1 or state.pending_a[first][9] != 0x94:
            return False
        candidate_slot = first if candidate_a else second
        self._logger.log_event(
            "rx_marker_tolerance",
            "Retained picture bytes after one-bit Fast Data marker damage; application checks required",
            stream_id=stream_id, sequence=candidate_slot[1],
            half="A" if candidate_a else "B",
        )
        state.strict_text_checksum = True
        return True

    def _decode_fast_pair(
        self, state: _TransportState, stream_id: int,
        first: tuple[int, int], second: tuple[int, int], *, final: bool,
    ) -> tuple[bool, bytes | None]:
        """Return readiness and chunk bytes; a ready ``None`` is an erasure.

        Sync extensions may arrive after their A/B pair. Keep that pair pending
        until the reorder window expires, then mark the missing bytes in order.
        """
        cycle_a, seq_a = first
        plain_a = state.pending_a[first]
        plain_b = state.pending_b[second]
        length = (plain_a[9] & 0x0F) + (16 if (plain_a[9] & 0xF0) == 0x90 else 0)
        extra_len = length - 20 if seq_a == 1 and 21 <= length <= 24 else 0
        if extra_len and cycle_a not in state.pending_seq00:
            latest_slot = state.superframe_cycle * _SEQUENCE_SLOTS + (state.last_seq or 0)
            pair_slot = cycle_a * _SEQUENCE_SLOTS + seq_a
            if not final and latest_slot - pair_slot <= _FAST_REORDER_SLOTS:
                return False, None
            self._logger.log_gap(
                "rx-seq00", stream_id, expected=cycle_a, actual=None,
                note=f"missing {extra_len} trailing bytes from sequence 0; discarded affected chunk"
            )
            # An erasure clears the partial application frame in order. Do not
            # splice a truncated chunk into the next tile or message.
            return True, None
        data = (
            plain_a[10:12] + plain_b[10:12] + plain_a[0:4] +
            plain_a[5:9] + plain_b[0:4] + plain_b[5:9]
        )[:min(20, length)]
        if extra_len:
            data += state.pending_seq00.pop(cycle_a)[:extra_len]
        return True, data

    def _process_ready_chunks(self, state: _TransportState, stream_id: int) -> list[RsMs1Event]:
        """Align chunk IDs, log gaps, and emit bytes into the protocol parser."""
        events = []
        if state.ready_chunks:
            if state.next_chunk_id is None:
                if 0 in state.ready_chunks or len(state.ready_chunks) >= 2 or (max(state.ready_chunks) - min(state.ready_chunks) >= 2):
                    state.next_chunk_id = min(state.ready_chunks)
            elif min(state.ready_chunks) > state.next_chunk_id:
                gap = min(state.ready_chunks) - state.next_chunk_id
                if gap >= 2 or len(state.ready_chunks) >= 2:
                    idle = sum(chunk in state.idle_chunks
                               for chunk in range(state.next_chunk_id, min(state.ready_chunks)))
                    missing = gap - idle
                    if missing:
                        self._logger.log_gap(
                            "rx-chunk", stream_id, expected=state.next_chunk_id,
                            actual=min(state.ready_chunks),
                            note=f"{missing} slots without complete Fast Data pairs; may include slow-data interludes ({idle} confirmed idle pairs excluded)"
                        )
                    state.next_chunk_id = min(state.ready_chunks)

            if state.next_chunk_id is not None:
                while state.next_chunk_id in state.ready_chunks:
                    chunk_data = state.ready_chunks.pop(state.next_chunk_id)
                    if chunk_data is None:
                        state.buffer.clear()
                    else:
                        events.extend(self._feed_bytes(state, chunk_data))
                    state.next_chunk_id += 1
        return events

    def _prune_old_fast_frames(self, state: _TransportState) -> None:
        """Discard out-of-sequence frame data that is too old to be completed."""
        for k in list(state.pending_a):
            if state.superframe_cycle - k[0] > 2:
                del state.pending_a[k]
        for k in list(state.pending_b):
            if state.superframe_cycle - k[0] > 2:
                del state.pending_b[k]
        for k in list(state.pending_seq00):
            if state.superframe_cycle - k > 2:
                del state.pending_seq00[k]

        state.marker_candidates.intersection_update(state.pending_a.keys() | state.pending_b.keys())
        state.idle_slots = {key for key in state.idle_slots
                            if state.superframe_cycle - key[0] <= 2}
        state.idle_chunks = {chunk for chunk in state.idle_chunks
                             if state.superframe_cycle - chunk // 10 <= 2}

        if len(state.pending_a) > 40:
            del state.pending_a[min(state.pending_a)]
        if len(state.pending_b) > 40:
            del state.pending_b[min(state.pending_b)]
        if len(state.pending_seq00) > 10:
            del state.pending_seq00[min(state.pending_seq00)]

    def feed(
        self,
        stream_id: int,
        sequence: int,
        dv_payload: bytes,
    ) -> tuple[bool, list[RsMs1Event]]:
        """Process one incoming DV payload.

        Returns a tuple `(claimed, events)` where `claimed` is True if the
        payload was identified as Fast Data (meaning the caller should mute audio),
        and `events` is a list of completed picture/text events.
        """
        if len(dv_payload) != 12 or not 0 <= sequence < _SEQUENCE_SLOTS:
            return False, []
        state = self._streams.setdefault(stream_id, _TransportState())
        if state.last_sequence == sequence:
            return state.active, []
        state.last_sequence = sequence

        pkt_cycle = self._update_sequence_cycle(state, sequence)
        plain = _xor_scrambler(dv_payload)
        if sequence == 0:
            return state.active, self._feed_sync_frame(state, stream_id, pkt_cycle, plain)

        if self._cache_fast_frame(state, plain, sequence, pkt_cycle):
            # A reordered B half is cached, but cannot by itself claim voice.
            return state.active, self._assemble_fast_chunks(state, stream_id)
        if self._cache_damaged_fast_frame(state, plain, sequence, pkt_cycle):
            return False, self._assemble_fast_chunks(state, stream_id)

        self._record_idle_pair(state, pkt_cycle, sequence, dv_payload[9:12])
        state.active = False
        events = self._assemble_fast_chunks(state, stream_id)
        events.extend(self._feed_slow_data(state, sequence, dv_payload[9:12]))
        return False, events

    def _feed_sync_frame(
        self, state: _TransportState, stream_id: int, cycle: int, plain: bytes,
    ) -> list[RsMs1Event]:
        """Reset slow-data pairing and retain a possible Fast Data extension."""
        state.slow_remaining = 0
        state.slow_even_sequence = None
        state.slow_header = None
        if plain[4:9] == b"\x02\x00\x00\x00\x00" and plain[9:12] == bytes.fromhex("25 62 85"):
            state.pending_seq00[cycle] = bytes(plain[:4])
            # Sync also occurs in voice. Cache its extension but require
            # actual A/B framing before claiming a Fast Data stream.
            return self._assemble_fast_chunks(state, stream_id)
        return []

    def _cache_fast_frame(
        self, state: _TransportState, plain: bytes, sequence: int, cycle: int,
    ) -> bool:
        """Cache strict A/B framing and update whether it can claim voice."""
        is_fast_a, is_fast_b = self._route_fast_data_frame(state, plain, sequence, cycle)
        if is_fast_a:
            state.marker_candidates.discard((cycle, sequence))
            state.pending_a[(cycle, sequence)] = plain
            state.active = True
        elif is_fast_b:
            state.marker_candidates.discard((cycle, sequence))
            state.pending_b[(cycle, sequence)] = plain
            expected_a = (cycle - 1, 20) if sequence == 1 else (cycle, sequence - 1)
            state.active = state.active or expected_a in state.pending_a
        return is_fast_a or is_fast_b

    def _cache_damaged_fast_frame(
        self, state: _TransportState, plain: bytes, sequence: int, cycle: int,
    ) -> bool:
        """Cache only bounded picture-marker candidates; never establish activity."""
        candidate_a, candidate_b = self._damaged_fast_role(state, plain, sequence, cycle)
        if candidate_a or candidate_b:
            pending = state.pending_a if candidate_a else state.pending_b
            pending[(cycle, sequence)] = plain
            state.marker_candidates.add((cycle, sequence))
            return True
        return False

    @staticmethod
    def _record_idle_pair(state: _TransportState, cycle: int, sequence: int, slow: bytes) -> None:
        """Track confirmed idle pairs so loss reports can exclude them."""
        if slow == _SLOW_IDLE:
            state.idle_slots.add((cycle, sequence))
            odd = sequence if sequence & 1 else sequence - 1
            if (cycle, odd) in state.idle_slots and (cycle, odd + 1) in state.idle_slots:
                state.idle_chunks.add(cycle * 10 + (odd - 1) // 2)

    def _feed_slow_data(self, state: _TransportState, sequence: int, payload: bytes) -> list[RsMs1Event]:
        """Decode ordinary text and serial data carried alongside voice."""
        slow = bytes(value ^ mask for value, mask in zip(payload, _SCRAMBLER[:3]))
        first_half_slot = bool(sequence & 1)
        if not first_half_slot and state.slow_remaining:
            count = min(3, state.slow_remaining)
            expected = state.slow_even_sequence
            header = state.slow_header
            state.slow_remaining = 0
            state.slow_even_sequence = None
            state.slow_header = None
            if expected is not None and sequence == (expected + 1) % _SEQUENCE_SLOTS:
                if header is not None and 0x40 <= header <= 0x43:
                    return self._feed_text_chunk(state, header, slow[:count])
                return self._feed_bytes(state, slow[:count])
        if first_half_slot:
            if (slow[0] & 0xF0) == 0x30:
                length = slow[0] & 0x0F
                if 1 <= length <= 5:
                    count = min(2, length)
                    state.slow_remaining = max(0, length - count)
                    state.slow_even_sequence = sequence if state.slow_remaining else None
                    state.slow_header = slow[0]
                    return self._feed_bytes(state, slow[1 : 1 + count])
            elif 0x40 <= slow[0] <= 0x43:
                state.slow_remaining = 3
                state.slow_even_sequence = sequence
                state.slow_header = slow[0]
                return self._feed_text_chunk(state, slow[0], slow[1:3])
        return []

    def _feed_text_chunk(self, state: _TransportState, header: int, data: bytes) -> list[RsMs1Event]:
        if len(data) == 2:
            state.text_chunks[header] = bytearray(data)
        elif len(data) == 3 and header in state.text_chunks:
            state.text_chunks[header].extend(data)

        if len(state.text_chunks) == 4 and all(len(c) == 5 for c in state.text_chunks.values()):
            full_text = b"".join(bytes(state.text_chunks[k]) for k in range(0x40, 0x44))
            text = full_text.decode("ascii", errors="replace").strip()
            state.text_chunks.clear()
            if text and text != state.last_text_emitted:
                state.last_text_emitted = text
                self._logger.log_event("app_text", text)
                return [RsMs1Event("app_text", text, {"text": text})]
        return []

    def _feed_bytes(self, state: _TransportState, data: bytes) -> list[RsMs1Event]:
        state.buffer.extend(data)
        events: list[RsMs1Event] = []
        while state.buffer:
            start = self._find_frame_marker(state.buffer)
            if start < 0:
                if len(state.buffer) > 6:
                    del state.buffer[:-6]
                break
            del state.buffer[:start]

            end = self._find_frame_end(state.buffer)
            if end < 0:
                if len(state.buffer) > 8192:
                    del state.buffer[:]
                    events.append(RsMs1Event("app_warning", "Discarded an oversized frame"))
                break

            next_start = self._find_frame_marker(state.buffer, offset=1)
            if 0 < next_start < end:
                del state.buffer[:next_start]
                continue

            frame = bytes(state.buffer[:end])
            del state.buffer[: end + 1]
            if state.buffer.startswith(b"\n"):
                del state.buffer[:1]

            event = self._parse_frame(frame, require_text_checksum=state.strict_text_checksum)
            if event is not None:
                if event.kind == "app_picture":
                    state.picture_active = True
                events.append(event)
        return events

    @staticmethod
    def _find_frame_marker(buffer: bytearray, *, offset: int = 0) -> int:
        """Find the earliest supported frame marker, including during recovery."""
        starts = (index for marker in _FRAME_MARKERS
                  if (index := buffer.find(marker, offset)) >= 0)
        return min(starts, default=-1)

    @staticmethod
    def _find_frame_end(buffer: bytearray) -> int:
        """Respect binary RS-MS1 payloads while accepting textual GPS newlines."""
        end = buffer.find(b"\r")
        # LF can be image data or a checksum in RS-MS1. Only GPS formats also
        # permit it as a terminator.
        if not buffer.startswith((b"$$Msg,", b"$$Pic,")):
            newline = buffer.find(b"\n")
            if newline >= 0 and (end < 0 or newline < end):
                return newline
        return end

    def _parse_frame(self, frame: bytes, *, require_text_checksum: bool = False) -> RsMs1Event | None:
        try:
            if frame.startswith(b"$$Msg,"):
                return self._parse_text(frame, require_checksum=require_text_checksum)
            if frame.startswith(b"$$Pic,"):
                return self._parse_picture(frame)
            if frame.startswith(b"$GPRMC,") or frame.startswith(b"$GPGGA,"):
                return self._parse_nmea(frame)
            if frame.startswith(b"$$CRC"):
                return self._parse_gpsa(frame)
        except (OSError, UnicodeError, ValueError) as exc:
            self._logger.log_event("app_warning", f"Rejected an RS-MS1 frame: {exc}")
            return RsMs1Event("app_warning", f"Rejected an RS-MS1 frame: {exc}")
        return None

    @staticmethod
    def _nmea_to_decimal(val: str, hem: str = "") -> str:
        import re
        s = (val + hem).strip().upper()
        match = re.fullmatch(r"(\d+)(\d\d\.\d+)\s*([NSEW])", s)
        if not match:
            return s
        deg_str, min_str, dir_char = match.groups()
        try:
            if float(min_str) >= 60:
                return ""
            degrees = float(deg_str) + float(min_str) / 60.0
            if degrees > (90 if dir_char in "NS" else 180):
                return ""
            if dir_char in ("S", "W"):
                degrees = -degrees
            return f"{degrees:.4f}"
        except ValueError:
            return s

    def _parse_nmea(self, frame: bytes) -> RsMs1Event | None:
        text = frame.decode("ascii", errors="ignore").strip()
        parts = text.split(",")
        required_fields = 7 if text.startswith("$GPRMC") else 6
        if len(parts) < required_fields:
            raise ValueError("truncated NMEA coordinates")

        lat_str = ""
        lon_str = ""

        if text.startswith("$GPRMC"):
            if parts[2] != "A":
                return None
            lat_str = self._nmea_to_decimal(parts[3], parts[4])
            lon_str = self._nmea_to_decimal(parts[5], parts[6])
        elif text.startswith("$GPGGA"):
            if len(parts) > 6 and parts[6] == "0":
                return None
            lat_str = self._nmea_to_decimal(parts[2], parts[3])
            lon_str = self._nmea_to_decimal(parts[4], parts[5])

        if lat_str and lon_str:
            coord = f"{lat_str} {lon_str}"
            self._logger.log_event("app_gps", coord)
            return RsMs1Event("app_gps", coord, {"coordinates": coord})
        return None

    def _parse_gpsa(self, frame: bytes) -> RsMs1Event | None:
        import re
        text = frame.decode("ascii", errors="ignore").strip()
        match = re.search(r"([0-9]{4,5}\.[0-9]+[NS])(.)([0-9]{4,5}\.[0-9]+[EW])", text)
        if match:
            lat = self._nmea_to_decimal(match.group(1))
            lon = self._nmea_to_decimal(match.group(3))
            coord = f"{lat} {lon}"
            self._logger.log_event("app_gps", coord)
            return RsMs1Event("app_gps", coord, {"coordinates": coord})
        return None

    @staticmethod
    def _split_address(frame: bytes, prefix_length: int, control_length: int) -> tuple[str, str, str, bytes]:
        body = frame[prefix_length:]
        # Recognize standard comma (0x2C) or high-bit RF flipped comma (0xAC)
        first = -1
        for i, b in enumerate(body):
            if b in (0x2C, 0xAC):
                first = i
                break
        second = -1
        if first >= 0:
            for i in range(first + 1, len(body)):
                if body[i] in (0x2C, 0xAC):
                    second = i
                    break
        if first < 0 or second < 0 or len(body) < second + 1 + control_length:
            raise ValueError("bad address header")
        source = body[:first].decode("ascii", errors="replace")
        destination = body[first + 1 : second].decode("ascii", errors="replace")
        control = body[second + 1 : second + 1 + control_length].decode("ascii", errors="replace")
        payload = body[second + 1 + control_length :]
        return source, destination, control, payload

    def _parse_text(self, frame: bytes, *, require_checksum: bool = False) -> RsMs1Event:
        source, destination, control, encoded = self._split_address(frame, 6, 6)
        address_ok = (control.upper() == text_control(source, destination))
        payload, picture_number = self._split_picture_binding(unescape_bytes(encoded))
        payload_ok = bool(payload) and (sum(payload[:-1]) & 0xFF) == payload[-1]
        checksum_ok = address_ok and payload_ok
        if require_checksum and not checksum_ok:
            raise ValueError("bad text checksum after Fast Data marker damage")
        text_bytes = payload[:-1] if payload_ok else payload
        text = text_bytes.decode("utf-8", errors="replace").strip()
        if not text and not payload:
            raise ValueError("empty text payload")
        if picture_number is not None and checksum_ok:
            self._captions[(source, destination, picture_number)] = text
        err_tag = "" if checksum_ok else " [CHECKSUM ERROR]"
        self._logger.log_event("app_text", f"{source or 'Unknown'}: {text}{err_tag}", pic_num=picture_number)
        return RsMs1Event(
            "app_text",
            f"{source or 'Unknown'}: {text}",
            {
                "source": source,
                "destination": destination,
                "text": text,
                "picture_number": picture_number,
                "checksum_ok": checksum_ok,
            },
        )

    @staticmethod
    def _split_picture_binding(payload: bytes) -> tuple[bytes, int | None]:
        """Separate a validated caption suffix from the checksummed text body."""
        if (
            len(payload) >= 5
            and payload[-4] == ord(",")
            and all(chr(value) in "0123456789ABCDEFabcdef" for value in payload[-3:-1])
            and (sum(payload[-3:-1]) & 0xFF) == payload[-1]
        ):
            return payload[:-4], int(payload[-3:-1].decode("ascii"), 16)
        return payload, None

    def _parse_picture(self, frame: bytes) -> RsMs1Event:
        """Parse a received picture frame, maintaining assembly state over multiple frames."""
        source, destination, control, encoded = self._split_address(frame, 6, 4)
        if len(control) != 4 or any(ch not in "0123456789ABCDEFabcdef" for ch in control):
            raise ValueError("bad picture control")
        number = int(control[:2], 16)
        payload = unescape_bytes(encoded)
        if len(payload) < 5 or (sum(payload[:-1]) & 0xFF) != payload[-1]:
            raise ValueError("bad picture tile checksum")
        record, x, y = payload[:3]
        preset = preset_for_record(record)
        if preset is None:
            raise ValueError(f"unsupported picture record 0x{record:02X}")
        if x >= preset.tiles_x or y >= preset.tiles_y:
            raise ValueError(f"picture tile coordinate ({x},{y}) outside {preset.label}")

        current, source, destination, picture_id = self._match_picture_state(
            source, destination, number, record, control
        )

        tile = _decode_tile(payload[3:-1], preset.quality)
        current = self._update_picture_tile(current, source, destination, number, preset, picture_id, tile, x, y)

        return self._format_picture_event(current, source, destination, number, preset, picture_id, x, y)

    def _match_picture_state(
        self, source: str, destination: str, number: int, record: int, control: str
    ) -> tuple[_PictureState | None, str, str, str]:
        """Resolve the picture state and address block by matching metadata."""
        picture_id = f"{source}|{destination}|{number:02X}|{record:02X}"
        if _picture_address_matches(source, destination, control):
            return self._pictures.get(picture_id), source, destination, picture_id

        # Recover a damaged address only when the checksum and picture metadata
        # identify one known picture. Picture numbers are local to each sender.
        candidates = [
            (pid, picture)
            for pid, picture in self._pictures.items()
            if picture.number == number
            and picture.preset.record == record
            and _picture_address_matches(picture.source, picture.destination, control)
        ]
        if len(candidates) == 1:
            picture_id, current = candidates[0]
            return current, current.source, current.destination, picture_id
        raise ValueError("bad or ambiguous picture address checksum")

    def _update_picture_tile(
        self, current: _PictureState | None, source: str, destination: str, number: int,
        preset: PicturePreset, picture_id: str, tile: Image.Image, x: int, y: int
    ) -> _PictureState:
        """Paste a decoded tile into the image state tracking buffer."""
        if current is None or ((x, y) == (0, 0) and len(current.received) == preset.tile_count):
            current = _PictureState(
                source,
                destination,
                number,
                preset,
                Image.new("RGB", (preset.width, preset.height), "white"),
            )
            self._pictures[picture_id] = current
            if len(self._pictures) > 8:
                oldest = next(iter(self._pictures))
                if oldest != picture_id:
                    self._pictures.pop(oldest, None)
        coordinate = (x, y)
        if coordinate not in current.received:
            current.image.paste(tile, (x * 16, y * 16))
            current.received.add(coordinate)
        return current

    def _format_picture_event(
        self, current: _PictureState, source: str, destination: str, number: int,
        preset: PicturePreset, picture_id: str, x: int, y: int
    ) -> RsMs1Event:
        """Package the assembled picture stream metadata into an event."""
        received = len(current.received)
        caption = self._captions.get((source, destination, number), "")
        complete = received == preset.tile_count

        display_image = current.image.copy()
        display_height = preset.height
        if preset.label == "160x120" and display_image.height == 128:
            display_image = display_image.crop((0, 0, 160, 120))
            display_height = 120

        self._logger.log_event(
            "app_picture",
            f"Pic {number} tile ({x},{y}) {received}/{preset.tile_count} from {source}",
            complete=complete,
        )
        return RsMs1Event(
            "app_picture",
            f"Picture {number} from {source}: {received}/{preset.tile_count} tiles",
            {
                "picture_id": picture_id,
                "source": source,
                "destination": destination,
                "number": number,
                "size": preset.label,
                "width": preset.width,
                "height": display_height,
                "quality": preset.quality,
                "received": received,
                "tile_x": x,
                "tile_y": y,
                "total": preset.tile_count,
                "complete": complete,
                "caption": caption,
                "image": display_image,
            },
        )


__all__ = [
    "CROP_OPTIONS",
    "FAST_FILL_DV_PAYLOAD",
    "FAST_SYNC_DV_PAYLOAD",
    "PICTURE_QUALITY_OPTIONS",
    "PICTURE_SIZE_OPTIONS",
    "PicturePreset",
    "PreparedPicture",
    "RsMs1Event",
    "RsMs1Receiver",
    "build_picture_tile_frame",
    "build_text_frame",
    "is_fast_data_payload",
    "iter_fast_data_frames",
    "normalize_payload_callsign",
    "picture_preset",
    "prepare_picture",
    "resize_picture",
]
