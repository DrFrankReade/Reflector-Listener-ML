"""D-Star reflector wire protocols used by the full-duplex client.

The module deliberately has no socket, GUI, audio, or third-party dependencies.
It turns user-facing reflector names into exact protocol identities, builds
control and transmit-media datagrams, and parses incoming media into a common
representation for the session layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping


DSTAR_RF_HEADER_SIZE = 41
DSTAR_DV_PAYLOAD_SIZE = 12
AMBE_FRAME_SIZE = 9
SEQUENCE_MODULUS = 21
NULL_AMBE_FRAME = bytes.fromhex("9e 8d 32 88 26 1a 3f 61 e8")
END_PATTERN = bytes.fromhex("55 55 55 55 c8 7a")
SLOW_DATA_SYNC = bytes.fromhex("55 2d 16")
SLOW_DATA_IDLE = bytes.fromhex("16 29 f5")

DPLUS_PORT = 20_001
DEXTRA_PORT = 30_001
DCS_PORT = 30_051

_TARGET_RE = re.compile(r"^(REF|XRF|DCS|XLX)[A-Z0-9]{3}$")
_ASCII_PRINTABLE_RE = re.compile(r"^[ -~]*$")


class Protocol(str, Enum):
    """Reflector protocol carried over UDP."""

    DPLUS = "DPlus"
    DEXTRA = "DExtra"
    DCS = "DCS"


class ReflectorFamily(str, Enum):
    REF = "REF"
    XRF = "XRF"
    DCS = "DCS"
    XLX = "XLX"


class ControlKind(str, Enum):
    LINK = "link"
    UNLINK = "unlink"
    LOGIN = "login"
    ACK = "ack"
    NAK = "nak"
    POLL = "poll"


class MediaKind(str, Enum):
    HEADER = "header"
    VOICE = "voice"
    END = "end"


_FAMILY_PROTOCOL: Mapping[ReflectorFamily, Protocol] = {
    ReflectorFamily.REF: Protocol.DPLUS,
    ReflectorFamily.XRF: Protocol.DEXTRA,
    ReflectorFamily.DCS: Protocol.DCS,
    ReflectorFamily.XLX: Protocol.DCS,
}

_PROTOCOL_PORT: Mapping[Protocol, int] = {
    Protocol.DPLUS: DPLUS_PORT,
    Protocol.DEXTRA: DEXTRA_PORT,
    Protocol.DCS: DCS_PORT,
}


@dataclass(frozen=True, slots=True)
class ReflectorTarget:
    """A validated logical reflector and module selection."""

    name: str
    module: str
    family: ReflectorFamily

    def __post_init__(self) -> None:
        canonical_name = self.name.strip().upper()
        canonical_module = self.module.strip().upper()
        if not _TARGET_RE.fullmatch(canonical_name):
            raise ValueError(
                "reflector name must be six characters beginning REF, XRF, DCS, or XLX"
            )
        expected_family = ReflectorFamily(canonical_name[:3])
        if self.family is not expected_family:
            raise ValueError("reflector family does not match its name")
        if len(canonical_module) != 1 or not ("A" <= canonical_module <= "Z"):
            raise ValueError("reflector module must be one letter A through Z")
        object.__setattr__(self, "name", canonical_name)
        object.__setattr__(self, "module", canonical_module)

    @property
    def protocol(self) -> Protocol:
        return _FAMILY_PROTOCOL[self.family]

    @property
    def port(self) -> int:
        return _PROTOCOL_PORT[self.protocol]

    @property
    def wire_name(self) -> str:
        # XLX reflectors use their corresponding DCS identity on a DCS link.
        if self.family is ReflectorFamily.XLX:
            return "DCS" + self.name[3:]
        return self.name

    @property
    def logical_callsign(self) -> bytes:
        return f"{self.name} {self.module}".encode("ascii")

    @property
    def wire_callsign(self) -> bytes:
        return f"{self.wire_name} {self.module}".encode("ascii")

    def __str__(self) -> str:
        return f"{self.name} {self.module}"


@dataclass(frozen=True, slots=True)
class ControlPacket:
    """A parsed reflector connection or keepalive datagram."""

    protocol: Protocol
    kind: ControlKind
    raw: bytes
    local_callsign: str | None = None
    local_module: str | None = None
    reflector: str | None = None
    reflector_module: str | None = None
    dongle: bool | None = None
    client_description: str | None = None
    reply: str | None = None


@dataclass(frozen=True, slots=True)
class RFHeader:
    """The callsign fields and CRC state of a 41-byte D-Star radio header."""

    raw: bytes
    flags: bytes
    rpt2: str
    rpt1: str
    your_call: str
    my_call: str
    my_suffix: str
    stored_crc: int | None
    computed_crc: int
    crc_valid: bool | None

    @property
    def formatted_my_call(self) -> str:
        if self.my_suffix:
            return f"{self.my_call:<8}/{self.my_suffix}"
        return self.my_call

    @property
    def callsigns(self) -> Mapping[str, str]:
        return {
            "from": self.my_call,
            "suffix": self.my_suffix,
            "to": self.your_call,
            "rpt1": self.rpt1,
            "rpt2": self.rpt2,
        }


@dataclass(frozen=True, slots=True)
class MediaPacket:
    """A protocol-neutral D-Star header, voice frame, or stream terminator."""

    protocol: Protocol
    kind: MediaKind
    stream_id: int
    raw: bytes
    sequence: int | None = None
    dv_payload: bytes | None = None
    ambe: bytes | None = None
    slow_data: bytes | None = None
    header: RFHeader | None = None
    packet_counter: int | None = None

    @property
    def end(self) -> bool:
        return self.kind is MediaKind.END


def parse_target(value: str | ReflectorTarget, module: str | None = None) -> ReflectorTarget:
    """Parse ``REF030``, ``REF030C``, or ``REF030 C`` without truncation.

    A separate *module* defaults to C.  If both an inline and a separate module
    are supplied, they must agree; this catches stale GUI state rather than
    silently connecting somewhere else.
    """

    if isinstance(value, ReflectorTarget):
        if module is not None and module.strip().upper() != value.module:
            raise ValueError("module conflicts with ReflectorTarget")
        return value
    if not isinstance(value, str):
        raise TypeError("reflector target must be text or ReflectorTarget")

    compact = value.strip().upper().replace("-", "")
    inline_module: str | None = None
    if len(compact) == 8 and compact[6] == " ":
        inline_module = compact[7]
        name = compact[:6]
    elif len(compact) == 7 and compact[:3] in {item.value for item in ReflectorFamily}:
        inline_module = compact[6]
        name = compact[:6]
    else:
        name = compact

    explicit_module = module.strip().upper() if module is not None else None
    if inline_module is not None and explicit_module is not None:
        if inline_module != explicit_module:
            raise ValueError("inline and explicit reflector modules disagree")
    selected_module = inline_module or explicit_module or "C"
    if not _TARGET_RE.fullmatch(name):
        raise ValueError(
            "reflector name must be exactly six characters beginning REF, XRF, DCS, or XLX"
        )
    return ReflectorTarget(name, selected_module, ReflectorFamily(name[:3]))


def format_station_callsign(callsign: str | bytes, module: str = "D") -> bytes:
    """Return an eight-byte D-Star station identity with a module in byte 8."""

    text = _ascii_text(callsign, "callsign").strip().upper()
    if not text or len(text) > 8:
        raise ValueError("callsign must contain one to eight printable ASCII characters")
    selected_module = module.upper()
    if selected_module != " " and (
        len(selected_module) != 1 or not ("A" <= selected_module <= "Z")
    ):
        raise ValueError("local module must be one letter A through Z, or a space")
    if len(selected_module) != 1:
        raise ValueError("local module must be one character")

    # An already formatted call (for example ``N0CALL D``) still has its
    # module replaced by the explicit argument, keeping builder behavior clear.
    base = text[:7].rstrip()
    return f"{base:<7}{selected_module}".encode("ascii")


def build_link_packet(
    target: str | ReflectorTarget,
    callsign: str | bytes,
    *,
    local_module: str = "D",
    client_description: str = "reflectorListen",
) -> bytes:
    """Build the first link datagram for any supported reflector family.
    
    This is the first packet sent to a reflector to request a connection.
    Depending on the protocol, it encapsulates the user's callsign, the 
    requested module, and an optional client description.
    """

    selected = parse_target(target)
    if selected.protocol is Protocol.DPLUS:
        return b"\x05\x00\x18\x00\x01"

    local = format_station_callsign(callsign, local_module)
    if selected.protocol is Protocol.DEXTRA:
        packet = bytearray(b" " * 11)
        packet[0:7] = local[0:7]
        packet[8] = local[7]
        packet[9] = ord(selected.module)
        packet[10] = 0
        return bytes(packet)

    description = _ascii_text(client_description, "client description").encode("ascii")
    packet = bytearray(519)
    _write_dcs_local_identity(packet, local)
    _write_dcs_reflector_identity(packet, selected.wire_callsign)
    packet[10] = 0
    # Bytes 19 onward are an optional, NUL-terminated client description.
    packet[19 : 19 + min(len(description), 499)] = description[:499]
    return bytes(packet)


def build_unlink_packet(
    target: str | ReflectorTarget,
    callsign: str | bytes,
    *,
    local_module: str = "D",
) -> bytes:
    """Build a reflector-specific unlink datagram."""

    selected = parse_target(target)
    if selected.protocol is Protocol.DPLUS:
        return b"\x05\x00\x18\x00\x00"

    local = format_station_callsign(callsign, local_module)
    if selected.protocol is Protocol.DEXTRA:
        packet = bytearray(b" " * 11)
        packet[0:7] = local[0:7]
        packet[8] = local[7]
        packet[9] = ord(" ")
        packet[10] = 0
        return bytes(packet)

    packet = bytearray(19)
    _write_dcs_local_identity(packet, local)
    _write_dcs_reflector_identity(packet, selected.wire_callsign)
    packet[9] = ord(" ")
    packet[10] = 0
    return bytes(packet)


def build_login_packet(callsign: str | bytes) -> bytes:
    """Build the 28-byte D-Plus login sent after the link-control echo."""

    text = _ascii_text(callsign, "callsign").strip().upper()
    if not text or len(text) > 16:
        raise ValueError("D-Plus login callsign must contain one to sixteen characters")
    packet = bytearray(28)
    packet[:4] = b"\x1c\xc0\x04\x00"
    packet[4 : 4 + len(text)] = text.encode("ascii")
    packet[20:28] = b"DV019999"
    return bytes(packet)


def build_poll_packet(
    target: str | ReflectorTarget,
    callsign: str | bytes,
    *,
    local_module: str = "D",
) -> bytes:
    """Build the client keepalive appropriate for *target*."""

    selected = parse_target(target)
    if selected.protocol is Protocol.DPLUS:
        return b"\x03\x60\x00"

    if selected.protocol is Protocol.DEXTRA:
        # DExtra keepalives conventionally carry the seven-byte base without
        # the local module; the final byte is the dongle flag.
        local = format_station_callsign(callsign, " ")
        return local + b"\x00"

    local = format_station_callsign(callsign, local_module)
    return local + b"\x00" + selected.wire_callsign


def parse_control_packet(data: bytes, protocol: Protocol | str) -> ControlPacket | None:
    """Parse one protocol's control datagram, returning ``None`` if malformed."""

    if not isinstance(data, bytes):
        data = bytes(data)
    selected_protocol = _coerce_protocol(protocol)
    if selected_protocol is Protocol.DPLUS:
        return _parse_dplus_control(data)
    if selected_protocol is Protocol.DEXTRA:
        return _parse_dextra_control(data)
    return _parse_dcs_control(data)


def parse_rf_header(data: bytes, *, crc_available: bool = True) -> RFHeader:
    """Extract callsigns and CRC state from a conventional radio header."""

    if len(data) != DSTAR_RF_HEADER_SIZE:
        raise ValueError("D-Star RF header must be exactly 41 bytes")
    raw = bytes(data)
    computed = dstar_crc(raw[:39])
    stored = raw[39] | (raw[40] << 8) if crc_available else None
    return RFHeader(
        raw=raw,
        flags=raw[:3],
        rpt2=_decode_fixed_field(raw[3:11]),
        rpt1=_decode_fixed_field(raw[11:19]),
        your_call=_decode_fixed_field(raw[19:27]),
        my_call=_decode_fixed_field(raw[27:35]),
        my_suffix=_decode_fixed_field(raw[35:39]),
        stored_crc=stored,
        computed_crc=computed,
        crc_valid=(computed == stored) if stored is not None else None,
    )


def dstar_crc(data: bytes) -> int:
    """Return the reflected CCITT CRC used over the first 39 header bytes."""

    if len(data) < 39:
        raise ValueError("D-Star CRC needs at least 39 bytes")
    crc = 0xFFFF
    for byte in data[:39]:
        current = byte
        for _ in range(8):
            low_bit_differs = bool((crc ^ current) & 1)
            crc >>= 1
            if low_bit_differs:
                crc ^= 0x8408
            current >>= 1
    return (~crc) & 0xFFFF


def build_rf_header(
    target: str | ReflectorTarget,
    callsign: str | bytes,
    *,
    local_module: str = "D",
    suffix: str = "PC",
) -> bytes:
    """Build the routed 41-byte RF header used for a reflector transmission."""

    selected = parse_target(target)
    local = format_station_callsign(callsign, local_module)
    # MYCALL has all eight bytes; routing identities reserve byte eight for
    # the station module, but must not truncate the transmitted call sign.
    my_call = _ascii_text(callsign, "callsign").strip().upper().encode("ascii").ljust(8, b" ")
    if selected.protocol is Protocol.DEXTRA:
        rpt1 = selected.wire_callsign[:7] + b"G"
    else:
        rpt1 = local
    suffix_bytes = _ascii_text(suffix, "MYCALL suffix").strip().upper().encode("ascii")
    if len(suffix_bytes) > 4:
        raise ValueError("MYCALL suffix must contain at most four characters")
    first_39 = (
        b"\x00\x00\x00"
        + selected.wire_callsign
        + rpt1
        + b"CQCQCQ  "
        + my_call
        + suffix_bytes.ljust(4, b" ")
    )
    crc = dstar_crc(first_39)
    return first_39 + crc.to_bytes(2, "little")


def build_header_packet(
    target: str | ReflectorTarget, rf_header: bytes, stream_id: int
) -> bytes | None:
    """Build a D-Plus/DExtra stream header; DCS repeats it in every frame."""

    selected = parse_target(target)
    _validate_tx_common(rf_header, stream_id)
    if selected.protocol is Protocol.DCS:
        return None
    if selected.protocol is Protocol.DPLUS:
        packet = bytearray(58)
        packet[:2] = b"\x3a\x80"
        offset = 2
        stream_offset = 14
        sequence_offset = 16
        header_offset = 17
    else:
        packet = bytearray(56)
        offset = 0
        stream_offset = 12
        sequence_offset = 14
        header_offset = 15
    packet[offset : offset + 4] = b"DSVT"
    packet[offset + 4] = 0x10
    packet[offset + 8] = 0x20
    packet[offset + 9 : offset + 12] = b"\x00\x02\x01"
    packet[stream_offset : stream_offset + 2] = stream_id.to_bytes(2, "little")
    packet[sequence_offset] = 0x80
    packet[header_offset : header_offset + DSTAR_RF_HEADER_SIZE] = rf_header
    return bytes(packet)


def build_voice_packet(
    target: str | ReflectorTarget,
    rf_header: bytes,
    ambe: bytes | None,
    sequence: int,
    stream_id: int,
    *,
    end: bool = False,
    packet_counter: int = 0,
    slow_data: bytes | None = None,
) -> bytes:
    """Build one protocol-specific D-Star voice or end datagram.
    
    This function serializes a 9-byte AMBE audio frame and 3-byte slow data
    into a UDP datagram according to the selected target's protocol format.
    The sequence number correctly wraps from 0 to 20, and `end` controls 
    whether an end-of-stream terminator is generated instead of voice.
    """

    selected = parse_target(target)
    _validate_tx_common(rf_header, stream_id)
    payload = _build_voice_payload(ambe, sequence, end=end, slow_data=slow_data)
    if selected.protocol is Protocol.DPLUS:
        return _build_dplus_voice_packet(payload, sequence, stream_id, end=end)
    if selected.protocol is Protocol.DEXTRA:
        return _build_dextra_voice_packet(payload, sequence, stream_id, end=end)
    return _build_dcs_voice_packet(
        rf_header, payload, sequence, stream_id, end=end, packet_counter=packet_counter
    )


def _build_voice_payload(
    ambe: bytes | None, sequence: int, *, end: bool, slow_data: bytes | None
) -> bytes:
    """Validate media fields and choose the shared voice/sync payload."""
    if isinstance(sequence, bool) or not 0 <= sequence < SEQUENCE_MODULUS:
        raise ValueError("sequence must be between 0 and 20")
    if not end and (ambe is None or len(ambe) != AMBE_FRAME_SIZE):
        raise ValueError("voice packet requires one nine-byte AMBE frame")
    if slow_data is not None and len(slow_data) != 3:
        raise ValueError("slow data must contain exactly three bytes")
    slow = (
        SLOW_DATA_SYNC
        if sequence == 0
        else (SLOW_DATA_IDLE if slow_data is None else bytes(slow_data))
    )
    return (NULL_AMBE_FRAME if end else bytes(ambe)) + slow  # type: ignore[arg-type]


def _build_dplus_voice_packet(
    payload: bytes, sequence: int, stream_id: int, *, end: bool
) -> bytes:
    """Serialize D-Plus voice, including its longer end datagram."""
    packet = bytearray(32 if end else 29)
    packet[:2] = b"\x20\x80" if end else b"\x1d\x80"
    packet[2:6] = b"DSVT"
    packet[6] = 0x20
    packet[10] = 0x20
    packet[11:14] = b"\x00\x02\x01"
    packet[14:16] = stream_id.to_bytes(2, "little")
    packet[16] = sequence | (0x40 if end else 0)
    packet[17:29] = payload
    if end:
        packet[26:32] = END_PATTERN
    return bytes(packet)


def _build_dextra_voice_packet(
    payload: bytes, sequence: int, stream_id: int, *, end: bool
) -> bytes:
    """Serialize DExtra voice and its in-payload end marker."""
    packet = bytearray(27)
    packet[:4] = b"DSVT"
    packet[4] = 0x20
    packet[8] = 0x20
    packet[9:12] = b"\x00\x02\x01"
    packet[12:14] = stream_id.to_bytes(2, "little")
    packet[14] = sequence | (0x40 if end else 0)
    packet[15:27] = END_PATTERN.ljust(12, b"\x00") if end else payload
    return bytes(packet)


def _build_dcs_voice_packet(
    rf_header: bytes, payload: bytes, sequence: int, stream_id: int, *,
    end: bool, packet_counter: int,
) -> bytes:
    """Serialize DCS voice with its repeated routing header and counter."""
    packet = bytearray(100)
    packet[:4] = b"0001"
    packet[4:43] = rf_header[:39]
    packet[43:45] = stream_id.to_bytes(2, "little")
    packet[45] = sequence | (0x40 if end else 0)
    packet[46:58] = payload
    if end:
        packet[55:58] = b"\x55\x55\x55"
    packet[58:61] = (packet_counter & 0xFFFFFF).to_bytes(3, "little")
    packet[61:64] = b"\x01\x00\x21"
    return bytes(packet)


def _validate_tx_common(rf_header: bytes, stream_id: int) -> None:
    if len(rf_header) != DSTAR_RF_HEADER_SIZE:
        raise ValueError("D-Star RF header must be exactly 41 bytes")
    if isinstance(stream_id, bool) or not 1 <= stream_id <= 0xFFFF:
        raise ValueError("stream ID must be between 1 and 65535")


def parse_media_packet(
    data: bytes, protocol: Protocol | str | None = None
) -> MediaPacket | None:
    """Parse one D-Plus, DExtra, or DCS media datagram.

    When *protocol* is omitted the non-overlapping media signatures are used
    for detection.  Control datagrams are never returned from this function.
    """

    if not isinstance(data, bytes):
        data = bytes(data)
    if protocol is None:
        if data.startswith(b"0001"):
            selected_protocol = Protocol.DCS
        elif data.startswith(b"DSVT"):
            selected_protocol = Protocol.DEXTRA
        elif len(data) >= 6 and data[2:6] == b"DSVT":
            selected_protocol = Protocol.DPLUS
        else:
            return None
    else:
        selected_protocol = _coerce_protocol(protocol)

    if selected_protocol is Protocol.DPLUS:
        return _parse_dplus_media(data)
    if selected_protocol is Protocol.DEXTRA:
        return _parse_dextra_media(data)
    return _parse_dcs_media(data)


def header_targets_target(
    header: RFHeader | bytes, target: str | ReflectorTarget
) -> bool:
    """Return whether an RF header is routed through the selected reflector.

    XLX-over-DCS accepts both its logical ``XLXnnn`` and on-wire ``DCSnnn``
    identity.  DCS requires RPT2 to match; REF and XRF accept a match in either
    repeater route field, which covers the gateway/module forms seen in use.
    """

    parsed_header = parse_rf_header(header) if isinstance(header, bytes) else header
    selected = parse_target(target)
    identities = {selected.logical_callsign, selected.wire_callsign}
    rpt2 = parsed_header.raw[3:11]
    rpt1 = parsed_header.raw[11:19]
    if selected.protocol is Protocol.DCS:
        return rpt2 in identities
    return rpt2 in identities or rpt1 in identities


class TargetStreamFilter:
    """Track the stream IDs whose headers match one selected reflector."""

    def __init__(self, target: str | ReflectorTarget) -> None:
        self.target = parse_target(target)
        self._accepted_streams: set[int] = set()
        self._rejected_streams: set[int] = set()

    @property
    def accepted_streams(self) -> frozenset[int]:
        return frozenset(self._accepted_streams)

    def reset(self) -> None:
        self._accepted_streams.clear()
        self._rejected_streams.clear()

    def admit_stream(self, stream_id: int) -> bool:
        """Allow a late join only when no header has rejected this stream."""
        if stream_id in self._rejected_streams:
            return False
        self._accepted_streams.add(stream_id)
        return True

    def accept(self, packet: MediaPacket) -> bool:
        if packet.protocol is not self.target.protocol:
            return False

        if packet.header is not None:
            matches = header_targets_target(packet.header, self.target)
            if not matches and packet.protocol in (Protocol.DPLUS, Protocol.DEXTRA):
                rpt2 = packet.header.rpt2
                if rpt2 and not any(rpt2.startswith(prefix) for prefix in ("REF", "XRF", "DCS", "XLX")):
                    matches = True
            if not matches:
                self._accepted_streams.discard(packet.stream_id)
                if packet.end:
                    self._rejected_streams.discard(packet.stream_id)
                else:
                    self._rejected_streams.add(packet.stream_id)
                return False
            self._rejected_streams.discard(packet.stream_id)
            self._accepted_streams.add(packet.stream_id)
        elif packet.stream_id not in self._accepted_streams:
            if packet.end:
                self._rejected_streams.discard(packet.stream_id)
            return False

        accepted = packet.stream_id in self._accepted_streams
        if packet.end:
            self._accepted_streams.discard(packet.stream_id)
        return accepted


def next_sequence(sequence: int) -> int:
    """Advance a D-Star DV sequence number, wrapping 20 to zero."""

    _validate_sequence(sequence)
    return (sequence + 1) % SEQUENCE_MODULUS


def sequence_distance(start: int, end: int) -> int:
    """Return the forward number of 20 ms slots from *start* to *end*."""

    _validate_sequence(start)
    _validate_sequence(end)
    return (end - start) % SEQUENCE_MODULUS


def is_next_sequence(previous: int, candidate: int) -> bool:
    _validate_sequence(previous)
    _validate_sequence(candidate)
    return next_sequence(previous) == candidate


def _parse_dplus_control(data: bytes) -> ControlPacket | None:
    """Parse D-Plus link, unlink, poll, and ACK datagrams.
    
    D-Plus commands are highly variable in length (e.g. 13-byte link, 
    23-byte ACK, 5-byte keepalive) and usually lack explicit reflector IDs.
    """
    if data == b"\x03\x60\x00":
        return ControlPacket(Protocol.DPLUS, ControlKind.POLL, data)
    if len(data) == 5 and data[:4] == b"\x05\x00\x18\x00":
        if data[4] == 1:
            return ControlPacket(Protocol.DPLUS, ControlKind.LINK, data)
        if data[4] == 0:
            return ControlPacket(Protocol.DPLUS, ControlKind.UNLINK, data)
        return None
    if len(data) == 8 and data[:4] == b"\x08\xc0\x04\x00":
        reply = _try_ascii(data[4:8])
        if reply == "OKRW":
            return ControlPacket(
                Protocol.DPLUS, ControlKind.ACK, data, reply=reply
            )
        if reply == "BUSY":
            return ControlPacket(
                Protocol.DPLUS, ControlKind.NAK, data, reply=reply
            )
        return None
    if len(data) == 28 and data[:4] == b"\x1c\xc0\x04\x00":
        if data[20:28] != b"DV019999":
            return None
        call_bytes = data[4:20].split(b"\x00", 1)[0].rstrip(b" ")
        call = _try_ascii(call_bytes)
        if not call:
            return None
        return ControlPacket(
            Protocol.DPLUS,
            ControlKind.LOGIN,
            data,
            local_callsign=call,
            client_description="DV019999",
        )
    return None


def _parse_dextra_control(data: bytes) -> ControlPacket | None:
    """Parse DExtra connection and keepalive datagrams.
    
    DExtra commands encode the local callsign explicitly and
    use ASCII NAK strings when link attempts are rejected.
    """
    if len(data) == 9:
        call = _decode_station_or_plain(data[:8])
        if call is None:
            return None
        return ControlPacket(
            Protocol.DEXTRA,
            ControlKind.POLL,
            data,
            local_callsign=call,
            dongle=data[8] != 0,
        )

    if len(data) not in (11, 14) or data[7] != ord(" "):
        return None
    local = _read_split_station(data[:7], data[8])
    if local is None:
        return None
    base, local_module = local
    remote_module = _module_or_none(data[9])

    if len(data) == 11:
        if data[10] != 0:
            return None
        kind = ControlKind.UNLINK if data[9] == ord(" ") else ControlKind.LINK
        if kind is ControlKind.LINK and remote_module is None:
            return None
        return ControlPacket(
            Protocol.DEXTRA,
            kind,
            data,
            local_callsign=base,
            local_module=local_module,
            reflector_module=remote_module,
        )

    if data[13] != 0 or data[10:13] not in (b"ACK", b"NAK"):
        return None
    kind = ControlKind.ACK if data[10:13] == b"ACK" else ControlKind.NAK
    return ControlPacket(
        Protocol.DEXTRA,
        kind,
        data,
        local_callsign=base,
        local_module=local_module,
        reflector_module=remote_module,
        reply=data[10:13].decode("ascii"),
    )


def _parse_dcs_control(data: bytes) -> ControlPacket | None:
    """Parse DCS (and XLX over DCS) link and connection state datagrams.
    
    DCS relies on 19-byte or 519-byte connection frames (which include 
    an optional client description string) and fixed-length polling frames.
    """
    if len(data) in (519, 19):
        return _parse_dcs_connection(data)
    if len(data) == 14:
        return _parse_dcs_reply(data)
    if len(data) == 17:
        return _parse_dcs_client_poll(data)
    if len(data) == 22:
        return _parse_dcs_reflector_poll(data)
    return None


def _parse_dcs_connection(data: bytes) -> ControlPacket | None:
    """Read a 519-byte link or 19-byte unlink identity and description."""
    if data[7] != ord(" ") or data[10] != 0:
        return None
    local = _read_split_station(data[:7], data[8])
    reflector = _read_dcs_reflector(data)
    if local is None or reflector is None:
        return None
    local_call, local_module = local
    reflector_name, remote_module = reflector
    if len(data) == 19:
        if data[9] != ord(" "):
            return None
        return ControlPacket(
            Protocol.DCS,
            ControlKind.UNLINK,
            data,
            local_callsign=local_call,
            local_module=local_module,
            reflector=reflector_name,
        )
    if remote_module is None:
        return None
    description_bytes = data[19:].split(b"\x00", 1)[0]
    description = _try_ascii(description_bytes)
    if description is None:
        return None
    return ControlPacket(
        Protocol.DCS,
        ControlKind.LINK,
        data,
        local_callsign=local_call,
        local_module=local_module,
        reflector=reflector_name,
        reflector_module=remote_module,
        client_description=description,
    )


def _parse_dcs_reply(data: bytes) -> ControlPacket | None:
    """Read the station and module from a 14-byte ACK or NAK."""
    if data[7] != ord(" ") or data[13] != 0:
        return None
    local = _read_split_station(data[:7], data[8])
    if local is None or data[10:13] not in (b"ACK", b"NAK"):
        return None
    local_call, local_module = local
    kind = ControlKind.ACK if data[10:13] == b"ACK" else ControlKind.NAK
    return ControlPacket(
        Protocol.DCS,
        kind,
        data,
        local_callsign=local_call,
        local_module=local_module,
        reflector_module=_module_or_none(data[9]),
        reply=data[10:13].decode("ascii"),
    )


def _parse_dcs_client_poll(data: bytes) -> ControlPacket | None:
    """Read the 17-byte client-to-reflector keepalive layout."""
    if data[8] != 0:
        return None
    local = _decode_station_or_plain(data[:8])
    reflector = _decode_reflector_field(data[9:17])
    if local is None or reflector is None:
        return None
    reflector_name, reflector_module = reflector
    return ControlPacket(
        Protocol.DCS,
        ControlKind.POLL,
        data,
        local_callsign=local,
        reflector=reflector_name,
        reflector_module=reflector_module,
    )


def _parse_dcs_reflector_poll(data: bytes) -> ControlPacket | None:
    """Read the reversed identities in a 22-byte reflector keepalive."""
    if data[18:20] != b"\x0a\x00":
        return None
    reflector = _decode_reflector_field(data[:8])
    local = _read_split_station(data[9:16], data[17])
    if reflector is None or local is None:
        return None
    reflector_name, reflector_module = reflector
    local_call, local_module = local
    return ControlPacket(
        Protocol.DCS,
        ControlKind.POLL,
        data,
        local_callsign=local_call,
        local_module=local_module,
        reflector=reflector_name,
        reflector_module=reflector_module,
    )


def _parse_dplus_media(data: bytes) -> MediaPacket | None:
    if (
        len(data) == 58
        and data[:2] == b"\x3a\x80"
        and data[2:6] == b"DSVT"
        and data[6] == 0x10
        and data[10] == 0x20
        and data[16] == 0x80
    ):
        stream_id = data[14] | (data[15] << 8)
        if stream_id == 0:
            return None
        return MediaPacket(
            Protocol.DPLUS,
            MediaKind.HEADER,
            stream_id,
            data,
            header=parse_rf_header(data[17:58]),
        )

    if (
        len(data) in (29, 32)
        and data[0] in (0x1D, 0x20)
        and data[1] == 0x80
        and data[2:6] == b"DSVT"
        and data[6] == 0x20
        and data[10] == 0x20
    ):
        sequence = data[16] & 0x1F
        if sequence >= SEQUENCE_MODULUS:
            return None
        stream_id = data[14] | (data[15] << 8)
        if stream_id == 0:
            return None
        is_end = data[0] == 0x20 or bool(data[16] & 0x40)
        payload = data[17:29]
        return _voice_packet(
            Protocol.DPLUS, data, stream_id, sequence, payload, is_end
        )
    return None


def _parse_dextra_media(data: bytes) -> MediaPacket | None:
    if (
        len(data) == 56
        and data[:4] == b"DSVT"
        and data[4] == 0x10
        and data[8] == 0x20
        and data[14] == 0x80
    ):
        stream_id = data[12] | (data[13] << 8)
        if stream_id == 0:
            return None
        return MediaPacket(
            Protocol.DEXTRA,
            MediaKind.HEADER,
            stream_id,
            data,
            header=parse_rf_header(data[15:56]),
        )

    if (
        len(data) == 27
        and data[:4] == b"DSVT"
        and data[4] == 0x20
        and data[8] == 0x20
    ):
        sequence = data[14] & 0x1F
        if sequence >= SEQUENCE_MODULUS:
            return None
        stream_id = data[12] | (data[13] << 8)
        if stream_id == 0:
            return None
        return _voice_packet(
            Protocol.DEXTRA,
            data,
            stream_id,
            sequence,
            data[15:27],
            bool(data[14] & 0x40),
        )
    return None


def _parse_dcs_media(data: bytes) -> MediaPacket | None:
    if len(data) != 100 or data[:4] != b"0001":
        return None
    sequence = data[45] & 0x1F
    if sequence >= SEQUENCE_MODULUS:
        return None
    stream_id = data[43] | (data[44] << 8)
    if stream_id == 0:
        return None
    raw_header = data[4:43] + b"\xff\xff"
    header = parse_rf_header(raw_header, crc_available=False)
    payload = data[46:58]
    is_end = bool(data[45] & 0x40) or data[55:58] == b"\x55\x55\x55"
    packet = _voice_packet(
        Protocol.DCS, data, stream_id, sequence, payload, is_end, header=header
    )
    return MediaPacket(
        protocol=packet.protocol,
        kind=packet.kind,
        stream_id=packet.stream_id,
        raw=packet.raw,
        sequence=packet.sequence,
        dv_payload=packet.dv_payload,
        ambe=packet.ambe,
        slow_data=packet.slow_data,
        header=packet.header,
        packet_counter=data[58] | (data[59] << 8) | (data[60] << 16),
    )


def _voice_packet(
    protocol: Protocol,
    raw: bytes,
    stream_id: int,
    sequence: int,
    payload: bytes,
    is_end: bool,
    *,
    header: RFHeader | None = None,
) -> MediaPacket:
    return MediaPacket(
        protocol=protocol,
        kind=MediaKind.END if is_end else MediaKind.VOICE,
        stream_id=stream_id,
        raw=raw,
        sequence=sequence,
        dv_payload=payload,
        ambe=None if is_end else payload[:AMBE_FRAME_SIZE],
        slow_data=None if is_end else payload[AMBE_FRAME_SIZE:DSTAR_DV_PAYLOAD_SIZE],
        header=header,
    )


def _write_dcs_local_identity(packet: bytearray, local: bytes) -> None:
    packet[0:8] = b" " * 8
    packet[0:7] = local[:7]
    packet[8] = local[7]


def _write_dcs_reflector_identity(packet: bytearray, reflector: bytes) -> None:
    packet[9] = reflector[7]
    packet[11:19] = b" " * 8
    packet[11:18] = reflector[:7]


def _read_dcs_reflector(data: bytes) -> tuple[str, str | None] | None:
    field = data[11:18] + bytes((data[9],))
    return _decode_reflector_field(field)


def _decode_reflector_field(field: bytes) -> tuple[str, str | None] | None:
    if len(field) != 8:
        return None
    name = _try_ascii(field[:7])
    if name is None:
        return None
    name = name.rstrip()
    if not _TARGET_RE.fullmatch(name):
        return None
    module = _module_or_none(field[7])
    if field[7] != ord(" ") and module is None:
        return None
    return name, module


def _read_split_station(base_field: bytes, module_byte: int) -> tuple[str, str] | None:
    base = _try_ascii(base_field)
    module = _module_or_none(module_byte)
    if base is None or module is None:
        return None
    base = base.rstrip()
    if not base:
        return None
    return base, module


def _decode_station_or_plain(field: bytes) -> str | None:
    text = _try_ascii(field)
    if text is None:
        return None
    text = text.rstrip()
    return text or None


def _decode_fixed_field(field: bytes) -> str:
    return "".join(chr(byte) if 0x20 <= byte <= 0x7E else "?" for byte in field).rstrip()


def _module_or_none(value: int) -> str | None:
    if ord("A") <= value <= ord("Z"):
        return chr(value)
    return None


def _try_ascii(data: bytes) -> str | None:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return None
    return text if _ASCII_PRINTABLE_RE.fullmatch(text) else None


def _ascii_text(value: str | bytes, label: str) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} must be ASCII") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError(f"{label} must be text or bytes")
    if not _ASCII_PRINTABLE_RE.fullmatch(text):
        raise ValueError(f"{label} must contain printable ASCII only")
    return text


def _coerce_protocol(value: Protocol | str) -> Protocol:
    if isinstance(value, Protocol):
        return value
    normalized = value.strip().lower()
    aliases = {
        "dplus": Protocol.DPLUS,
        "d-plus": Protocol.DPLUS,
        "ref": Protocol.DPLUS,
        "dextra": Protocol.DEXTRA,
        "d-extra": Protocol.DEXTRA,
        "xrf": Protocol.DEXTRA,
        "dcs": Protocol.DCS,
        "xlx": Protocol.DCS,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported reflector protocol: {value!r}") from exc


def _validate_sequence(sequence: int) -> None:
    if type(sequence) is not int or not 0 <= sequence < SEQUENCE_MODULUS:
        raise ValueError("D-Star sequence must be an integer from 0 through 20")


__all__ = [
    "AMBE_FRAME_SIZE",
    "ControlKind",
    "ControlPacket",
    "DCS_PORT",
    "DEXTRA_PORT",
    "DPLUS_PORT",
    "DSTAR_DV_PAYLOAD_SIZE",
    "DSTAR_RF_HEADER_SIZE",
    "MediaKind",
    "MediaPacket",
    "Protocol",
    "RFHeader",
    "ReflectorFamily",
    "ReflectorTarget",
    "TargetStreamFilter",
    "build_link_packet",
    "build_login_packet",
    "build_poll_packet",
    "build_unlink_packet",
    "dstar_crc",
    "format_station_callsign",
    "header_targets_target",
    "is_next_sequence",
    "next_sequence",
    "parse_control_packet",
    "parse_media_packet",
    "parse_rf_header",
    "parse_target",
    "sequence_distance",
]
