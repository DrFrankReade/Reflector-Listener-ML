"""IC-9700 CI-V DV receive data, usable while the radio remains in RF mode.

Icom IC-9700 CI-V Reference Guide, command 22 01 01 (DV RX data).
Each notification contains up to 30 application bytes. FA..FF are escaped
as FF 0A..FF 0F at this layer, independently of RS-MS1's EF escaping.
"""
from __future__ import annotations


def unescape_civ_data(data: bytes) -> bytes:
    result = bytearray()
    index = 0
    while index < len(data):
        value = data[index]
        index += 1
        if value == 0xFF:
            if index == len(data) or not 0x0A <= data[index] <= 0x0F:
                raise ValueError("invalid CI-V data escape")
            value = 0xF0 | data[index]
            index += 1
        elif value >= 0xFA:
            raise ValueError("unescaped CI-V reserved byte")
        result.append(value)
    return bytes(result)


class CivDataReceiver:
    """Incrementally unwrap radio-to-controller DV RX notifications.

    Do not treat a CI-V acknowledgement, echoed request, or unrelated radio's
    bytes as picture data. Preserve RS-MS1 framing across serial chunks and
    CI-V message boundaries. Capture starts may contain a partial message.
    """
    def __init__(self, radio_address: int = 0xA2, controller_address: int = 0xE0):
        self.radio_address = radio_address
        self.controller_address = controller_address
        self.buffer = bytearray()
        self.frames = 0
        self.dv_frames = 0
        self.ignored_frames = 0
        self.discarded_bytes = 0
        self.errors: list[str] = []
        self.frequency_hz: int | None = None

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        chunks = []
        while self.buffer:
            start = self.buffer.find(b"\xfe\xfe")
            if start < 0:
                keep = 1 if self.buffer[-1:] == b"\xfe" else 0
                self.discarded_bytes += len(self.buffer) - keep
                self.buffer[:] = self.buffer[-1:] if keep else b""
                break
            self.discarded_bytes += start
            del self.buffer[:start]
            end = self.buffer.find(b"\xfd", 2)
            restart = self.buffer.find(b"\xfe\xfe", 2)
            if restart >= 0 and (end < 0 or restart < end):
                self.discarded_bytes += restart
                self.errors.append("unterminated CI-V frame before next preamble")
                del self.buffer[:restart]
                continue
            if end < 0:
                if len(self.buffer) > 4096:
                    self.discarded_bytes += len(self.buffer)
                    self.errors.append("oversized CI-V frame")
                    self.buffer.clear()
                break
            frame = bytes(self.buffer[2:end])
            del self.buffer[:end + 1]
            self.frames += 1
            frequency_prefix = bytes((self.controller_address, self.radio_address, 0x03))
            if frame.startswith(frequency_prefix) and len(frame) == 8:
                bcd = frame[3:]
                if all((value & 15) <= 9 and value >> 4 <= 9 for value in bcd):
                    self.frequency_hz = sum(((value >> 4) * 10 + (value & 15)) * 100**index
                                            for index, value in enumerate(bcd))
            prefix = bytes((self.controller_address, self.radio_address, 0x22, 0x01, 0x01))
            if not frame.startswith(prefix):
                self.ignored_frames += 1
                continue
            try:
                payload = unescape_civ_data(frame[5:])
                if not 1 <= len(payload) <= 30:
                    raise ValueError("CI-V DV RX payload must contain 1..30 bytes")
            except ValueError as exc:
                self.errors.append(str(exc))
                # A delimiter prevents application records spanning this loss
                # from silently joining and possibly passing a weak checksum.
                chunks.append(b"\r")
                continue
            self.dv_frames += 1
            chunks.append(payload)
        return chunks

    def summary(self):
        return {"frames": self.frames, "dv_frames": self.dv_frames,
                "ignored_frames": self.ignored_frames, "discarded_bytes": self.discarded_bytes,
                "pending_bytes": len(self.buffer), "errors": self.errors}
