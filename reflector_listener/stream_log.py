"""Bounded binary UDP archive with stable, expiring per-transmission ranges.

Files: RFLSTRM1 magic, followed by <dBH> (UTC timestamp, protocol, length)
and the exact UDP payload. No decoder or UI state is stored as audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import struct
import threading
import time
import uuid
import weakref
from typing import Callable, Iterator

from .protocols import MediaKind, Protocol, parse_media_packet
from .rsms1 import is_fast_data_payload


MAGIC = b"RFLSTRM1"
RECORD = struct.Struct("<dBH")
PROTOCOLS = (Protocol.DPLUS, Protocol.DEXTRA, Protocol.DCS)


@dataclass(frozen=True)
class RecordRef:
    path: Path
    offset: int
    end: int


@dataclass
class Recording:
    stream_id: int
    protocol: Protocol
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    ranges: dict[Path, tuple[int, int]] = field(default_factory=dict)
    complete: bool = False
    expired: bool = False
    frames: int = 0
    discarded: bool = False


@dataclass(frozen=True)
class RecordingGroup:
    """One message's received segments, without copying or widening file ranges."""

    segments: tuple[Recording, ...]
    token: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.segments or any(
            (clip.stream_id, clip.protocol) != (self.stream_id, self.protocol)
            for clip in self.segments
        ):
            raise ValueError("Recording segments must belong to the same stream and protocol.")

    @property
    def stream_id(self) -> int:
        return self.segments[0].stream_id

    @property
    def protocol(self) -> Protocol:
        return self.segments[0].protocol

    @property
    def frames(self) -> int:
        return sum(clip.frames for clip in self.segments)

    @property
    def complete(self) -> bool:
        return all(clip.complete for clip in self.segments)

    @property
    def expired(self) -> bool:
        return any(clip.expired for clip in self.segments)


class BinaryStreamLog:
    """Own only stream_*.rfl files in a dedicated archive directory.

    A lock protects rotation and small reads; playback never touches the live
    session decoder or waits for its lifecycle lock. Old recordings expire as
    whole UI clips when any of their underlying file segments are removed.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).resolve()
        self._lock = threading.RLock()
        self._file = None
        self._path: Path | None = None
        self._sizes: dict[Path, int] = {}
        self._clips: weakref.WeakValueDictionary[str, Recording] = weakref.WeakValueDictionary()
        # append() precedes stream identification/include() on the receiver.
        # Keep files safe during that handoff, including superseding headers.
        self._unclaimed: weakref.WeakSet[RecordRef] = weakref.WeakSet()
        self._pending_delete: set[Path] = set()
        self._cleanup_error = ""
        self._next_cleanup = 0.0
        self._rotate_on_append = False
        self.enabled = False
        self.max_bytes = 100_000_000
        self.error = ""

    def configure(self, enabled: bool, max_bytes: int) -> None:
        if max_bytes < 1024:
            raise ValueError("Stream log budget must be at least 1024 bytes.")
        with self._lock:
            self.close()
            self.max_bytes = max_bytes
            self.error = ""
            if enabled:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._sizes = {p: p.stat().st_size for p in sorted(self.directory.glob("stream_*.rfl")) if p.is_file() and not p.is_symlink()}
                self._trim(0)
                self._rotate()
                self.enabled = True

    def _rotate(self) -> None:
        self._rotate_on_append = False
        if self._file is not None:
            self._file.close()
        self._file = None
        self._path = None
        self._collect_discarded()
        self._path = self.directory / f"stream_{time.time_ns()}_{uuid.uuid4().hex[:8]}.rfl"
        self._trim(len(MAGIC))
        self._file = self._path.open("xb", buffering=0)
        self._file.write(MAGIC)
        self._sizes[self._path] = len(MAGIC)

    def _trim(self, incoming: int) -> None:
        for path in list(self._sizes):
            if sum(self._sizes.values()) + incoming <= self.max_bytes:
                break
            if path == self._path and self._file is not None and not self._file.closed:
                continue
            self._remove_file(path)

    def _remove_file(self, path: Path) -> None:
        # Only exact archive-owned names directly inside the archive directory.
        if path not in self._sizes or path.parent != self.directory or path.is_symlink():
            raise OSError("Unsafe archive path")
        path.unlink(missing_ok=True)
        self._sizes.pop(path)
        self._pending_delete.discard(path)
        for clip in list(self._clips.values()):
            if path in clip.ranges:
                if clip.discarded:
                    del clip.ranges[path]
                else:
                    clip.expired = True
                    clip.ranges.clear()

    def discard(self, clip: Recording | RecordingGroup) -> None:
        """Release a FIFO-evicted message; shared or active files wait safely."""
        with self._lock:
            segments = clip.segments if isinstance(clip, RecordingGroup) else (clip,)
            for segment in segments:
                if self._clips.get(segment.token) is segment:
                    segment.discarded = segment.expired = True
                    self._pending_delete.update(segment.ranges)
            # New calls should not keep extending files awaiting FIFO cleanup.
            if self._path in self._pending_delete:
                self._rotate_on_append = True
            self._collect_discarded()

    def _collect_discarded(self) -> None:
        """Called under the archive lock; never delete another clip's storage."""
        protected = {ref.path for ref in self._unclaimed}
        for clip in list(self._clips.values()):
            if not clip.expired or (clip.discarded and not clip.complete):
                protected.update(clip.ranges)
        for path in tuple(self._pending_delete - protected):
            if path not in self._sizes:
                self._pending_delete.discard(path)
                continue
            try:
                if path == self._path and self._file is not None:
                    self._file.close()
                    self._file = None
                    self._path = None
                self._remove_file(path)
            except OSError as exc:
                # A locked file stays queued for retry; live recording continues.
                if str(exc) != self._cleanup_error:
                    logging.getLogger(__name__).warning("Cannot remove discarded recording; will retry: %s", exc)
                    self._cleanup_error = str(exc)

    def append(self, data: bytes, protocol: Protocol, timestamp: float | None = None) -> RecordRef | None:
        with self._lock:
            if not self.enabled:
                return None
            try:
                if self._pending_delete and time.monotonic() >= self._next_cleanup:
                    self._collect_discarded()
                    self._next_cleanup = time.monotonic() + 1.0
                encoded = RECORD.pack(time.time() if timestamp is None else timestamp,
                                      PROTOCOLS.index(protocol), len(data)) + data
                if len(encoded) + len(MAGIC) > self.max_bytes // 2:
                    return None
                if (self._file is None or self._rotate_on_append
                        or self._sizes[self._path] + len(encoded) > min(1_000_000, self.max_bytes // 2)):
                    self._rotate()
                self._trim(len(encoded))
                offset = self._file.tell()
                self._file.write(encoded)
                self._sizes[self._path] += len(encoded)
                ref = RecordRef(self._path, offset, offset + len(encoded))
                self._unclaimed.add(ref)
                return ref
            except (OSError, ValueError, struct.error) as exc:
                self.error = str(exc)
                self.close()
                return None

    def begin(self, stream_id: int, protocol: Protocol) -> Recording | None:
        with self._lock:
            if not self.enabled:
                return None
            clip = Recording(stream_id, protocol)
            self._clips[clip.token] = clip
            return clip

    def include(self, clip: Recording | None, ref: RecordRef | None, *, voice: bool = False) -> None:
        with self._lock:
            if clip is None or ref is None or (clip.expired and not clip.discarded) or clip.complete:
                return
            start, _ = clip.ranges.get(ref.path, (ref.offset, ref.end))
            clip.ranges[ref.path] = start, ref.end
            clip.frames += int(voice)
            self._unclaimed.discard(ref)
            if clip.discarded:
                self._pending_delete.add(ref.path)

    def finish(self, clip: Recording | None) -> None:
        with self._lock:
            if clip:
                clip.complete = True
            self._collect_discarded()

    def records(self, clip: Recording) -> Iterator[tuple[RecordRef, float, bytes]]:
        with self._lock:
            if clip.expired:
                raise FileNotFoundError("This recording has expired from the rolling log.")
            ranges = list(clip.ranges.items())
        for path, (start, end) in ranges:
            with self._lock:
                with path.open("rb") as source:
                    if source.read(len(MAGIC)) != MAGIC:
                        raise ValueError("Invalid stream log header")
                    source.seek(start)
                    segment = source.read(end - start)
            offset = 0
            while offset + RECORD.size <= len(segment):
                timestamp, proto, length = RECORD.unpack_from(segment, offset)
                next_offset = offset + RECORD.size + length
                if next_offset > len(segment) or proto >= len(PROTOCOLS):
                    raise ValueError("Truncated or invalid stream record")
                data = segment[offset + RECORD.size:next_offset]
                if PROTOCOLS[proto] is clip.protocol:
                    yield RecordRef(path, start + offset, start + next_offset), timestamp, data
                offset = next_offset

    def read(self, ref: RecordRef) -> bytes:
        with self._lock:
            with ref.path.open("rb") as source:
                source.seek(ref.offset)
                header = source.read(RECORD.size)
                if len(header) != RECORD.size:
                    raise ValueError("Truncated stream record")
                _, _, length = RECORD.unpack(header)
                data = source.read(length)
                if len(data) != length:
                    raise ValueError("Truncated stream payload")
                return data

    def close(self) -> None:
        with self._lock:
            self.enabled = False
            if self._file is not None:
                self._file.close()
            self._file = None
            self._collect_discarded()

    def purge(self) -> None:
        """Clean exit: close the writer and delete only this archive's RFL files."""
        with self._lock:
            self.close()
            for path in self.directory.glob("stream_*.rfl"):
                if path.parent != self.directory or path.is_symlink() or not path.is_file():
                    continue
                try:
                    path.unlink(missing_ok=True)
                    self._sizes.pop(path, None)
                    self._pending_delete.discard(path)
                except OSError as exc:
                    logging.getLogger(__name__).warning("Cannot remove stream file on exit %s: %s", path, exc)
            for clip in list(self._clips.values()):
                clip.expired = True
                clip.ranges.clear()


def voice_index(archive: BinaryStreamLog, clip: Recording | RecordingGroup, cancelled: Callable[[], bool] = lambda: False) -> dict[int, RecordRef]:
    """Map 20 ms audio slots to raw records, retaining short reordering/gaps."""
    if isinstance(clip, RecordingGroup):
        result = {}
        offset = 0
        for segment in clip.segments:
            if cancelled():
                return {}
            index = voice_index(archive, segment, cancelled)
            result.update((offset + slot, ref) for slot, ref in index.items())
            offset += max(index, default=-1) + 1
        return result
    result = {}
    sequence = None
    last_time = 0.0
    slot = 0
    for ref, timestamp, raw in archive.records(clip):
        if cancelled():
            return {}
        packet = parse_media_packet(raw, clip.protocol)
        if packet is None or packet.stream_id != clip.stream_id or packet.kind is not MediaKind.VOICE:
            continue
        if is_fast_data_payload(packet.dv_payload):
            continue
        if sequence is not None:
            delta = (packet.sequence - sequence + 10) % 21 - 10
            if timestamp - last_time > 0.42:
                elapsed = min(1500, round((timestamp - last_time) / 0.02))
                delta += max(0, round((elapsed - delta) / 21)) * 21
            target = slot + delta
        else:
            target = 0
        if target >= 0:
            result.setdefault(target, ref)
        if sequence is None or target > slot:
            sequence, last_time, slot = packet.sequence, timestamp, target
    return result
