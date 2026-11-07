"""Forensic packet and RS-MS1 protocol logger for diagnosing network and RF issues."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import TextIO

from reflector_listener.rsms1 import _xor_scrambler


class PacketLogger:
    """Thread-safe parsed packet log and separate, lossless receive log."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._file: TextIO | None = None
        self._path: Path | None = None
        self._raw_file: TextIO | None = None
        self._raw_path: Path | None = None
        self._raw_packet_count = 0
        self._started_at_perf: float | None = None
        self.max_bytes = 100_000_000
        self._directory: Path | None = None
        self._pair_bytes = 0
        self._sizes: dict[Path, int] = {}

    def set_max_bytes(self, maximum: int) -> None:
        if maximum < 1024:
            raise ValueError("Debug log budget must be at least 1024 bytes")
        with self._lock:
            self.max_bytes = maximum
            if self._file is not None and self._pair_bytes > maximum // 2:
                self.start(self._directory)
            self._prune_locked()

    def _prune_locked(self) -> None:
        for path in list(self._sizes):
            if sum(self._sizes.values()) <= self.max_bytes:
                break
            if path in (self._path, self._raw_path):
                continue
            if path.parent != self._directory or path.is_symlink():
                continue
            path.unlink(missing_ok=True)
            self._sizes.pop(path)

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._file is not None

    @property
    def current_path(self) -> Path | None:
        with self._lock:
            return self._path

    @property
    def current_raw_path(self) -> Path | None:
        with self._lock:
            return self._raw_path

    def start(self, directory: Path | str = "logs") -> Path:
        """Enable a matched pair of parsed and raw logs; return the parsed path."""
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        with self._lock, ExitStack() as opened:
            # Windows clocks can repeat even with microseconds in the name.
            suffix = ""
            collision = 0
            while True:
                target_path = dir_path / f"debug_{timestamp_str}{suffix}.log"
                raw_path = dir_path / f"raw_rx_{timestamp_str}{suffix}.log"
                if not target_path.exists() and not raw_path.exists():
                    break
                collision += 1
                suffix = f"_{collision}"
            header_line = (
                f"# D-Star Debug Log started at {datetime.now().isoformat()}\n"
                f"# Raw UDP receive log: {raw_path.name}\n"
                f"# Columns: timestamp [DIR] proto stream seq [cnt] kind payload plain | details\n"
            )
            new_file = opened.enter_context(target_path.open("x", encoding="utf-8", buffering=1, newline=""))
            new_raw_file = opened.enter_context(raw_path.open("x", encoding="utf-8", buffering=1, newline=""))
            new_file.write(header_line)
            new_file.flush()
            new_raw_file.write(
                f"# D-Star Raw UDP Receive Log started at {datetime.now().astimezone().isoformat()}\n"
                f"# Parsed packet log: {target_path.name}\n"
                "# One line per datagram, before parsing/filtering; data is the complete UDP payload in hex.\n"
                "# Includes control, unknown, malformed, and empty datagrams; no IP/UDP headers.\n"
            )
            new_raw_file.flush()
            # Sort matched pairs by timestamp, rather than grouping every
            # debug file ahead of every raw file by its filename prefix.
            paths = sorted(
                (p for p in dir_path.glob("*.log")
                 if (p.name.startswith("debug_") or p.name.startswith("raw_rx_"))
                 and p.is_file() and not p.is_symlink()),
                key=lambda p: p.name.removeprefix("debug_").removeprefix("raw_rx_"),
            )
            sizes = {p: p.stat().st_size for p in paths}
            self._close_locked()
            self._file = new_file
            self._path = target_path
            self._raw_file = new_raw_file
            self._raw_path = raw_path
            self._started_at_perf = time.perf_counter()
            self._directory = dir_path
            self._sizes = sizes
            self._pair_bytes = self._sizes[target_path] + self._sizes[raw_path]
            opened.pop_all()
            try:
                self._prune_locked()
            except OSError:
                self._close_locked()
                raise

        return target_path

    def stop(self) -> Path | None:
        """Flush, close, and disable both logs; return the parsed log path."""
        with self._lock:
            saved_path = self._path
            if self._file is not None:
                self._write_locked(f"# D-Star Debug Log closed at {datetime.now().isoformat()}\n")
            if self._raw_file is not None:
                self._write_locked(
                    f"# D-Star Raw UDP Receive Log closed at {datetime.now().astimezone().isoformat()}\n",
                    raw=True,
                )
            self._close_locked()
            return saved_path

    def _close_locked(self) -> None:
        """Clear state even when a broken file cannot be flushed or closed."""
        outputs = (self._file, self._raw_file)
        self._file = None
        self._path = None
        self._raw_file = None
        self._raw_path = None
        self._raw_packet_count = 0
        self._started_at_perf = None
        for output in outputs:
            if output is not None:
                try:
                    output.close()
                except Exception:
                    pass

    def _write_locked(self, line: str, *, raw: bool = False) -> None:
        """A failed diagnostic file must not interrupt socket processing."""
        output = self._raw_file if raw else self._file
        if output is None:
            return
        try:
            size = len(line.encode("utf-8"))
            if size > self.max_bytes // 2:
                return
            if self._pair_bytes + size > self.max_bytes // 2:
                self.start(self._directory)
                output = self._raw_file if raw else self._file
            output.write(line)
            output.flush()
            path = self._raw_path if raw else self._path
            self._sizes[path] = self._sizes.get(path, 0) + size
            self._pair_bytes += size
            self._prune_locked()
        except Exception:
            self._close_locked()
            logging.getLogger(__name__).warning("Packet logging disabled after a write failure", exc_info=True)

    def log_raw_rx(
        self,
        protocol: str,
        data: bytes,
        *,
        source: tuple[str, int],
        reflector: str,
        module: str,
        connection: int,
    ) -> None:
        """Preserve a complete received datagram without interpreting its bytes."""
        with self._lock:
            if self._raw_file is None:
                return
            self._raw_packet_count += 1
            timestamp = datetime.now().astimezone().isoformat(timespec="microseconds")
            elapsed = time.perf_counter() - self._started_at_perf
            self._write_locked(
                f"{timestamp} (+{elapsed:.6f}s) [RX-RAW] {protocol} "
                f"packet={self._raw_packet_count} connection={connection} "
                f"source={source[0]}:{source[1]} reflector={reflector} module={module} "
                f"length={len(data)} data={data.hex()}\n",
                raw=True,
            )

    @staticmethod
    def _payload_details(payload: bytes | None) -> tuple[str, str, str]:
        """Format the same packet interpretation for receive and transmit logs."""
        payload_str = payload.hex() if payload else "None"
        if payload is None or len(payload) != 12:
            return payload_str, "", ""
        plain = _xor_scrambler(payload)
        fast_info = ""
        if plain[4] == 0x02:
            if (plain[9] & 0xF0) in (0x80, 0x90) and plain[9] != 0x83:
                length = (plain[9] & 0x0F) + (16 if (plain[9] & 0xF0) == 0x90 else 0)
                fast_info = f" | FastA(len={length})"
            elif plain[9] == 0x83:
                fast_info = " | FastB"
            else:
                fast_info = f" | Fast?(0x{plain[9]:02X})"
        elif plain[9:12] == b"fff":
            fast_info = " | FastFill"
        return payload_str, f" plain={plain.hex()}", fast_info

    def _format_time(
        self, sent_at: float | None = None, sent_wall_time: datetime | None = None
    ) -> str:
        captured = sent_at is not None or sent_wall_time is not None
        now_dt = (sent_wall_time if sent_wall_time is not None else datetime.now()).strftime("%H:%M:%S.%f")
        if not captured:
            now_dt = now_dt[:-3]
        if self._started_at_perf is not None:
            elapsed = (time.perf_counter() if sent_at is None else sent_at) - self._started_at_perf
            if captured:
                return f"{now_dt} ({elapsed:+.6f}s)"
            return f"{now_dt} (+{elapsed:6.3f}s)"
        return f"{now_dt}"

    def log_rx(
        self,
        protocol: str,
        stream_id: int,
        sequence: int | None,
        kind: str,
        dv_payload: bytes | None = None,
        details: str = "",
    ) -> None:
        """Log an inbound packet received from the reflector socket."""
        with self._lock:
            if self._file is None:
                return
            t_str = self._format_time()
            seq_str = f"seq={sequence:02d}" if sequence is not None else "seq=--"
            stream_str = f"stream={stream_id:04X}"
            payload_str, plain_str, fast_info = self._payload_details(dv_payload)

            extra = f" {details}" if details else ""
            line = (
                f"{t_str} [RX] {protocol:<6} {stream_str} {seq_str} "
                f"kind={kind:<5} payload={payload_str}{plain_str}{fast_info}{extra}\n"
            )
            self._write_locked(line)

    def log_tx(
        self,
        protocol: str,
        stream_id: int,
        sequence: int,
        packet_counter: int,
        kind: str,
        dv_payload: bytes | None = None,
        is_sync: bool = False,
        sleep_ms: float = 0.0,
        details: str = "",
        *,
        sent_at: float | None = None,
        sent_wall_time: datetime | None = None,
    ) -> None:
        """Log an outbound packet transmitted to the reflector socket."""
        with self._lock:
            if self._file is None:
                return
            t_str = self._format_time(sent_at, sent_wall_time)
            seq_str = f"seq={sequence:02d}"
            cnt_str = f"cnt={packet_counter:06X}"
            stream_str = f"stream={stream_id:04X}"
            payload_str, plain_str, fast_info = self._payload_details(dv_payload)
            sync_tag = " [SYNC]" if is_sync else ""
            sleep_str = f" sleep={sleep_ms:5.1f}ms" if sleep_ms > 0 else ""

            extra = f" {details}" if details else ""
            line = (
                f"{t_str} [TX] {protocol:<6} {stream_str} {seq_str} {cnt_str} "
                f"kind={kind:<5}{sync_tag} payload={payload_str}{plain_str}{sleep_str}{fast_info}{extra}\n"
            )
            self._write_locked(line)

    def log_gap(
        self,
        direction: str,
        stream_id: int,
        expected: int,
        actual: int | None,
        note: str = "",
    ) -> None:
        """Log a detected sequence or chunk gap."""
        with self._lock:
            if self._file is None:
                return
            t_str = self._format_time()
            extra = f" ({note})" if note else ""
            actual_str = f"{actual:02d}" if actual is not None else "--"
            line = (
                f"{t_str} [{direction.upper()}-GAP] stream={stream_id:04X} "
                f"expected={expected:02d}, got={actual_str}{extra}\n"
            )
            self._write_locked(line)

    def log_event(
        self, event_type: str, message: str, *,
        sent_at: float | None = None, sent_wall_time: datetime | None = None,
        **fields: object,
    ) -> None:
        """Log a high-level application or transport event."""
        with self._lock:
            if self._file is None:
                return
            t_str = self._format_time(sent_at, sent_wall_time)
            field_str = f" {fields}" if fields else ""
            line = f"{t_str} [EVENT] type={event_type} {message}{field_str}\n"
            self._write_locked(line)
