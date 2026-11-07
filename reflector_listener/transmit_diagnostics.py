"""Bounded, asynchronous diagnostics for a single application transmission."""

from __future__ import annotations

from datetime import datetime
import logging
import queue
import threading
import time
from typing import Any

from .packet_logger import PacketLogger


class TransmitDiagnostics:
    """Keep diagnostic disk work off the frame sender.

    Logging enabled at construction applies to this transmission. Submission
    never waits for disk or queue space; overflow discards only diagnostics.
    Call ``close`` after sending, outside the session lifecycle lock, to drain.
    """

    def __init__(self, logger: PacketLogger, *, max_pending: int = 256) -> None:
        if max_pending < 1:
            raise ValueError("Transmit diagnostic capacity must be positive")
        self._logger = logger
        self._pending: queue.Queue[tuple[str, tuple[Any, ...], dict[str, Any]] | None] = queue.Queue(max_pending)
        self._state_lock = threading.Lock()
        self._closed = False
        self._omitted = 0
        self._thread: threading.Thread | None = None
        if logger.is_enabled:
            self._thread = threading.Thread(target=self._run, name="transmit-diagnostics", daemon=True)
            try:
                self._thread.start()
            except RuntimeError:
                self._thread = None
                logging.getLogger(__name__).warning("Cannot start transmit diagnostics", exc_info=True)

    def log_tx(self, *args: Any, **kwargs: Any) -> None:
        self._submit("log_tx", args, kwargs)

    def log_event(self, event_type: str, message: str, **fields: Any) -> None:
        self._submit("log_event", (event_type, message), fields)

    def _submit(self, method: str, args: tuple[Any, ...], fields: dict[str, Any]) -> None:
        if self._thread is None:
            return
        # Capture at submission, before a slow diagnostic writer can delay it.
        if fields.get("sent_at") is None:
            fields["sent_at"] = time.perf_counter()
        if fields.get("sent_wall_time") is None:
            fields["sent_wall_time"] = datetime.now()
        with self._state_lock:
            if self._closed:
                return
            try:
                self._pending.put_nowait((method, args, fields))
            except queue.Full:
                self._omitted += 1

    def _run(self) -> None:
        failed = False
        while (record := self._pending.get()) is not None:
            if failed:
                continue
            method, args, fields = record
            try:
                getattr(self._logger, method)(*args, **fields)
            except Exception:
                failed = True
                logging.getLogger(__name__).warning("Transmit diagnostics disabled after a logger failure", exc_info=True)
        if self._omitted and not failed:
            try:
                self._logger.log_event(
                    "tx_diagnostics_dropped",
                    f"Omitted {self._omitted} transmit diagnostic records because the log queue filled; network packets were unaffected",
                    omitted=self._omitted,
                )
            except Exception:
                logging.getLogger(__name__).warning("Cannot report omitted transmit diagnostics", exc_info=True)

    def close(self) -> None:
        """Finish queued writes after transmission; safe to call repeatedly."""
        if self._thread is None:
            return
        with self._state_lock:
            first_close = not self._closed
            self._closed = True
        if first_close:
            self._pending.put(None)
        self._thread.join()
