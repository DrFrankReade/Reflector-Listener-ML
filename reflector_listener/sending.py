"""Own the lifetime of one outgoing message, including its preparation.

The controller has no GUI dependencies. Its callbacks run on the worker thread;
UI callers should enqueue their results for dispatch on the UI thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class SendOperation:
    """Unique identity and cancellation token for a single send attempt."""

    operation_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class PreparedSend:
    payload: bytes
    label: str = "application data"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SendResult:
    operation: SendOperation
    completed: bool = False
    cancelled: bool = False
    error: Exception | None = None


class ApplicationTransmitter(Protocol):
    def __call__(
        self,
        payload: bytes,
        *,
        label: str,
        metadata: Mapping[str, Any],
        cancel_event: threading.Event,
    ) -> bool: ...


def _start_worker(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="rsms1-send", daemon=True).start()


class SendController:
    """Serialize preparation and transmission, using a token that never resets.

    Cancellation remains busy until the worker acknowledges it. This prevents a
    replacement send from racing the old session's socket cleanup. Operation IDs
    must be unique; ``is_current`` keeps the latest identity after completion so
    a queued UI result can still be accepted until another operation starts.
    """

    def __init__(
        self,
        transmit: ApplicationTransmitter,
        on_finished: Callable[[SendResult], None],
        *,
        start_worker: Callable[[Callable[[], None]], None] = _start_worker,
    ) -> None:
        self._transmit = transmit
        self._on_finished = on_finished
        self._start_worker = start_worker
        self._lock = threading.Lock()
        self._active: SendOperation | None = None
        self._current: SendOperation | None = None

    @property
    def active(self) -> SendOperation | None:
        with self._lock:
            return self._active

    def is_current(self, operation_id: str) -> bool:
        with self._lock:
            return self._current is not None and self._current.operation_id == operation_id

    def start(
        self,
        operation_id: str,
        prepare: Callable[[SendOperation], PreparedSend],
    ) -> SendOperation:
        """Reserve an operation before launching its preparation worker."""
        if not operation_id:
            raise ValueError("a send operation needs a unique ID")
        operation = SendOperation(operation_id)
        with self._lock:
            if self._active is not None:
                raise RuntimeError("another send is already active")
            if self._current is not None and self._current.operation_id == operation_id:
                raise ValueError("a send operation ID cannot be reused")
            self._active = self._current = operation
        try:
            self._start_worker(lambda: self._run(operation, prepare))
        except Exception:
            # Failure to create a thread must not leave the controller busy.
            with self._lock:
                if self._active is operation:
                    self._active = None
            raise
        return operation

    def cancel(self, operation_id: str | None = None) -> bool:
        """Signal the current worker, optionally only if its identity matches."""
        with self._lock:
            operation = self._active
            if operation is None or (
                operation_id is not None and operation.operation_id != operation_id
            ):
                return False
            operation.cancel_event.set()
            return True

    def _run(
        self,
        operation: SendOperation,
        prepare: Callable[[SendOperation], PreparedSend],
    ) -> None:
        completed = False
        error: Exception | None = None
        try:
            if not operation.cancel_event.is_set():
                prepared = prepare(operation)
                if not operation.cancel_event.is_set():
                    completed = self._transmit(
                        prepared.payload,
                        label=prepared.label,
                        metadata=dict(prepared.metadata),
                        cancel_event=operation.cancel_event,
                    )
        except Exception as exc:
            error = exc
        finally:
            # Cancellation and completion are ordered under the same lock.
            # A late worker must never release a newer operation's reservation.
            with self._lock:
                cancelled = operation.cancel_event.is_set()
                if self._active is operation:
                    self._active = None
            self._on_finished(
                SendResult(
                    operation,
                    completed=completed and not cancelled,
                    cancelled=cancelled,
                    error=None if cancelled else error,
                )
            )
