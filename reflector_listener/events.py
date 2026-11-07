"""Shared worker events and bounded, failure-isolated delivery to the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import queue
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


class EventKind(str, Enum):
    STATE = "state"
    ERROR = "error"
    WARNING = "warning"
    NETWORK = "network"
    DECODER = "decoder"
    STREAM_START = "stream_start"
    VOICE_PROGRESS = "voice_progress"
    STREAM_END = "stream_end"
    APP_STREAM = "app_stream"
    APP_TEXT = "app_text"
    APP_GPS = "app_gps"
    APP_PICTURE = "app_picture"
    APP_STREAM_END = "app_stream_end"
    OUTGOING_PICTURE = "outgoing_picture"
    APP_TX_START = "app_tx_start"
    APP_TX_PROGRESS = "app_tx_progress"
    APP_TX_CANCEL = "app_tx_cancel"
    APP_TX_END = "app_tx_end"
    APP_SEND_FINISHED = "app_send_finished"
    TX_START = "tx_start"
    TX_PROGRESS = "tx_progress"
    TX_END = "tx_end"
    TX_TIMEOUT = "tx_timeout"
    TX_CAPTURE_STOP = "tx_capture_stop"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SessionEvent:
    """An immutable envelope; string kinds remain supported for extensions.

    Payload objects such as images are transferred to the receiving thread and
    must not be mutated by the producer after enqueueing.
    """

    kind: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Snapshot the envelope while retaining large transferred payloads (e.g.
        # images) by reference. Producers no longer own those payload objects.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


class ApplicationEvent(Protocol):
    """Read-only event contract for injectable application-data receivers."""

    @property
    def kind(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def fields(self) -> Mapping[str, Any]: ...


def drain_events(
    events: queue.SimpleQueue[SessionEvent],
    dispatch: Callable[[SessionEvent], None],
    report_error: Callable[[SessionEvent, Exception], None],
    *,
    limit: int = 100,
) -> int:
    """Deliver at most ``limit`` events so input and paint callbacks can run.

    A bad event (including a failing error reporter) cannot discard later
    events or prevent the caller from scheduling its next UI tick.
    """
    if limit < 1:
        raise ValueError("event limit must be positive")
    processed = 0
    while processed < limit:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        processed += 1
        try:
            dispatch(event)
        except Exception as exc:
            logging.getLogger(__name__).exception("Failed to handle %s event", event.kind)
            try:
                report_error(event, exc)
            except Exception:
                logging.getLogger(__name__).exception("Failed to report event error")
    return processed
