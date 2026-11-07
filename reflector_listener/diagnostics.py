"""Optional diagnostic interfaces shared by protocol receivers and loggers."""

from __future__ import annotations

import logging
from typing import Protocol


class DiagnosticLogger(Protocol):
    """The diagnostic surface needed by application-data receivers."""

    def log_gap(
        self,
        direction: str,
        stream_id: int,
        expected: int,
        actual: int | None,
        note: str = "",
    ) -> None: ...

    def log_event(self, event_type: str, message: str, **fields: object) -> None: ...


class SafeDiagnostics:
    """Keep optional diagnostic failures outside the receive path.

    A failing delegate is disabled until replaced; report its first failure to
    standard logging instead of repeatedly raising for every received packet.
    """

    def __init__(self, logger: DiagnosticLogger | None = None) -> None:
        self._logger = logger

    def _disable(self) -> None:
        self._logger = None
        logging.getLogger(__name__).warning(
            "Receiver diagnostics disabled after a logger failure", exc_info=True
        )

    def log_gap(
        self,
        direction: str,
        stream_id: int,
        expected: int,
        actual: int | None,
        note: str = "",
    ) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log_gap(direction, stream_id, expected, actual, note)
        except Exception:
            self._disable()

    def log_event(self, event_type: str, message: str, **fields: object) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log_event(event_type, message, **fields)
        except Exception:
            self._disable()
