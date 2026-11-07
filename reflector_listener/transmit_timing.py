"""Monotonic D-STAR application pacing and socket-send timing measurements."""

from __future__ import annotations

import time
from typing import Callable


class FramePacer:
    """Keep 20 ms cadence, allowing at most 1 ms recovery after a late send.

    Record completion of the actual socket send, after any lock contention.
    A missed deadline must never cause a burst of queued picture frames.
    """

    interval = 0.020
    recovery = 0.001

    def __init__(
        self, *, clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._deadline = clock() + self.interval
        self.last_sent_at: float | None = None
        self._intervals = 0
        self._total = 0.0
        self._minimum = float("inf")
        self._maximum = 0.0
        self._over_25ms = 0
        self._max_lateness = 0.0

    def wait(self) -> float:
        """Wait outside the session lock; return the measured wait in ms."""
        before = self._clock()
        remaining = self._deadline - before
        if remaining > 0:
            self._sleep(remaining)
        return (self._clock() - before) * 1000

    def sent(self, sent_at: float) -> None:
        """Advance only after the socket accepted a complete datagram."""
        self._max_lateness = max(self._max_lateness, sent_at - self._deadline)
        if self.last_sent_at is not None:
            gap = sent_at - self.last_sent_at
            self._intervals += 1
            self._total += gap
            self._minimum = min(self._minimum, gap)
            self._maximum = max(self._maximum, gap)
            self._over_25ms += gap > 0.025
        self.last_sent_at = sent_at
        self._deadline = max(self._deadline, sent_at - self.recovery) + self.interval

    def wait_end(self) -> None:
        """Let the last DV frame occupy its full slot before a clean END."""
        if self.last_sent_at is not None:
            remaining = self.last_sent_at + self.interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)

    def summary(self) -> dict[str, int | float]:
        """Socket acceptance timing, independent of diagnostic disk latency."""
        return {
            "intervals": self._intervals,
            "mean_interval_ms": round(self._total * 1000 / self._intervals, 3) if self._intervals else 0.0,
            "min_interval_ms": round(self._minimum * 1000, 3) if self._intervals else 0.0,
            "max_interval_ms": round(self._maximum * 1000, 3),
            "intervals_over_25ms": self._over_25ms,
            "max_lateness_ms": round(self._max_lateness * 1000, 3),
        }
