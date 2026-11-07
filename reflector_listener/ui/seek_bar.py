"""Keyboard-accessible playback scrubber with a round teal thumb."""

import tkinter as tk
from typing import Callable


class SeekBar(tk.Canvas):
    def __init__(self, parent, *, on_preview: Callable[[float], None],
                 on_seek: Callable[[float], None], background: str) -> None:
        super().__init__(parent, width=220, height=32, background=background,
                         borderwidth=0, highlightthickness=1,
                         highlightbackground=background, highlightcolor="#3fbfa7",
                         takefocus=True, cursor="hand2")
        self._on_preview, self._on_seek = on_preview, on_seek
        self._value = 0.0
        self._duration = 1.0
        self._enabled = True
        self.dragging = False
        self._track = self.create_line(0, 0, 0, 0, fill="#d8d8d8", width=4, capstyle="round")
        self._progress = self.create_line(0, 0, 0, 0, fill="#3fbfa7", width=4, capstyle="round")
        self._thumb = self.create_oval(0, 0, 0, 0, fill="#3fbfa7", outline="")
        self.bind("<Configure>", self._draw)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        for key in ("Left", "Right", "Home", "End"):
            self.bind(f"<Key-{key}>", self._key)

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        value = max(0.0, min(value, self._duration))
        if value == self._value:
            return
        self._value = value
        self._draw()

    def set_range(self, duration: float, *, enabled: bool) -> None:
        duration = max(0.02, duration)
        if duration == self._duration and enabled == self._enabled:
            return
        self._duration = duration
        self._enabled = enabled
        if not enabled:
            self.dragging = False
        self.configure(takefocus=enabled, cursor="hand2" if enabled else "")
        self._value = min(self._value, self._duration)
        self._draw()

    def _draw(self, _event=None) -> None:
        left, right = 10, max(11, self.winfo_width() - 10)
        middle = self.winfo_height() / 2
        head = left + (right - left) * self._value / self._duration
        color = "#3fbfa7" if self._enabled else "#adb6b8"
        self.coords(self._track, left, middle, right, middle)
        self.coords(self._progress, left, middle, head, middle)
        self.itemconfigure(self._progress, fill=color, state="normal" if head > left else "hidden")
        self.coords(self._thumb, head - 7, middle - 7, head + 7, middle + 7)
        self.itemconfigure(self._thumb, fill=color)

    def _move(self, x: float) -> None:
        self.set((x - 10) / max(1, self.winfo_width() - 20) * self._duration)
        self._on_preview(self._value)

    def _press(self, event) -> str:
        if self._enabled:
            self.focus_set()
            self.dragging = True
            self._move(event.x)
        return "break"

    def _motion(self, event) -> str:
        if self.dragging:
            self._move(event.x)
        return "break"

    def _release(self, event) -> str:
        if self.dragging:
            self._move(event.x)
            self.dragging = False
            self._on_seek(self._value)
        return "break"

    def _key(self, event) -> str:
        if self._enabled:
            value = {"Home": 0, "End": self._duration,
                     "Left": self._value - 5, "Right": self._value + 5}[event.keysym]
            self.set(value)
            self._on_preview(self._value)
            self._on_seek(self._value)
        return "break"
