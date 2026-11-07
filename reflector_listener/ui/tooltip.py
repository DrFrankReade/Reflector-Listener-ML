from __future__ import annotations
import tkinter as tk
from typing import Any

class ToolTip:
    """Lightweight tooltip popup for widgets."""

    def __init__(self, widget: tk.Widget, text: str = "") -> None:
        self.widget = widget
        self.text = text
        self.suffix = lambda: ""
        self.tip_window: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")
        widget.bind("<Destroy>", self._on_leave, add="+")

    def display_text(self) -> str:
        return "\n".join(part for part in (self.text, self.suffix()) if part)

    def set_text(self, text: str) -> None:
        self.text = text
        if self.tip_window is not None:
            for child in self.tip_window.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(text=self.display_text())

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
        self._after_id = self.widget.after(300, self._show_tip)

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide_tip()

    def _show_tip(self) -> None:
        self._after_id = None
        if not self.display_text() or self.tip_window is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tw,
                text=self.display_text(),
                justify="left",
                background="#2c2c2c",
                foreground="#ffffff",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 9),
                padx=8,
                pady=4,
                wraplength=460,
            )
            label.pack()
        except Exception:
            pass

    def _hide_tip(self) -> None:
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None

