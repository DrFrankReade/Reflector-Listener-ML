"""Scrollable message rows without a single oversized native Tk frame."""

import tkinter as tk


class MessageTimeline(tk.Canvas):
    """Lay out rows in canvas coordinates; map only rows near the viewport.

    A long history can exceed Windows' native widget height limits. Individual
    row windows keep their natural heights regardless of total history length.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._rows = {}
        self._layout_after = None
        self._scroll_callback = None
        self.configure(yscrollcommand=self._scrolled)
        self.bind("<Configure>", self._schedule_layout)
        self.bind("<Destroy>", self._destroy)

    def set_scrollbar(self, callback):
        self._scroll_callback = callback

    def add_row(self, row, *, padx=0, pady=0):
        item = self.create_window(0, 0, window=row, anchor="nw", state="hidden")
        self._rows[row] = [item, padx, pady, 0, 0]
        row.bind("<Configure>", self._schedule_layout, add="+")
        row.bind("<Destroy>", lambda event: self._remove_row(row) if event.widget is row else None, add="+")
        self._schedule_layout()

    def _remove_row(self, row):
        entry = self._rows.pop(row, None)
        if entry is not None:
            self.delete(entry[0])
            self._schedule_layout()

    def _schedule_layout(self, _event=None):
        if self._layout_after is None:
            self._layout_after = self.after_idle(self._layout)

    def _layout(self):
        self._layout_after = None
        at_bottom = self.yview()[1] >= 0.999
        width = self.winfo_width()
        y = 0
        for row, entry in self._rows.items():
            item, padx, pady, _, _ = entry
            height = row.winfo_reqheight()
            entry[3:] = [y + pady, height]
            self.coords(item, padx, y + pady)
            # Leave height automatic: Tk then emits Configure when a playback
            # panel changes the row's requested height. A fixed height clips it.
            self.itemconfigure(item, width=max(1, width - 2 * padx))
            y += height + 2 * pady
        self.configure(scrollregion=(0, 0, width, y))
        if at_bottom:
            self.yview_moveto(1.0)
        self._show_visible()

    def _scrolled(self, first, last):
        if self._scroll_callback is not None:
            self._scroll_callback(first, last)
        self._show_visible()

    def _show_visible(self):
        top = self.canvasy(0)
        bottom = top + self.winfo_height()
        for item, _, _, y, height in self._rows.values():
            state = "normal" if y + height >= top and y <= bottom else "hidden"
            if self.itemcget(item, "state") != state:
                self.itemconfigure(item, state=state)

    def _destroy(self, event):
        if event.widget is self and self._layout_after is not None:
            self.after_cancel(self._layout_after)
            self._layout_after = None
