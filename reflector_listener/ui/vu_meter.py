"""Small segmented audio meter; all updates are made on the Tk thread."""
import math
import tkinter as tk


class VuMeter(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=28, width=200, background="#292d30",
                         highlightthickness=1, highlightbackground="#585e62", **kwargs)
        self.level = 0.0
        self.bind("<Configure>", lambda _event: self._draw())

    def set_level(self, level: float) -> None:
        try:
            value = float(level)
        except (ValueError, TypeError):
            value = 0.0
        self.level = max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0
        self._draw()

    @staticmethod
    def fraction(level: float) -> float:
        return max(0.0, min(1.0, (20 * math.log10(max(level, 1e-9)) + 60) / 60))

    def _draw(self):
        self.delete("bar")
        width, height = self.winfo_width(), self.winfo_height()
        count = max(1, min(72, (width - 8) // 6))
        step = max(1.0, (width - 8) / count)
        lit = round(self.fraction(self.level) * count)
        for index in range(count):
            position = (index + 1) / count
            bright, dim = (("#e6453d", "#542823") if position > .95 else
                           ("#e6cb32", "#514b22") if position > .8 else
                           ("#39ce49", "#24482b"))
            x = 4 + index * step
            self.create_rectangle(x, 4, x + max(1, step - 2), max(5, height - 4),
                                  fill=bright if index < lit else dim, outline="", tags="bar")
