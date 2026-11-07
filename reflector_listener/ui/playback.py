"""Single-row replay controls disclosed by the message's From label."""

import tkinter as tk

from .seek_bar import SeekBar
from .tooltip import ToolTip


def timecode(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class PlaybackControl:
    def __init__(self, bubble, header, clip, controller) -> None:
        self.clip, self.controller, self.header = clip, controller, header
        self.open = False
        self.position = 0.0
        self._pending_seek = False
        self._after = None
        self.panel = None
        self._bubble = bubble
        self._last_snapshot = None
        self._header_text = header.cget("text")
        header.configure(cursor="hand2", takefocus=True)
        for event in ("<Button-1>", "<Return>", "<space>"):
            header.bind(event, self._toggle, add="+")
        self.set_header(self._header_text)

    def _build_panel(self) -> None:
        background, foreground = self._bubble.cget("background"), "#596775"
        self.panel = tk.Frame(self._bubble, background=background, padx=0, pady=2)
        self.panel.columnconfigure(3, weight=1)
        button_style = dict(background=background, foreground=foreground,
                            activebackground="#eff7f5", activeforeground=foreground,
                            disabledforeground="#adb6b8", relief="flat", borderwidth=0,
                            highlightthickness=1, highlightbackground=background,
                            highlightcolor="#3fbfa7", cursor="hand2", padx=5, pady=0)
        self.play_button = tk.Button(self.panel, text="\u25b6", font=("Segoe UI Symbol", 16),
                                     command=self._play, **button_style)
        self.play_button.grid(row=0, column=0)
        self.play_tip = ToolTip(self.play_button, "Play recording")
        self.stop_button = tk.Button(self.panel, text="\u25a0", font=("Segoe UI Symbol", 12),
                                     command=self._stop, **button_style)
        self.stop_button.grid(row=0, column=1, padx=(0, 4))
        ToolTip(self.stop_button, "Stop and return to the beginning")
        label_style = dict(background=background, foreground=foreground, font=("Segoe UI", 11))
        self.elapsed = tk.Label(self.panel, text="0:00", **label_style)
        self.elapsed.grid(row=0, column=2, padx=(2, 6))
        ToolTip(self.elapsed, "Elapsed playback time")
        self.scale = SeekBar(self.panel, on_preview=self._preview, on_seek=self._seek,
                             background=background)
        self.scale.grid(row=0, column=3, sticky="ew")
        ToolTip(self.scale, "Seek in the recording. Arrow keys skip 5 seconds; Home/End jump to either end.")
        self.duration = tk.Label(self.panel, text="0:00", **label_style)
        self.duration.grid(row=0, column=4, padx=(6, 4))
        ToolTip(self.duration, "Recording duration")
        # Only failures need a caption; recording and loading states use tooltips.
        self.error = tk.Label(self.panel, text="", background=background,
                              foreground="#a31515", font=("Segoe UI", 9), wraplength=330)
        self.panel.bind("<Destroy>", self._destroy, add="+")

    def set_header(self, text: str, **style) -> None:
        self._header_text = text
        marker = "\u25bc" if self.open else "\u25b6"
        self.header.configure(text=f"{marker} {text}", **style)
        if self.panel is not None and "background" in style:
            background = style["background"]
            self.panel.configure(background=background)
            for widget in self.panel.winfo_children():
                widget.configure(background=background)
                if isinstance(widget, (tk.Button, SeekBar)):
                    widget.configure(highlightbackground=background)

    def set_clip(self, clip) -> None:
        if clip is self.clip:
            return
        if self.controller.state.token == self.clip.token:
            self.controller.pause()
        self.clip = clip
        self._last_snapshot = None
        if self.open:
            self._refresh()

    def _toggle(self, _event=None):
        self.open = not self.open
        self.set_header(self._header_text)
        if self.open:
            if self.panel is None:
                self._build_panel()
            self.panel.pack(fill="x", pady=(5, 0))
            self._update()
        else:
            self.panel.pack_forget()
            if self._after:
                self.panel.after_cancel(self._after)
                self._after = None
        return "break"

    def _play(self):
        state = self.controller.state
        if state.token == self.clip.token and state.playing:
            self.controller.pause()
        elif self.clip.complete and not self.clip.expired:
            at_end = state.token == self.clip.token and state.duration > 0 and state.position >= state.duration
            position = 0 if at_end and not self._pending_seek else self.position
            self._pending_seek = False
            self.controller.play(self.clip, position)
        self._refresh()

    def _stop(self):
        if self.controller.state.token == self.clip.token:
            self.controller.stop()
        self._pending_seek = False
        self.position = 0
        self.scale.set(0)
        self._refresh()

    def _preview(self, position):
        self.elapsed.configure(text=timecode(position))

    def _seek(self, position):
        self.position = position
        state = self.controller.state
        self._pending_seek = True
        if state.token == self.clip.token and state.playing:
            self._pending_seek = False
            self.controller.play(self.clip, position)
        elif state.token == self.clip.token:
            # Keep a paused seek local until Play is pressed.
            self.controller.stop()
        self._preview(position)

    def _refresh(self):
        state = self.controller.state
        snapshot = (state, self.clip.frames, self.clip.complete, self.clip.expired,
                    self.position, self._pending_seek, self.scale.dragging)
        if snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot
        active = state.token == self.clip.token
        enabled = self.clip.complete and not self.clip.expired
        self._refresh_position(state, active=active, enabled=enabled)
        self._refresh_transport(state, active=active, enabled=enabled)
        self._refresh_error(state, active=active)

    def _refresh_position(self, state, *, active: bool, enabled: bool) -> None:
        """Follow playback unless the user is dragging or has a paused seek."""
        duration = state.duration if active and state.duration else self.clip.frames * 0.02
        if active and not self.scale.dragging and not self._pending_seek:
            self.position = state.position
        self.scale.set_range(duration, enabled=enabled)
        if not self.scale.dragging:
            self.scale.set(self.position)
        self.elapsed.configure(text=timecode(self.scale.get()))
        self.duration.configure(text=timecode(duration))

    def _refresh_transport(self, state, *, active: bool, enabled: bool) -> None:
        playing = active and state.playing
        self.play_button.configure(text="\u23f8" if playing else "\u25b6",
                                   state="normal" if enabled else "disabled")
        self.stop_button.configure(state="normal" if enabled else "disabled")
        self.play_tip.set_text(self._play_tooltip(state, active=active))

    def _play_tooltip(self, state, *, active: bool) -> str:
        if self.clip.expired:
            return "Recording expired"
        if not self.clip.complete:
            return "Available when this transmission ends"
        if active and state.loading:
            return "Pause loading"
        if active and state.playing:
            return "Pause recording"
        return "Play recording"

    def _refresh_error(self, state, *, active: bool) -> None:
        error = "Recording expired" if self.clip.expired else state.error if active else ""
        self.error.configure(text=error)
        if error:
            self.error.grid(row=1, column=0, columnspan=5, sticky="w", pady=(3, 0))
        else:
            self.error.grid_remove()

    def _update(self):
        self._after = None
        if self.open:
            self._refresh()
            state = self.controller.state
            interval = 100 if state.token == self.clip.token and state.playing else 500
            self._after = self.panel.after(interval, self._update)

    def _destroy(self, event):
        if event.widget == self.panel and self._after:
            self.panel.after_cancel(self._after)
            self._after = None
