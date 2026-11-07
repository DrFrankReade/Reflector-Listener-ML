"""Modal, tabbed settings editor with isolated drafts and validated Save."""

from __future__ import annotations

from dataclasses import asdict
import re
import tkinter as tk
from tkinter import font as tkfont, messagebox, ttk
from typing import Callable

from reflector_listener.location import grid_center
from reflector_listener.preferences import Preferences
from reflector_listener.audio_devices import enumerate_audio_devices
from .tooltip import ToolTip
from .avatars import AvatarPicker
from .vu_meter import VuMeter


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, preferences: Preferences, *, callsign: str, module: str,
                 debug: bool, connected: bool, on_save: Callable, open_logs: Callable,
                 clear_logs: Callable, on_audio_preview: Callable | None = None,
                 get_microphone_level: Callable | None = None) -> None:
        super().__init__(parent)
        self.title("Settings")
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(590, 460)
        self.geometry("760x500")
        self.on_save = on_save
        self._on_audio_preview = on_audio_preview or (lambda device, gain, enabled: None)
        self._get_microphone_level = get_microphone_level or (lambda: 0.0)
        self._preview_active = False
        self._preview_after = None
        self.vars = {key: (tk.BooleanVar(self, value=value) if isinstance(value, bool)
                           else tk.StringVar(self, value=str(value)))
                     for key, value in asdict(preferences).items()}
        self.vars.update(callsign=tk.StringVar(self, value=callsign),
                         module=tk.StringVar(self, value=module),
                         debug=tk.BooleanVar(self, value=debug))
        self.vars["transmit_gain"] = tk.DoubleVar(self, value=preferences.transmit_gain)
        self._from_grid = False
        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="Settings", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(container, text="Preferences are saved on this computer.").pack(anchor="w", pady=(3, 14))
        notebook = ttk.Notebook(container)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True)
        ToolTip(notebook, "Choose a settings section. Changes take effect when you select Save.")
        self.sections = {}
        for title in ("Station", "Location", "Transmission", "Audio", "Appearance", "Logging"):
            tab = ttk.Frame(notebook, padding=18)
            tab.columnconfigure(1, weight=1)
            notebook.add(tab, text=f" {title} ")
            self.sections[title] = tab
        self._station(connected)
        self._location()
        self._transmission()
        self._audio()
        self._appearance()
        self._logging(open_logs, clear_logs)
        notebook.bind("<<NotebookTabChanged>>", self._preview_changed)
        footer = ttk.Frame(container)
        footer.pack(fill="x", pady=(16, 0))
        cancel = ttk.Button(footer, text="Cancel", command=self.destroy)
        cancel.pack(side="right")
        ToolTip(cancel, "Close without applying changes to preferences.")
        save = ttk.Button(footer, text="Save", command=self._save)
        save.pack(side="right", padx=(0, 8))
        ToolTip(save, "Validate, apply, and save these settings for future launches.")
        self.bind("<Escape>", lambda _e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after_idle(self._fit_device_names)
        self.grab_set()
        self.focus_set()

    def _control(self, tab, key, tip, *, values=None, state="normal", numeric=False,
                 width=24, maxlength=None, pattern=None):
        if values is not None:
            control = ttk.Combobox(tab, textvariable=self.vars[key], values=values,
                                   state="readonly" if state == "normal" else state, width=width)
        elif numeric:
            control = ttk.Spinbox(tab, textvariable=self.vars[key], from_=1, to=100000, width=width)
        else:
            control = ttk.Entry(tab, textvariable=self.vars[key], state=state, width=width)
        if maxlength is not None or pattern is not None:
            def valid(proposed):
                return ((maxlength is None or len(proposed) <= maxlength)
                        and (pattern is None or re.fullmatch(pattern, proposed) is not None))
            control.configure(validate="key", validatecommand=(self.register(valid), "%P"))
        ToolTip(control, tip)
        return control

    def _entry(self, tab, row, label, key, tip, **options):
        ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=8)
        control = self._control(tab, key, tip, **options)
        control.grid(row=row, column=1, sticky="w", pady=8)
        return control

    def _check(self, tab, row, text, key, tip):
        control = ttk.Checkbutton(tab, text=text, variable=self.vars[key])
        control.grid(row=row, column=0, columnspan=2, sticky="w", pady=9)
        ToolTip(control, tip)
        return control

    def _station(self, connected):
        tab = self.sections["Station"]
        state = "disabled" if connected else "normal"
        ttk.Label(tab, text="Call sign / module").grid(row=0, column=0, sticky="w", padx=(0, 20), pady=8)
        identity = ttk.Frame(tab)
        identity.grid(row=0, column=1, sticky="w")
        self._control(identity, "callsign", "Your licensed call sign, 3–8 characters. Disconnect before changing station identity.",
                      state=state, width=10, maxlength=8).pack(side="left")
        ttk.Label(identity, text=" / ").pack(side="left")
        self._control(identity, "module", "Local station module A–Z. This is separate from the reflector module. Disconnect before changing it.",
                      state=state, values=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), width=3).pack(side="left")
        self._entry(tab, 1, "Call sign message", "station_message",
                    "Optional 20-character D-STAR station message, sent with voice, text, and pictures. Printable ASCII characters only.",
                    width=22, maxlength=20, pattern=r"[\x20-\x7e]*")
        self._entry(tab, 2, "Distance units", "units", "Units for great-circle distances in received-message tooltips. Bearings use degrees from true north.", values=("miles", "kilometers"))
        if connected:
            ttk.Label(tab, text="Disconnect to edit your call sign or station module.").grid(row=3, column=0, columnspan=2, sticky="w", pady=12)

    def _location(self):
        tab = self.sections["Location"]
        self._entry(tab, 0, "Grid square", "grid", "Enter a 4 or 6 character Maidenhead locator. Latitude and longitude are filled with its center.", width=8, maxlength=6)
        self._entry(tab, 1, "Latitude", "latitude", "Decimal degrees, −90 to 90, up to six decimal places. Leave both coordinates empty to clear your location.", width=13, pattern=r"[+-]?\d{0,2}(?:\.\d{0,6})?")
        self._entry(tab, 2, "Longitude", "longitude", "Decimal degrees, −180 to 180, up to six decimal places. Editing coordinates clears the grid entry.", width=13, pattern=r"[+-]?\d{0,3}(?:\.\d{0,6})?")
        self._check(tab, 3, "Send my location as D-PRS", "send_dprs", "OFF by default. When enabled, include this location with voice, text, and picture transmissions. This does not transmit an idle beacon.")
        ttk.Label(tab, text="Your location is also used for distance and bearing tooltips.", wraplength=480).grid(row=4, column=0, columnspan=2, sticky="w", pady=8)
        self.vars["grid"].trace_add("write", self._grid_changed)
        for key in ("latitude", "longitude"):
            self.vars[key].trace_add("write", self._coordinates_changed)

    def _transmission(self):
        tab = self.sections["Transmission"]
        for row, (label, key, default) in enumerate((("Voice suffix", "voice_suffix", "PC"), ("Picture suffix", "picture_suffix", "SSTV"), ("Text suffix", "text_suffix", "TEXT"))):
            self._entry(tab, row, label, key, f"Up to four letters or digits shown after your call sign as /ABCD. Default /{default}. Applies to the next transmission; blank sends no suffix.", width=6, maxlength=4)
        self._check(tab, 3, "Picture transfer heartbeat beep", "picture_beep", "ON is the default official behavior. Sends Icom-compatible notification tones about once a second during picture transfers. Your choice is saved across restarts.")

    def _audio(self):
        tab = self.sections["Audio"]
        self._device_choices = {}
        self.transmit_device_combo = self._device_selector(tab, 0, "Transmit with", "transmit_device", "input")
        ttk.Label(tab, text="Microphone gain").grid(row=1, column=0, sticky="w", padx=(0, 20), pady=6)
        gain_row = ttk.Frame(tab)
        gain_row.grid(row=1, column=1, sticky="ew")
        gain_row.columnconfigure(0, weight=1)
        self.transmit_gain_scale = ttk.Scale(gain_row, from_=0, to=6, orient="horizontal",
                                             variable=self.vars["transmit_gain"], command=self._preview_changed)
        self.transmit_gain_scale.grid(row=0, column=0, sticky="ew")
        self.transmit_gain_label = ttk.Label(gain_row, width=5, anchor="e")
        self.transmit_gain_label.grid(row=0, column=1, padx=(6, 0))
        ToolTip(self.transmit_gain_scale, "Adjust microphone level before transmission. The meter previews this gain without transmitting.")
        self.microphone_meter = VuMeter(tab)
        self.microphone_meter.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ToolTip(self.microphone_meter, "Live microphone level after gain. Green is normal; yellow and red indicate high levels.")
        self.receive_device_combo = self._device_selector(tab, 3, "Receive on", "receive_device", "output")
        self._check(tab, 4, "Packet Loss Concealment", "packet_loss_concealment",
                    "ON by default. Conceal short missing-packet gaps using the decoder's bounded repeat/mute policy. FEC stays enabled in either setting.")
        self._check(tab, 5, "ML Enhanced 48kHz Decoder", "ml_enhanced_decoder",
                    "Experimental, best-effort 48 kHz decoding for received voice and replay. Adds approximately 520 ms of buffering. OFF by default.")
        self._preview_changed()

    def _device_selector(self, tab, row, label, key, direction):
        ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=8)
        display = tk.StringVar(self)
        combo = ttk.Combobox(tab, textvariable=display, state="readonly", width=48)
        combo.grid(row=row, column=1, sticky="ew", pady=8)
        tip = ToolTip(combo, "Attach this audio device. If it is unavailable, Windows default is used; the saved choice is retained.")
        tip.suffix = display.get
        def refresh():
            selected = self.vars[key].get()
            choices = [("Windows default", "")]
            try:
                devices = enumerate_audio_devices(direction)
            except (OSError, RuntimeError):
                devices = []
            for device in devices:
                name = device.name
                if any(name == title for title, _ in choices):
                    name = f"{name} ({len(choices)})"
                choices.append((name, device.id))
            if selected and all(identity != selected for _, identity in choices):
                choices.append(("Unavailable device — Windows default", selected))
            self._device_choices[key] = choices
            combo.configure(values=[title for title, _ in choices])
            display.set(next(title for title, identity in choices if identity == selected))
            font = tkfont.Font(self, font=combo.cget("font"))
            text_width = max(font.measure(title) for title, _ in choices)
            combo.configure(width=max(48, (text_width + font.measure("0") - 1) // max(1, font.measure("0"))))
            if self.winfo_ismapped():
                self.after_idle(self._fit_device_names)
        def changed(_event):
            choices = self._device_choices[key]
            self.vars[key].set(next(identity for title, identity in choices if title == display.get()))
            if key == "transmit_device":
                self._preview_changed()
        combo.configure(postcommand=refresh)
        combo.bind("<<ComboboxSelected>>", changed)
        refresh()
        return combo

    def _fit_device_names(self):
        """Fit full endpoint names, bounded by the display; retain manual resizing."""
        if not self.winfo_exists():
            return
        self.update_idletasks()
        width = min(max(760, self.winfo_width(), self.winfo_reqwidth()),
                    self.winfo_screenwidth() - 80)
        self.geometry(f"{width}x{max(460, self.winfo_height())}")

    def _preview_changed(self, *_args):
        gain = float(self.vars["transmit_gain"].get())
        self.transmit_gain_label.configure(text=f"{gain:.1f}×")
        enabled = self.notebook.select() == str(self.sections["Audio"])
        was_active = self._preview_active
        self._preview_active = enabled
        if enabled or was_active:
            self._on_audio_preview(self.vars["transmit_device"].get(), gain, enabled)
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if enabled:
            self._poll_microphone()
        else:
            self.microphone_meter.set_level(0)

    def _poll_microphone(self):
        self.microphone_meter.set_level(self._get_microphone_level())
        self._preview_after = self.after(50, self._poll_microphone)

    def destroy(self):
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if self._preview_active:
            self._preview_active = False
            self._on_audio_preview(self.vars["transmit_device"].get(),
                                   float(self.vars["transmit_gain"].get()), False)
        super().destroy()

    def _appearance(self):
        tab = self.sections["Appearance"]
        self._check(tab, 0, "Attach avatar to incoming callsigns", "incoming_avatars",
                    "Show a small animal and muted color beside incoming messages. Each call sign keeps its assignment across restarts.")
        ttk.Label(tab, text="Each operator gets a randomly assigned animal and color.\n"
                           "Their avatar stays the same between sessions.",
                  wraplength=480, justify="left").grid(row=1, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(tab, text="Your Avatar").grid(row=2, column=0, sticky="w", padx=(0, 20), pady=8)
        AvatarPicker(tab, self.vars["own_avatar_animal"], self.vars["own_avatar_color"]).grid(
            row=2, column=1, sticky="w", pady=8)
        history = self._entry(tab, 3, "Message history limit", "message_history_limit",
                    "Keep the newest 1–500 messages (default 100). Oldest messages and their unused RFL recordings are removed first. Changes apply immediately; diagnostic logs are retained.", numeric=True)
        history.configure(to=500)

    def _logging(self, open_logs, clear_logs):
        tab = self.sections["Logging"]
        self._check(tab, 0, "Debug log", "debug", "Save interpreted events and full UDP hex captures. Older debug files rotate out at the storage limit.")
        self._entry(tab, 1, "Max debug logs (MB)", "max_debug_mb", "Total rolling budget for debug and raw hex log files. Default 100 MB (1 MB = 1,000,000 bytes). Oldest files are removed first.", numeric=True)
        self._check(tab, 2, "Save binary reflector stream for replay", "stream_logging", "Record incoming UDP payloads for expandable voice replay. Recordings expire at the storage or history limit, and are deleted on clean exit.")
        self._entry(tab, 3, "Max stream logs (MB)", "max_stream_mb", "Total rolling budget for binary stream files, separate from debug logs. Default 100 MB. Replay mutes incoming voice while reception and recording continue.", numeric=True)
        actions = ttk.Frame(tab)
        actions.grid(row=4, column=0, columnspan=2, sticky="w", pady=8)
        for label, callback, tip in (("Open logs", open_logs, "Open the folder containing debug logs and the binary streams subfolder."),
                                      ("Clear logs", clear_logs, "Clear the Advanced Stream Info display. Saved diagnostic and replay files are retained.")):
            button = ttk.Button(actions, text=label, command=callback)
            button.pack(side="left", padx=(0, 8))
            ToolTip(button, tip)

    def _grid_changed(self, *_):
        try:
            lat, lon = grid_center(self.vars["grid"].get())
        except ValueError:
            return
        self._from_grid = True
        try:
            self.vars["latitude"].set(f"{lat:.6f}")
            self.vars["longitude"].set(f"{lon:.6f}")
        finally:
            self._from_grid = False

    def _coordinates_changed(self, *_):
        if not self._from_grid:
            self.vars["grid"].set("")

    def _save(self):
        try:
            callsign = self.vars["callsign"].get().strip().upper()
            if callsign and not re.fullmatch(r"[A-Z0-9/]{3,8}", callsign):
                raise ValueError("Call sign must have 3–8 letters, digits, or '/'.")
            if not re.fullmatch(r"[A-Z]", self.vars["module"].get()):
                raise ValueError("Station module must be A–Z.")
            values = {key: self.vars[key].get() for key in Preferences.__dataclass_fields__}
            for key in ("max_debug_mb", "max_stream_mb", "message_history_limit"):
                values[key] = int(values[key])
            if values["send_dprs"] and not callsign:
                raise ValueError("Enter your call sign before enabling D-PRS.")
            preferences = Preferences(**values)
            self.on_save(preferences, callsign, self.vars["module"].get(), self.vars["debug"].get())
        except (ValueError, TypeError, OSError, tk.TclError) as exc:
            messagebox.showerror("Check settings", str(exc), parent=self)
            return
        self.destroy()
