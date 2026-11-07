"""Layout view methods for the application controller."""

from __future__ import annotations

import time
from typing import Any
import tkinter as tk
from tkinter import scrolledtext, ttk
from reflector_listener.session import ConnectionState
from reflector_listener.rsms1 import CROP_OPTIONS, PICTURE_QUALITY_OPTIONS, PICTURE_SIZE_OPTIONS
from .tooltip import ToolTip
from .assets import get_add_photo_icon
from .timeline import MessageTimeline
from .vu_meter import VuMeter


class LayoutMixin:
    """UI concern shared by ListenerApp; all methods run on the Tk thread."""

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        connection = ttk.LabelFrame(outer, text="Connection", padding=12)
        connection.grid(row=0, column=0, sticky="ew")
        connection.columnconfigure(5, weight=1)

        ttk.Label(connection, text="Reflector").grid(row=0, column=2, sticky="w")
        self.reflector_combo = ttk.Combobox(
            connection,
            width=12,
            textvariable=self.reflector_var,
            values=self._reflector_names,
        )
        self.reflector_combo.grid(row=1, column=2, padx=(0, 12), sticky="ew")
        self.reflector_combo.bind("<<ComboboxSelected>>", self._on_reflector_selected)
        self.reflector_combo.bind("<FocusOut>", self._on_reflector_focus_out)

        ttk.Label(connection, text="Reflector module").grid(row=0, column=3, sticky="w", padx=(0, 12))
        self.module_combo = ttk.Combobox(
            connection,
            width=5,
            state="readonly",
            textvariable=self.module_var,
            values=tuple(chr(value) for value in range(ord("A"), ord("Z") + 1)),
        )
        self.module_combo.grid(row=1, column=3, padx=(0, 12), sticky="ew")
        self.module_combo.bind("<<ComboboxSelected>>", self._refresh_endpoint_label)

        ttk.Label(connection, text="Host / IPv4").grid(row=0, column=4, sticky="w")
        self.host_entry = ttk.Entry(connection, textvariable=self.host_var, width=16)
        self.host_entry.grid(row=1, column=4, padx=(0, 12), sticky="w")

        button_box = ttk.Frame(connection)
        button_box.grid(row=1, column=6, sticky="e", padx=(12, 0))
        self.connect_button = ttk.Button(
            button_box, text="Connect", command=self._connect
        )
        self.connect_button.grid(row=0, column=0, padx=(0, 6))
        self.disconnect_button = ttk.Button(
            button_box, text="Disconnect", command=self._disconnect, state="disabled"
        )
        self.disconnect_button.grid(row=0, column=1)

        self.status_banner = tk.Label(
            connection,
            text="Not linked",
            font=("Segoe UI", 10, "bold"),
            foreground="#000000",
            background="#e0e0e0",
            relief="solid",
            borderwidth=1,
            anchor="w",
            padx=10,
            pady=4,
            width=24,
        )
        self.status_banner.grid(row=1, column=5, sticky="ew")
        # Keep the full last-call summary readable at the minimum window width.
        self.status_banner.bind("<Configure>", lambda event: self.status_banner.configure(
            wraplength=max(80, event.width - 22)))

        controls_bar = ttk.Frame(outer, padding=(2, 8, 2, 8))
        controls_bar.grid(row=1, column=0, sticky="ew")
        controls_bar.columnconfigure(3, weight=1)

        ttk.Label(controls_bar, text="Audio Gain").grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )
        self.gain_scale = ttk.Scale(
            controls_bar,
            from_=0.0,
            to=6.0,
            variable=self.gain_var,
            command=self._on_gain,
            length=150,
        )
        self.gain_scale.grid(row=0, column=1, sticky="w")
        self.gain_label = ttk.Label(controls_bar, width=4, anchor="e")
        self.gain_label.grid(row=0, column=2, padx=(5, 12), sticky="w")
        self.output_meter = VuMeter(controls_bar)
        self.output_meter.grid(row=0, column=3, sticky="ew", padx=(0, 12))
        ToolTip(self.output_meter, "Live received or replay audio level. During PTT, shows your microphone level.")

        ptt_style = ttk.Style()
        ptt_style.configure(
            "BigPTT.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(14, 9),
        )
        self.ptt_button = ttk.Button(
            controls_bar,
            text="Hold to Talk (PTT)",
            style="BigPTT.TButton",
            state="disabled",
            width=30,
        )
        self.ptt_button.grid(row=0, column=4, sticky="e")
        self.ptt_button.bind("<ButtonPress-1>", self._ptt_press)
        self.ptt_button.bind("<ButtonRelease-1>", self._ptt_release)
        self.root.bind_all("<ButtonRelease-1>", self._ptt_release, add="+")
        self._update_gain_label()

        notebook = ttk.Notebook(outer)
        notebook.grid(row=2, column=0, sticky="nsew")
        messages_tab = tk.Frame(notebook, background="#f3eee6")
        log_frame = ttk.Frame(notebook, padding=8)
        notebook.add(messages_tab, text=" Messages ")
        notebook.add(log_frame, text=" Advanced Stream Info ")
        self._build_messages_tab(messages_tab)

        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            height=18,
            font=("Consolas", 10),
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.tag_configure("stream_start", foreground="#086a22")
        self.log.tag_configure("stream_recovery", foreground="#8a5300")
        self.log.tag_configure("error", foreground="#a31515")
        self.log.tag_configure("warning", foreground="#8a5300")
        self.log.tag_configure("state", foreground="#174a8b")
        self.log.tag_configure("decoder", foreground="#5b3f8c")
        self.log.tag_configure("network", foreground="#555555")

        menu_bar = tk.Menu(self.root)
        application_menu = tk.Menu(menu_bar, tearoff=False)
        application_menu.add_command(label="Settings", command=self._open_settings)
        menu_bar.add_cascade(label="File", menu=application_menu)
        self.root.configure(menu=menu_bar)
        self.root.bind("<Control-comma>", lambda _e: self._open_settings())
        ToolTip(self.reflector_combo, "Choose or enter the reflector to connect to.")
        ToolTip(self.module_combo, "Choose the module on the selected reflector.")
        ToolTip(self.host_entry, "Reflector hostname or IPv4 address; filled from the directory when available.")
        ToolTip(self.connect_button, "Connect using the call sign and station module saved in Settings.")
        ToolTip(self.disconnect_button, "Disconnect from the reflector.")
        ToolTip(self.gain_scale, "Adjust live reception and replay volume.")
        ToolTip(self.ptt_button, "Hold to transmit microphone audio. Release to stop.")


    def _build_messages_tab(self, parent: tk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        timeline = tk.Frame(parent, background="#f3eee6")
        timeline.grid(row=0, column=0, sticky="nsew")
        timeline.rowconfigure(0, weight=1)
        timeline.columnconfigure(0, weight=1)
        self.chat_canvas = MessageTimeline(
            timeline,
            background="#f3eee6",
            highlightthickness=0,
            borderwidth=0,
        )
        chat_scroll = ttk.Scrollbar(
            timeline, orient="vertical", command=self.chat_canvas.yview
        )
        self.chat_canvas.set_scrollbar(chat_scroll.set)
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_body = self.chat_canvas
        self.chat_canvas.bind("<MouseWheel>", self._on_chat_mousewheel)
        self.chat_canvas.bind(
            "<Enter>",
            lambda _e: self.chat_canvas.bind_all("<MouseWheel>", self._on_chat_mousewheel),
        )
        self.chat_canvas.bind(
            "<Leave>",
            lambda _e: self.chat_canvas.unbind_all("<MouseWheel>"),
        )

        welcome = tk.Frame(self.chat_body, background="#f3eee6", padx=24, pady=28)
        self.chat_canvas.add_row(welcome)
        welcome.bind("<MouseWheel>", self._on_chat_mousewheel)
        w_title = tk.Label(
            welcome,
            text="REFLECTOR MESSAGES",
            background="#f3eee6",
            foreground="#9a4d2e",
            font=("Bahnschrift SemiBold", 10),
        )
        w_title.pack()
        w_title.bind("<MouseWheel>", self._on_chat_mousewheel)
        w_sub = tk.Label(
            welcome,
            text="Text and pictures appear here as RS-MS1 data arrives.",
            background="#f3eee6",
            foreground="#5d5a55",
            font=("Segoe UI", 10),
        )
        w_sub.pack(pady=(4, 0))
        w_sub.bind("<MouseWheel>", self._on_chat_mousewheel)

        composer = tk.Frame(
            parent,
            background="#fffdf9",
            highlightbackground="#d5ccc0",
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        composer.grid(row=1, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)

        # Collapsible picture controls section (Req 14)
        self.picture_section_frame = tk.Frame(composer, background="#fffdf9")
        self.picture_section_frame.grid(row=0, column=0, sticky="ew")
        self.picture_section_frame.columnconfigure(7, weight=1)

        tk.Label(
            self.picture_section_frame,
            text="TO",
            background="#fffdf9",
            foreground="#726c64",
            font=("Bahnschrift SemiBold", 9),
        ).grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.chat_destination_entry = ttk.Entry(
            self.picture_section_frame, width=10, textvariable=self.chat_destination_var
        )
        self.chat_destination_entry.grid(row=0, column=1, padx=(0, 12), sticky="w")
        ttk.Label(self.picture_section_frame, text="Size").grid(row=0, column=2, padx=(0, 4))
        self.picture_size_combo = ttk.Combobox(
            self.picture_section_frame,
            width=9,
            state="readonly",
            values=PICTURE_SIZE_OPTIONS,
            textvariable=self.picture_size_var,
        )
        self.picture_size_combo.grid(row=0, column=3, padx=(0, 10))
        ttk.Label(self.picture_section_frame, text="Quality").grid(row=0, column=4, padx=(0, 4))
        self.picture_quality_combo = ttk.Combobox(
            self.picture_section_frame,
            width=5,
            state="readonly",
            values=PICTURE_QUALITY_OPTIONS,
            textvariable=self.picture_quality_var,
        )
        self.picture_quality_combo.grid(row=0, column=5, padx=(0, 10))
        ttk.Label(self.picture_section_frame, text="Crop").grid(row=0, column=6, padx=(0, 4))
        self.picture_crop_combo = ttk.Combobox(
            self.picture_section_frame,
            state="readonly",
            values=CROP_OPTIONS,
            textvariable=self.picture_crop_var,
        )
        self.picture_crop_combo.grid(row=0, column=7, padx=(0, 10), sticky="ew")
        ttk.Label(self.picture_section_frame, text="Picture #").grid(row=0, column=8, padx=(0, 4))
        self.picture_number_spin = ttk.Spinbox(
            self.picture_section_frame,
            from_=0,
            to=255,
            width=5,
            textvariable=self.picture_number_var,
        )
        self.picture_number_spin.grid(row=0, column=9)
        for combo in (
            self.picture_size_combo,
            self.picture_quality_combo,
            self.picture_crop_combo,
        ):
            combo.bind("<<ComboboxSelected>>", self._refresh_picture_preview)

        picture_row = tk.Frame(self.picture_section_frame, background="#fffdf9")
        picture_row.grid(row=1, column=0, columnspan=10, pady=(9, 7), sticky="ew")
        picture_row.columnconfigure(1, weight=1)
        self.choose_picture_button = ttk.Button(
            picture_row, text="Choose picture", command=self._choose_picture
        )
        self.choose_picture_button.grid(row=0, column=0, padx=(0, 8))
        tk.Label(
            picture_row,
            textvariable=self.picture_path_var,
            background="#fffdf9",
            foreground="#6d675f",
            anchor="w",
        ).grid(row=0, column=1, padx=(0, 10), sticky="w")

        self.picture_preview_frame = tk.Frame(
            self.picture_section_frame,
            background="#f3eee6",
            highlightbackground="#d5ccc0",
            highlightthickness=1,
            padx=8,
            pady=8,
        )
        self.picture_preview_frame.grid(
            row=2, column=0, columnspan=10, pady=(0, 8), sticky="ew"
        )
        self.picture_preview_frame.columnconfigure(1, weight=1)
        self.picture_preview_label = tk.Label(
            self.picture_preview_frame,
            background="#f3eee6",
            borderwidth=0,
        )
        self.picture_preview_label.grid(row=0, column=0, padx=(0, 10))
        self.picture_preview_details = tk.Label(
            self.picture_preview_frame,
            text="",
            background="#f3eee6",
            foreground="#5f5951",
            justify="left",
            anchor="w",
            font=("Segoe UI", 9),
        )
        self.picture_preview_details.grid(row=0, column=1, sticky="ew")
        self.picture_preview_frame.grid_remove()

        # Hidden by default (Req 14)
        self.picture_section_frame.grid_remove()

        # Modern message row with Add Photo toggle, entry with context menu, counter, and Send/Cancel
        message_row = tk.Frame(composer, background="#fffdf9")
        message_row.grid(row=1, column=0, pady=(4, 0), sticky="ew")
        message_row.columnconfigure(1, weight=1)

        self._add_photo_icon = get_add_photo_icon(20)
        self.add_photo_btn = tk.Button(
            message_row,
            image=self._add_photo_icon,
            width=32,
            height=26,
            relief="groove",
            background="#f5f5f5",
            activebackground="#e0f2f1",
            cursor="hand2",
            command=self._toggle_picture_controls,
        )
        self.add_photo_btn.grid(row=0, column=0, padx=(0, 6))
        self.mountain_btn = self.add_photo_btn  # backward compatibility alias
        ToolTip(self.add_photo_btn, "Attach picture / Photo controls")

        self.message_entry = ttk.Entry(
            message_row,
            textvariable=self.message_var,
            font=("Segoe UI Emoji", 11),
        )
        self.message_entry.grid(row=0, column=1, padx=(0, 6), sticky="ew")
        self.message_entry.bind("<Return>", self._send_from_key)

        entry_menu = tk.Menu(self.root, tearoff=0)
        entry_menu.add_command(
            label="Cut",
            command=lambda: self.message_entry.event_generate("<<Cut>>"),
        )
        entry_menu.add_command(
            label="Copy",
            command=lambda: self.message_entry.event_generate("<<Copy>>"),
        )
        entry_menu.add_command(
            label="Paste",
            command=lambda: self.message_entry.event_generate("<<Paste>>"),
        )
        entry_menu.add_separator()
        entry_menu.add_command(
            label="Select All",
            command=lambda: self.message_entry.select_range(0, "end"),
        )
        self.message_entry.bind("<Button-3>", lambda e: entry_menu.tk_popup(e.x_root, e.y_root))
        self._pending_high_surrogate: str | None = None
        self.message_entry.bind("<KeyPress>", self._on_entry_keypress, add="+")
        self._setup_windows_emoji_input()

        self.char_counter_label = tk.Label(
            message_row,
            text="0/128",
            font=("Segoe UI", 9),
            foreground="#756f67",
            background="#fffdf9",
            width=6,
            anchor="e",
        )
        self.char_counter_label.grid(row=0, column=2, padx=(0, 8))

        self.send_button = ttk.Button(
            message_row,
            text="Send",
            command=self._on_send,
            state="disabled",
        )
        self.send_button.grid(row=0, column=3)

        self.cancel_button = tk.Button(
            message_row,
            text="Cancel",
            command=self._cancel_send,
            bg="#ffcdd2",
            fg="#b71c1c",
            activebackground="#ef9a9a",
            activeforeground="#b71c1c",
            relief="groove",
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.cancel_button.grid(row=0, column=3)
        self.cancel_button.grid_remove()


    def _update_gain_label(self) -> None:
        if hasattr(self, "gain_label"):
            self.gain_label.configure(text=f"{self.gain_var.get():.1f}×")


    def _scroll_chat_to_end(self) -> None:
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)


    def _on_chat_mousewheel(self, event: tk.Event[Any]) -> str:
        if hasattr(event, "delta") and event.delta:
            self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"


    def _toggle_picture_controls(self) -> None:
        self._picture_controls_visible = not self._picture_controls_visible
        btn = getattr(self, "add_photo_btn", getattr(self, "mountain_btn", None))
        if self._picture_controls_visible:
            self.picture_section_frame.grid()
            if btn is not None:
                btn.configure(relief="sunken", background="#d0e8e6")
        else:
            self.picture_section_frame.grid_remove()
            if btn is not None:
                btn.configure(relief="groove", background="#f5f5f5")


    def _set_rxtx_listening(self, last_call: str | None = None, mode: str | None = None,
                            details: str | None = None, when: str | None = None) -> None:
        if last_call:
            self._last_call_summary = f"Last Call {last_call.strip().upper()} {when or time.strftime('%H:%M:%S')}"
        self._status_activity = "Listening"
        self._render_status()

    def _set_rxtx_receiving(self, callsign: str, mode: str, details: str | int = "") -> None:
        self._status_activity = f"Receiving {(callsign or 'Unknown').strip().upper()}"
        self._render_status()

    def _set_rxtx_transmitting(self, callsign: str, mode: str, details: str | int = "") -> None:
        self._status_activity = f"Transmitting {(callsign or 'MYCALL').strip().upper()}"
        self._render_status()

    def _update_status_banner(self, state: ConnectionState, message: str = "") -> None:
        self._status_connection_state = state
        self._status_message = message
        if state is not ConnectionState.LINKED:
            self._status_activity = "Listening"
        self._render_status()

    def _render_status(self) -> None:
        if not hasattr(self, "status_banner"):
            return
        state = getattr(self, "_status_connection_state", ConnectionState.DISCONNECTED)
        ref = self.reflector_var.get().strip().upper()
        mod = self.module_var.get().strip().upper()
        background = "#e0e0e0"
        if state is ConnectionState.LINKED:
            text = f"Linked {ref}{mod} - {getattr(self, '_status_activity', 'Listening')}"
            if getattr(self, "_last_call_summary", None):
                text += f" - {self._last_call_summary}"
            background = "#c8e6c9"
        elif state is ConnectionState.CONNECTING:
            text = f"Connecting {ref}{mod}..."
        elif state in (ConnectionState.DISCONNECTED, ConnectionState.STOPPING):
            text = "Not linked"
        elif state is ConnectionState.ERROR:
            text = f"Error: {getattr(self, '_status_message', '') or state.value}"
            background = "#ffcdd2"
        else:
            text = getattr(self, "_status_message", "") or state.value
        self.status_banner.configure(text=text, background=background, foreground="#000000")

    def _set_status_error(self, message: str) -> None:
        self._update_status_banner(ConnectionState.ERROR, message)


    def _set_controls_for_state(self, state: ConnectionState) -> None:
        active = state in {
            ConnectionState.CONNECTING,
            ConnectionState.LINKED,
            ConnectionState.STOPPING,
        }
        self.connect_button.configure(state="disabled" if active else "normal")
        self.disconnect_button.configure(state="normal" if active else "disabled")
        entry_state = "disabled" if active else "normal"
        self.reflector_combo.configure(state=entry_state)
        self.host_entry.configure(state=entry_state)
        self.module_combo.configure(state="disabled" if active else "readonly")
        self._update_status_banner(state)
        self._set_app_send_active(self._app_send_active)
        if state is not ConnectionState.LINKED:
            self.microphone.stop()
            self._ptt_held = False
            self._ptt_timed_out = False
            self.ptt_button.configure(text="Hold to Talk (PTT)")

