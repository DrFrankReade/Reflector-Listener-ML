"""Windows desktop client for listening and PTT transmission on D-Star reflectors."""

from __future__ import annotations

import sys
if '--self-test-output' in sys.argv:
    from release_check import run
    run(sys.argv[sys.argv.index('--self-test-output') + 1])
    raise SystemExit(0)

import json
import math
import os
import queue
import subprocess
import sys
from pathlib import Path


from reflector_listener.bootstrap import (
    _configure_tk_runtime,
    _extended_windows_path,
    _tcl_runtime_path,
    prepare_startup,
)


APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)

# Run bootstrap before importing optional GUI/image dependencies only when this
# file is the executable entry point. Importing app never invokes pip or Tk.
if __name__ == "__main__" and not prepare_startup(APP_DIR):
    raise SystemExit(1)

import tkinter as tk
from tkinter import messagebox
from typing import Any

from PIL import ImageTk

from reflector_listener.audio import WaveInRecorder
from reflector_listener.receive_audio import ReceiveAudioPlayer
from reflector_listener.decoder import AmbeDecoder, DecoderError, DEFAULT_GAIN, GAIN_PROFILE
from reflector_listener.hosts import HostDirectory, HostDirectoryError
from reflector_listener.protocols import parse_target
from reflector_listener.rsms1 import RsMs1Receiver
from reflector_listener.config import load_buffer_config
from reflector_listener.sending import SendController, SendOperation
from reflector_listener.preferences import Preferences
from reflector_listener.location import coordinates, distance_text
from reflector_listener.stations import AvatarAssignments, station_key
from reflector_listener.ui.avatars import AvatarBadges
from reflector_listener.stream_log import BinaryStreamLog
from reflector_listener.replay import ReplayController
from reflector_listener.ui.settings_dialog import SettingsDialog
from reflector_listener.session import (
    ConnectionConfig,
    ConnectionState,
    ReflectorSession,
    SessionEvent,
)


PROJECT_DIR = APP_DIR
SETTINGS_PATH = PROJECT_DIR / "settings.json"
BUFFER_CONFIG_PATH = PROJECT_DIR / "buffers.toml"


from reflector_listener.ui.layout import LayoutMixin
from reflector_listener.ui.messages import MessagesMixin
from reflector_listener.ui.composer import ComposerMixin
from reflector_listener.ui.event_handlers import EventHandlersMixin
from reflector_listener.ui.tooltip import ToolTip


class ListenerApp(LayoutMixin, MessagesMixin, ComposerMixin, EventHandlersMixin):
    """Tkinter shell around the network, decoder, and audio workers.

    This class orchestrates the application logic by bridging user interface
    actions to the `ReflectorSession`. It manages the background event queue
    polling (`_drain_events`), manages settings persistence, and directs
    received payload events (text and pictures) into the UI components.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("D-Star Reflector Messenger")
        self.root.minsize(860, 650)
        self.root.geometry("1080x780")
        self.root.configure(background="#e8e1d6")

        self._events: queue.SimpleQueue[SessionEvent] = queue.SimpleQueue()
        self._closing = False
        self._settings = self._load_settings()
        self.preferences = Preferences.load(self._settings)
        self.avatar_badges = AvatarBadges(root, AvatarAssignments(APP_DIR / "avatars.json"),
                                          enabled=self.preferences.incoming_avatars,
                                          own_avatar=self.preferences.own_avatar)
        self._station_positions: dict[str, str] = {}
        self._settings_dialog = None
        self._directory: HostDirectory | None = None
        self._host_by_name: dict[str, str] = {}
        self._reflector_names: tuple[str, ...] = ()
        self._last_reflector_name = ""

        self.callsign_var = tk.StringVar(value=str(self._settings.get("callsign", "")))
        self.reflector_var = tk.StringVar(
            value=str(self._settings.get("reflector", "REF030"))
        )
        self.module_var = tk.StringVar(value=str(self._settings.get("module", "C")))
        self.local_module_var = tk.StringVar(
            value=str(self._settings.get("local_module", "D"))
        )
        self.host_var = tk.StringVar(value=str(self._settings.get("host", "")))
        self.gain_var = tk.DoubleVar(value=self._setting_gain(DEFAULT_GAIN))
        self.status_var = tk.StringVar(value="Disconnected")
        self.endpoint_var = tk.StringVar(value="")
        self.chat_destination_var = tk.StringVar(
            value=str(self._settings.get("chat_destination", "CQCQCQ"))
        )
        self.message_var = tk.StringVar()
        self.message_var.trace_add("write", self._on_message_changed)
        self.picture_size_var = tk.StringVar(
            value=str(self._settings.get("picture_size", "320x240"))
        )
        self.picture_quality_var = tk.IntVar(
            value=int(self._settings.get("picture_quality", 50))
        )
        self.picture_crop_var = tk.StringVar(
            value=str(self._settings.get("picture_crop", "Fit / letterbox"))
        )
        self.picture_number_var = tk.IntVar(
            value=int(self._settings.get("picture_number", 1))
        )
        self.picture_caption_var = tk.StringVar()
        self.picture_path_var = tk.StringVar(value="No picture selected")
        self.app_tx_status_var = tk.StringVar(value="Ready")
        self.debug_packet_log_var = tk.BooleanVar(value=self._settings.get("debug_packet_log") is True)
        self._picture_path: Path | None = None
        self._app_send_active = False
        self._picture_controls_visible = False
        self._saved_tx_pictures: dict[str, dict[str, Any]] = {}
        self._picture_tooltips: dict[str, ToolTip] = {}
        self._current_rx_callsign: str | None = None
        self._current_rx_mode: str | None = None
        self._current_rx_frames: int = 0
        self._voice_bubbles: dict[tuple, dict[str, Any]] = {}
        self._active_voice_bubble = None
        self._active_voice_segment = None
        self._current_rx_pic_tiles: str = ""
        self._current_tx_mode: str | None = None
        self._current_tx_frames: int = 0
        self._current_tx_pic_total: int = 300
        self._last_call_summary: str | None = None
        self._picture_bubbles: dict[str, dict[str, Any]] = {}
        self._picture_preview_photo: ImageTk.PhotoImage | None = None
        self._outgoing_counter = 0
        self._active_outgoing_id: str | None = None

        try:
            self.decoder = AmbeDecoder(gain=self.gain_var.get())
        except (DecoderError, OSError) as exc:
            messagebox.showerror("AMBE decoder", str(exc), parent=self.root)
            raise

        self.buffer_config = load_buffer_config(BUFFER_CONFIG_PATH)
        self.audio = ReceiveAudioPlayer(
            on_error=self._on_audio_error, on_warning=self._on_ml_warning, config=self.buffer_config
        )
        self.audio.set_ml_enabled(self.preferences.ml_enhanced_decoder)
        self._app_receiver = RsMs1Receiver()
        self.stream_log = BinaryStreamLog(APP_DIR / "logs" / "streams")
        self.replay = ReplayController(self.stream_log, on_playing=self.audio.set_muted,
                                       on_warning=self._on_ml_warning)
        self.replay.set_ml_enabled(self.preferences.ml_enhanced_decoder)
        self.replay.set_gain(self.gain_var.get())
        self.session = ReflectorSession(
            self.decoder,
            self.audio,
            self._events.put,
            config=self.buffer_config,
            app_receiver=self._app_receiver,
            stream_log=self.stream_log,
        )
        self._send_controller = SendController(
            self.session.transmit_application, self._on_send_finished
        )
        self._send_operation: SendOperation | None = None
        self.microphone = WaveInRecorder(
            self._transmit_pcm, on_error=self._on_microphone_error
        )
        self._ptt_held = False
        self._ptt_timed_out = False

        self._load_host_directory()
        self._build_ui()
        self.session.set_preferences(self.preferences)
        self.session.log_directory = APP_DIR / "logs"
        self.session.packet_logger.set_max_bytes(self.preferences.max_debug_mb * 1_000_000)
        try:
            self.stream_log.configure(self.preferences.stream_logging, self.preferences.max_stream_mb * 1_000_000)
        except OSError as exc:
            self._append_log("ERROR", f"Cannot start binary recording: {exc}", "error")
        if self.debug_packet_log_var.get():
            self._on_toggle_packet_log()
        self._apply_initial_host()
        self._append_log(
            "READY",
            f"AMBE decoder ready: {self.decoder.profile}",
            "decoder",
        )
        if self._directory is not None:
            self._append_log(
                "HOSTS",
                f"Loaded {len(self._directory):,} entries from {self._directory.source}",
                "network",
            )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._drain_events)


    def _load_host_directory(self) -> None:
        try:
            self._directory = HostDirectory.load()
        except (FileNotFoundError, HostDirectoryError) as exc:
            self._directory_error = str(exc)
            return

        supported: list[str] = []
        for entry in self._directory:
            try:
                parse_target(entry.name, "C")
            except ValueError:
                continue
            supported.append(entry.name)
            self._host_by_name[entry.name] = entry.ipv4
        self._reflector_names = tuple(sorted(set(supported)))

    def _apply_initial_host(self) -> None:
        name = self.reflector_var.get().strip().upper()
        self.reflector_var.set(name)
        configured_reflector = str(self._settings.get("reflector", "")).upper()
        configured_host = str(self._settings.get("host", "")).strip()
        if not configured_host or configured_reflector != name:
            self.host_var.set(self._host_by_name.get(name, configured_host))
        self._last_reflector_name = name
        self._refresh_endpoint_label()
        if self._directory is None:
            self._append_log(
                "WARNING",
                getattr(self, "_directory_error", "Reflector host directory unavailable"),
                "warning",
            )

    def _on_reflector_selected(self, _event: tk.Event[Any] | None = None) -> None:
        name = self.reflector_var.get().strip().upper()
        self.reflector_var.set(name)
        if name in self._host_by_name:
            self.host_var.set(self._host_by_name[name])
        self._last_reflector_name = name
        self._refresh_endpoint_label()

    def _on_reflector_focus_out(self, _event: tk.Event[Any] | None = None) -> None:
        name = self.reflector_var.get().strip().upper()
        self.reflector_var.set(name)
        # Typing into an editable combobox does not emit
        # <<ComboboxSelected>>.  Replace the old directory-derived endpoint
        # when the reflector changed, while preserving a deliberate host
        # override when focus simply returns to the same reflector.
        if name != self._last_reflector_name:
            # Never carry a previous reflector's endpoint into a newly typed
            # unknown name.  The user can then enter that reflector's host.
            self.host_var.set(self._host_by_name.get(name, ""))
        elif name in self._host_by_name and not self.host_var.get().strip():
            self.host_var.set(self._host_by_name[name])
        self._last_reflector_name = name
        self._refresh_endpoint_label()

    def _refresh_endpoint_label(self, _event: tk.Event[Any] | None = None) -> None:
        try:
            target = parse_target(self.reflector_var.get(), self.module_var.get())
        except (TypeError, ValueError):
            self.endpoint_var.set("Enter REFnnn, XRFnnn, DCSnnn, or XLXnnn")
            return
        module_note = (
            " • D-Plus login is reflector-wide; the module is matched from stream headers"
            if target.protocol.value == "DPlus"
            else ""
        )
        self.endpoint_var.set(
            f"{target.protocol.value} UDP {target.port}{module_note}"
        )

    def _connect(self) -> None:
        try:
            name = self.reflector_var.get().strip().upper()
            if name != self._last_reflector_name:
                self.host_var.set(self._host_by_name.get(name, ""))
                self._last_reflector_name = name
            elif not self.host_var.get().strip() and name in self._host_by_name:
                self.host_var.set(self._host_by_name[name])
            config = ConnectionConfig(
                callsign=self.callsign_var.get(),
                reflector=name,
                module=self.module_var.get(),
                host=self.host_var.get(),
                local_module=self.local_module_var.get(),
                gain=self.gain_var.get(),
            ).normalized()
            self.session.connect(config)
        except Exception as exc:
            messagebox.showerror("Cannot connect", str(exc), parent=self.root)
            return
        self._save_settings()
        self._set_controls_for_state(ConnectionState.CONNECTING)

    def _disconnect(self) -> None:
        self._send_controller.cancel()
        self.disconnect_button.configure(state="disabled")
        self.root.update_idletasks()
        try:
            self.microphone.stop()
            self._ptt_held = False
            self._ptt_timed_out = False
            self.session.disconnect()
        except Exception as exc:
            self._append_log("ERROR", str(exc), "error")

    def _on_gain(self, _value: str) -> None:
        self._update_gain_label()
        if hasattr(self, "replay"):
            self.replay.set_gain(self.gain_var.get())
        try:
            self.session.set_gain(self.gain_var.get())
        except (DecoderError, ValueError) as exc:
            self._append_log("ERROR", str(exc), "error")


    def _on_audio_error(self, message: str) -> None:
        self._events.put(SessionEvent("error", f"Audio output failed: {message}"))

    def _on_ml_warning(self, message: str) -> None:
        self._events.put(SessionEvent("warning", message))

    def _on_microphone_error(self, message: str) -> None:
        self._events.put(SessionEvent("error", f"Microphone input failed: {message}"))
        self.session.stop_transmit(reason="microphone error")

    def _transmit_pcm(self, pcm: bytes) -> None:
        if not self.session.transmit_pcm(pcm):
            self._events.put(SessionEvent("tx_capture_stop", "Microphone capture stopped"))

    def _ptt_press(self, _event: tk.Event[Any] | None = None) -> str:
        if self._app_send_active:
            return "break"
        if self._receive_blocks_transmit():
            return "break"
        if self._ptt_held or self._ptt_timed_out:
            return "break"
        try:
            self.session.start_transmit()
            self.microphone.start()
        except Exception as exc:
            self.session.stop_transmit(reason="microphone start failed")
            messagebox.showerror("Cannot transmit", str(exc), parent=self.root)
            return "break"
        self._ptt_held = True
        self.ptt_button.configure(text="TRANSMITTING — release to stop")
        return "break"

    def _ptt_release(self, _event: tk.Event[Any] | None = None) -> str:
        was_active = self._ptt_held or self._ptt_timed_out
        self._ptt_held = False
        self._ptt_timed_out = False
        if was_active:
            self.microphone.stop()
            self.session.stop_transmit()
            if hasattr(self, "ptt_button"):
                self.ptt_button.configure(text="Hold to Talk (PTT)")
        return "break" if was_active else ""


    def _append_log(self, prefix: str, message: str, tag: str) -> None:
        if not hasattr(self, "log"):
            return
        self.log.configure(state="normal")
        self.log.insert("end", f"[{prefix}] {message}\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _load_settings(self) -> dict[str, Any]:
        try:
            value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_settings(self) -> None:
        document = {
            "callsign": self.callsign_var.get().strip().upper(),
            "reflector": self.reflector_var.get().strip().upper(),
            "module": self.module_var.get().strip().upper(),
            "local_module": (
                self.local_module_var.get().strip().upper()
                if hasattr(self, "local_module_var")
                else "D"
            ),
            "host": self.host_var.get().strip(),
            "gain": round(self.gain_var.get(), 2),
            "gain_profile": GAIN_PROFILE,
            "chat_destination": self.chat_destination_var.get().strip().upper(),
            "picture_size": self.picture_size_var.get(),
            "picture_quality": int(self.picture_quality_var.get()),
            "picture_crop": self.picture_crop_var.get(),
            "picture_number": int(self.picture_number_var.get()) & 0xFF,
            "debug_packet_log": (
                bool(self.debug_packet_log_var.get())
                if hasattr(self, "debug_packet_log_var")
                else False
            ),
        }
        temporary = SETTINGS_PATH.with_suffix(".json.tmp")
        if hasattr(self, "preferences"):
            document.update(self.preferences.saved())
        try:
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            temporary.replace(SETTINGS_PATH)
        except OSError as exc:
            self._append_log("WARNING", f"Could not save settings: {exc}", "warning")

    def _on_toggle_packet_log(self) -> None:
        enabled = self.debug_packet_log_var.get()
        if enabled:
            try:
                log_path = self.session.enable_packet_logging()
                self._append_log("DEBUG", f"Packet debug logging started: {log_path}", "state")
                self._append_log("DEBUG", f"Raw UDP receive logging started: {self.session.raw_packet_log_path}", "state")
            except Exception as exc:
                self.debug_packet_log_var.set(self.session.is_packet_logging_enabled())
                self._append_log("WARNING", f"Could not start packet logging: {exc}", "warning")
        else:
            raw_path = self.session.raw_packet_log_path
            saved_path = self.session.disable_packet_logging()
            if saved_path:
                self._append_log("DEBUG", f"Packet debug logging stopped: {saved_path}", "state")
            if raw_path:
                self._append_log("DEBUG", f"Raw UDP receive logging stopped: {raw_path}", "state")
        self._save_settings()

    def _open_settings(self) -> None:
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            return
        self._settings_dialog = SettingsDialog(
            self.root, self.preferences, callsign=self.callsign_var.get(),
            module=self.local_module_var.get(), debug=self.debug_packet_log_var.get(),
            connected=self.session.state not in (ConnectionState.DISCONNECTED, ConnectionState.ERROR),
            on_save=self._apply_settings, open_logs=self._open_logs_folder, clear_logs=self._clear_log,
        )

    def _apply_settings(self, preferences: Preferences, callsign: str, module: str, debug: bool) -> None:
        old = self.preferences
        if (preferences.stream_logging, preferences.max_stream_mb) != (old.stream_logging, old.max_stream_mb):
            self.stream_log.configure(preferences.stream_logging, preferences.max_stream_mb * 1_000_000)
        self.session.packet_logger.set_max_bytes(preferences.max_debug_mb * 1_000_000)
        self.callsign_var.set(callsign)
        self.local_module_var.set(module)
        self.preferences = preferences
        if preferences.ml_enhanced_decoder != old.ml_enhanced_decoder:
            self.audio.set_ml_enabled(preferences.ml_enhanced_decoder)
            self.replay.set_ml_enabled(preferences.ml_enhanced_decoder)
        if preferences.message_history_limit != old.message_history_limit:
            self._trim_message_history()
            self.root.after_idle(self._scroll_chat_to_end)
        if preferences.incoming_avatars != old.incoming_avatars:
            self.avatar_badges.set_enabled(preferences.incoming_avatars)
        if preferences.own_avatar != old.own_avatar:
            self.avatar_badges.set_own_avatar(preferences.own_avatar)
        self.session.set_preferences(preferences)
        if debug != self.session.is_packet_logging_enabled():
            self.debug_packet_log_var.set(debug)
            self._on_toggle_packet_log()
        else:
            self._save_settings()

    _station_key = staticmethod(station_key)

    def _distance_tooltip(self, callsign: str) -> str:
        preferences = getattr(self, "preferences", Preferences())
        remote = getattr(self, "_station_positions", {}).get(self._station_key(callsign), "")
        return distance_text(preferences.position, remote, preferences.units)

    def _remember_position(self, callsign: str, text: str) -> None:
        key = self._station_key(callsign)
        if not key or key == "UNKNOWN":
            return
        if not hasattr(self, "_station_positions"):
            self._station_positions = {}
        try:
            position = coordinates(*text.replace(",", " ").split())
        except (ValueError, TypeError):
            self._station_positions.pop(key, None)
            return
        if position == (0.0, 0.0):
            self._station_positions.pop(key, None)
        else:
            self._station_positions[key] = text

    def _open_logs_folder(self) -> None:
        logs_dir = APP_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(logs_dir)
        else:
            try:
                subprocess.Popen(["xdg-open", str(logs_dir)])
            except Exception:
                pass

    def _setting_gain(self, default: float) -> float:
        try:
            value = float(self._settings.get("gain", default))
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value) or not 0.0 <= value <= 6.0:
            return default
        # A Soft-AMBE listening level is not calibrated to mbelib-neo PCM16.
        # Start at unity once on migration, but preserve an explicit mute.
        if value != 0.0 and self._settings.get("gain_profile") != GAIN_PROFILE:
            return default
        return value

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._send_controller.cancel()
        self._save_settings()
        if self.session.is_packet_logging_enabled():
            self.session.disable_packet_logging()
        try:
            self.microphone.stop()
            self.session.disconnect()
        finally:
            self.replay.close()
            self.stream_log.purge()
            self.audio.close()
            self.decoder.close()
            self.root.destroy()


def main() -> int:
    _configure_tk_runtime()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Unable to start the Tk desktop interface: {exc}", file=sys.stderr)
        return 1
    try:
        ListenerApp(root)
    except Exception as exc:
        print(f"Unable to initialize the reflector listener: {exc}", file=sys.stderr)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
