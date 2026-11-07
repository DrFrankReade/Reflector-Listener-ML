"""Event handlers view methods for the application controller."""

from __future__ import annotations

from functools import cached_property
import time
from typing import Callable
from tkinter import messagebox
from PIL import Image
from reflector_listener.session import ConnectionState, SessionEvent
from reflector_listener.events import EventKind, drain_events
from reflector_listener.sending import SendResult
from reflector_listener.stream_log import RecordingGroup
from reflector_listener.stations import station_key
from .playback import PlaybackControl, timecode


class EventHandlersMixin:
    """UI concern shared by ListenerApp; all methods run on the Tk thread."""

    def _handle_event_state(self, event: SessionEvent, stamp: str) -> None:
        state_value = event.fields.get("state")
        try:
            state = ConnectionState(str(state_value))
        except ValueError:
            state = self.session.state
        if state is not ConnectionState.LINKED:
            self._send_controller.cancel()
        if state is ConnectionState.CONNECTING:
            # Stream IDs belong to the current reflector connection.
            self._voice_bubbles = {}
        self.status_var.set(event.message)
        self._update_status_banner(state, event.message)
        self._set_controls_for_state(state)
        if state is not ConnectionState.LINKED:
            self._set_rxtx_listening()


    def _handle_event_error(self, event: SessionEvent, stamp: str) -> None:
        self._set_status_error(event.message)


    @staticmethod
    def _voice_message_key(callsign, stream_id, protocol):
        # The four-character radio suffix is station metadata, not a call sign.
        call = station_key(callsign)
        if stream_id is None or call in {"", "UNKNOWN"}:
            return None
        return protocol, int(stream_id), call

    def _bind_voice_recordings(self, bubble) -> None:
        clips = tuple(part["recording"] for part in bubble["segments"] if part["recording"] is not None)
        if not clips:
            return
        clip = clips[0] if len(clips) == 1 else RecordingGroup(clips)
        playback = bubble.get("playback")
        if playback:
            playback.set_clip(clip)
        else:
            bubble["playback"] = PlaybackControl(bubble["bubble"], bubble["header"], clip, self.replay)

    def _resolve_voice_message(self, callsign) -> None:
        """Join a provisional UNKNOWN row when a later header identifies it."""
        active = getattr(self, "_active_voice_bubble", None)
        if not active or "segments" not in active:
            return
        key = self._voice_message_key(callsign, active["stream_id"], active["protocol"])
        if key is None:
            return
        existing = self._voice_bubbles.get(key)
        if existing is not None and existing is not active:
            existing["segments"].extend(active["segments"])
            if active.get("station_message"):
                existing["station_message"] = active["station_message"]
            if active.get("coordinates"):
                existing["coordinates"] = active["coordinates"]
            was_open = bool(active.get("playback") and active["playback"].open)
            self._bind_voice_recordings(existing)
            if was_open and existing.get("playback") and not existing["playback"].open:
                existing["playback"]._toggle()
            self._active_voice_bubble = existing
            active["row"].destroy()
        else:
            self._voice_bubbles[key] = active
        self._set_message_party(self._active_voice_bubble, callsign)

    def _voice_message_frames(self, event) -> int:
        part = getattr(self, "_active_voice_segment", None)
        previous = part["frames"] if part is not None else getattr(self, "_current_rx_frames", 0)
        frames = int(event.fields.get("frames", event.fields.get("received", previous)))
        if part is None:
            return frames
        part["frames"] = frames
        self._resolve_voice_message(str(event.fields.get("callsign") or self._current_rx_callsign or "Unknown"))
        return sum(segment["frames"] for segment in self._active_voice_bubble["segments"])


    def _handle_event_stream_start(self, event: SessionEvent, stamp: str) -> None:
        callsign = str(event.fields.get("callsign", "Unknown"))
        self._current_rx_callsign = callsign
        self._current_rx_mode = "voice"
        self._current_rx_frames = 0
        self._set_rxtx_receiving(callsign, "voice", 0)

        if not hasattr(self, "_voice_bubbles"):
            self._voice_bubbles = {}
        stream_id = event.fields.get("stream_id")
        protocol = event.fields.get("protocol", "")
        key = self._voice_message_key(callsign, stream_id, protocol)
        bubble = self._voice_bubbles.get(key)
        if bubble is None:
            bubble = self._append_chat_bubble(
                direction="rx", party=callsign, text="", is_voice_rx=True,
                timestamp=stamp, recording=event.fields.get("recording"),
            )
            bubble.update(stream_id=stream_id, protocol=protocol, timestamp=stamp, segments=[])
            if key is not None:
                self._voice_bubbles[key] = bubble
        self._active_voice_bubble = bubble
        self._active_voice_segment = {"frames": 0, "recording": event.fields.get("recording")}
        bubble["segments"].append(self._active_voice_segment)
        self._bind_voice_recordings(bubble)
        if len(bubble["segments"]) > 1:
            self._handle_event_voice_progress(event, stamp)
        header_tip = self._active_voice_bubble.get("header_tip")
        if header_tip is not None and stream_id is not None:
            lines = header_tip.text.splitlines()
            if not any(line.startswith("Stream ID:") for line in lines):
                header_tip.set_text(header_tip.text + f"\nStream ID: {int(stream_id):04X}")


    def _handle_event_voice_progress(self, event: SessionEvent, stamp: str) -> None:
        frames = self._voice_message_frames(event)
        callsign = str(event.fields.get("callsign") or self._current_rx_callsign or "Unknown")
        self._current_rx_callsign = callsign
        self._current_rx_frames = frames
        self._set_rxtx_receiving(callsign, "voice", frames)

        message = getattr(self, "_active_voice_bubble", None)
        if message:
            self._render_voice_message(
                message, self._received_voice_text(message, callsign, stamp, frames, "RX Voice"), frames,
                background="#ffdddd", border="#ef9a9a",
            )


    def _handle_event_stream_end(self, event: SessionEvent, stamp: str) -> None:
        callsign = str(event.fields.get("callsign") or self._current_rx_callsign or "Unknown")
        frames = self._voice_message_frames(event)
        self._set_rxtx_listening(last_call=callsign, mode="voice", details=str(frames), when=stamp)
        self._current_rx_callsign = None
        self._current_rx_mode = None

        message = getattr(self, "_active_voice_bubble", None)
        if message:
            self._render_voice_message(
                message, self._received_voice_text(message, callsign, stamp, frames, "Voice"), frames,
                background="#fffdf9", border="#d7cec3", foreground="#000000",
            )
            self._active_voice_bubble = None
            self._active_voice_segment = None

    @staticmethod
    def _received_voice_text(message: dict, callsign: str, stamp: str, frames: int, label: str) -> str:
        prefix = f"From {callsign}"
        if message.get("station_message"):
            prefix += f" \u2022 {message['station_message']}"
        timestamp = message.get("timestamp", stamp)
        return f"{prefix} \u2022 {timestamp} \u2022 {label} {timecode(frames * 0.02)}"

    @staticmethod
    def _render_voice_message(
        message: dict, text: str, frames: int, *,
        background: str | None = None, border: str | None = None, foreground: str | None = None,
    ) -> None:
        """Update one voice row while preserving replay disclosure and tooltip metadata."""
        style = {}
        if background is not None:
            style["background"] = background
            bubble = message.get("bubble")
            if bubble:
                bubble.configure(background=background, highlightbackground=border)
        if foreground is not None:
            style["foreground"] = foreground
        header = message.get("header")
        if header:
            playback = message.get("playback")
            update_header = playback.set_header if playback else header.configure
            update_header(text=text, **style)
        tip = message.get("header_tip")
        if tip:
            lines = tip.text.splitlines()
            lines[:1] = [f"{frames} frames"]
            tip.set_text("\n".join(lines))


    def _handle_event_app_stream(self, event: SessionEvent, stamp: str) -> None:
        # A stream header arrives before we can identify Fast Data. Remove the
        # provisional voice row once it is claimed by the application receiver.
        active_voice = getattr(self, "_active_voice_bubble", None)
        if active_voice:
            self._voice_bubbles = {key: value for key, value in self._voice_bubbles.items() if value is not active_voice}
            row = active_voice.get("row")
            if row is not None:
                row.destroy()
            self._active_voice_bubble = None
            self._active_voice_segment = None
        callsign = str(event.fields.get("callsign", "Unknown"))
        self._current_rx_callsign = callsign
        self._current_rx_mode = "app"
        self._set_rxtx_receiving(callsign, "app", "")


    def _handle_event_app_text(self, event: SessionEvent, stamp: str) -> None:
        if event.fields.get("picture_number") is not None and not event.fields.get("checksum_ok", True):
            # A damaged caption is not safe to attach to an image, but must
            # remain visible with the same checksum warning as ordinary text.
            self._append_chat_bubble(
                "rx", str(event.fields.get("source", "")),
                str(event.fields.get("text", "")), checksum_ok=False,
            )
            return
        if event.fields.get("picture_number") is None:
            text = str(event.fields.get("text", ""))
            self._resolve_voice_message(str(event.fields.get("callsign") or "Unknown"))
            if hasattr(self, "_active_voice_bubble") and self._active_voice_bubble:
                self._active_voice_bubble["station_message"] = text
            else:
                checksum_ok = bool(event.fields.get("checksum_ok", True))
                self._append_chat_bubble(
                    "rx",
                    str(event.fields.get("source", "")),
                    text,
                    checksum_ok=checksum_ok,
                )
            src = str(event.fields.get("source") or self._current_rx_callsign or "Unknown")
            self._current_rx_callsign = src
            self._current_rx_mode = "text"
            self._set_rxtx_listening(last_call=src, mode="text", when=stamp)


    def _handle_event_app_gps(self, event: SessionEvent, stamp: str) -> None:
        coord = str(event.fields.get("coordinates", ""))
        callsign = str(event.fields.get("source") or event.fields.get("callsign")
                       or getattr(self, "_current_rx_callsign", "") or "")
        self._resolve_voice_message(callsign)
        self._remember_position(callsign, coord)
        if hasattr(self, "_active_voice_bubble") and self._active_voice_bubble:
            self._active_voice_bubble["coordinates"] = coord
            self._set_message_party(self._active_voice_bubble, callsign)


    def _handle_event_app_picture(self, event: SessionEvent, stamp: str) -> None:
        self._show_received_picture(dict(event.fields))
        src = str(event.fields.get("source") or self._current_rx_callsign or "Unknown")
        received = int(event.fields.get("received", 0))
        total = int(event.fields.get("total", 0))
        self._current_rx_callsign = src
        self._current_rx_mode = "picture"
        self._current_rx_pic_tiles = f"{received}/{total}"
        self._set_rxtx_receiving(src, "picture", f"{received}/{total}")
        if event.fields.get("complete"):
            self._set_rxtx_listening(last_call=src, mode="picture", details=f"{total}/{total}", when=stamp)


    def _handle_event_app_stream_end(self, event: SessionEvent, stamp: str) -> None:
        src = str(event.fields.get("callsign") or self._current_rx_callsign or "Unknown")
        if self._current_rx_mode == "picture":
            tiles = getattr(self, "_current_rx_pic_tiles", "")
            self._set_rxtx_listening(last_call=src, mode="picture", details=tiles or "tiles", when=stamp)
        elif self._current_rx_mode == "text":
            self._set_rxtx_listening(last_call=src, mode="text", when=stamp)
        else:
            self._set_rxtx_listening()
        self._current_rx_callsign = None
        self._current_rx_mode = None


    def _handle_event_outgoing_picture(self, event: SessionEvent, stamp: str) -> None:
        image = event.fields.get("image")
        outgoing_id = str(event.fields.get("outgoing_id", ""))
        prepared = event.fields.get("prepared")
        if isinstance(image, Image.Image):
            display_out = image
            if event.fields.get("size") == "160x120" and image.height == 128:
                display_out = image.crop((0, 0, 160, 120))
            total_tiles = 80 if event.fields.get("size") == "160x120" else (
                300 if event.fields.get("size") == "320x240" else 1200
            )
            status_str = (
                f"{event.fields.get('size')}  Q{event.fields.get('quality')}  "
                f"{event.fields.get('crop')}  SENDING"
            )
            prep_dict = {
                "payload": getattr(prepared, "payload", None),
                "number": event.fields.get("number"),
                "size": event.fields.get("size"),
                "quality": event.fields.get("quality"),
                "crop": event.fields.get("crop"),
                "caption": event.fields.get("caption"),
                "destination": event.fields.get("destination"),
                "image": display_out,
                "total": total_tiles,
                "outgoing_id": outgoing_id,
            }
            self._append_chat_bubble(
                "tx",
                str(event.fields.get("destination", "")),
                str(event.fields.get("caption", "")),
                image=display_out,
                status=status_str,
                picture_id=outgoing_id or None,
                prepared_data=prep_dict,
            )
            tip = self._picture_tooltips.get(outgoing_id)
            if tip is not None:
                tip.set_text(
                    f"Picture #{event.fields.get('number')}\n"
                    f"Resolution: {event.fields.get('size')}\n"
                    f"Quality: Q{event.fields.get('quality')}\n"
                    f"Tiles: {total_tiles}/{total_tiles}\n"
                    f"Status: Sending..."
                )


    def _handle_event_app_tx_start(self, event: SessionEvent, stamp: str) -> None:
        self.app_tx_status_var.set(event.message)
        metadata = event.fields.get("metadata")
        is_pic = False
        if isinstance(metadata, dict):
            self._set_outgoing_status(
                str(metadata.get("outgoing_id", "")),
                "SENDING",
            )
            is_pic = bool(metadata.get("size") or event.fields.get("label", "").startswith("picture"))
        else:
            is_pic = event.fields.get("label", "").startswith("picture")
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        if is_pic:
            self._current_tx_mode = "picture"
            sz = metadata.get("size", "320x240") if isinstance(metadata, dict) else "320x240"
            total = 80 if sz == "160x120" else (300 if sz == "320x240" else 1200)
            self._current_tx_pic_total = total
            self._set_rxtx_transmitting(our_call, "picture", f"0/{total}")
        else:
            self._current_tx_mode = "text"
            self._set_rxtx_transmitting(our_call, "text", "")


    def _handle_event_app_tx_progress(self, event: SessionEvent, stamp: str) -> None:
        self.app_tx_status_var.set(event.message)
        metadata = event.fields.get("metadata")
        if isinstance(metadata, dict):
            self._set_outgoing_status(
                str(metadata.get("outgoing_id", "")),
                f"SENDING {event.fields.get('percent', 0)}%",
            )
        if self._current_tx_mode == "picture":
            our_call = self.callsign_var.get().strip().upper() or "MYCALL"
            pct = int(event.fields.get("percent", 0))
            total = getattr(self, "_current_tx_pic_total", 300)
            curr = int(total * pct / 100)
            self._set_rxtx_transmitting(our_call, "picture", f"{curr}/{total}")


    def _handle_event_app_tx_cancel(self, event: SessionEvent, stamp: str) -> None:
        metadata = event.fields.get("metadata")
        outgoing_id = (
            str(metadata.get("outgoing_id", ""))
            if isinstance(metadata, dict)
            else self._active_outgoing_id
        )
        self._set_outgoing_status(outgoing_id, "CANCELLED")
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        self._set_rxtx_listening(last_call=our_call, mode=self._current_tx_mode, when=stamp)
        self._current_tx_mode = None


    def _handle_event_app_tx_end(self, event: SessionEvent, stamp: str) -> None:
        metadata = event.fields.get("metadata")
        outgoing_id = (
            str(metadata.get("outgoing_id", ""))
            if isinstance(metadata, dict)
            else self._active_outgoing_id
        )
        is_resend = bool(metadata.get("is_resend")) if isinstance(metadata, dict) else False
        if is_resend and event.fields.get("reason") == "complete":
            final_status = f"Re-Sent {time.strftime('%H:%M:%S')}"
        else:
            final_status = (
                "SENT"
                if event.fields.get("reason") == "complete"
                else ("CANCELLED" if event.fields.get("reason") == "cancelled" else "STOPPED")
            )
        self._set_outgoing_status(outgoing_id, final_status)
        self.app_tx_status_var.set(final_status)
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        if self._current_tx_mode == "picture":
            total = getattr(self, "_current_tx_pic_total", 300)
            self._set_rxtx_listening(last_call=our_call, mode="picture", details=f"{total}/{total}", when=stamp)
        elif self._current_tx_mode == "text":
            self._set_rxtx_listening(last_call=our_call, mode="text", when=stamp)
        else:
            self._set_rxtx_listening()
        self._current_tx_mode = None


    def _handle_event_tx_start(self, event: SessionEvent, stamp: str) -> None:
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        self._current_tx_mode = "voice"
        self._current_tx_frames = 0
        self._set_rxtx_transmitting(our_call, "voice", 0)

        self._active_tx_voice_bubble = self._append_chat_bubble(
            direction="tx",
            party="CQCQCQ",
            text="",
            is_voice_tx=True,
            timestamp=stamp,
        )


    def _handle_event_tx_progress(self, event: SessionEvent, stamp: str) -> None:
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        frames = int(event.fields.get("frames", 0))
        self._current_tx_frames = frames
        self._set_rxtx_transmitting(our_call, "voice", frames)

        message = getattr(self, "_active_tx_voice_bubble", None)
        if message:
            self._render_voice_message(
                message, f"You to CQCQCQ \u2022 {stamp} \u2022 TX Voice {timecode(frames * 0.02)}", frames,
            )


    def _handle_event_tx_end(self, event: SessionEvent, stamp: str) -> None:
        self.microphone.stop()
        if not self._ptt_timed_out:
            self._ptt_held = False
            self.ptt_button.configure(text="Hold to Talk (PTT)")
            our_call = self.callsign_var.get().strip().upper() or "MYCALL"
            frames = int(event.fields.get("frames", self._current_tx_frames))
            self._set_rxtx_listening(last_call=our_call, mode="voice", details=str(frames), when=stamp)
        self._current_tx_mode = None

        message = getattr(self, "_active_tx_voice_bubble", None)
        if message:
            frames = int(event.fields.get("frames", self._current_tx_frames))
            self._render_voice_message(
                message, f"You to CQCQCQ \u2022 {stamp} \u2022 Voice {timecode(frames * 0.02)}", frames,
                background="#0c6f68", border="#0c6f68", foreground="#ffffff",
            )
            self._active_tx_voice_bubble = None


    def _handle_event_tx_timeout(self, event: SessionEvent, stamp: str) -> None:
        self._ptt_held = False
        self._ptt_timed_out = True
        self.microphone.stop()
        self.ptt_button.configure(text="TX limit reached - release PTT")
        our_call = self.callsign_var.get().strip().upper() or "MYCALL"
        self._set_rxtx_listening(last_call=our_call, mode="voice", details=str(self._current_tx_frames), when=stamp)
        self._current_tx_mode = None


    def _handle_event_tx_capture_stop(self, event: SessionEvent, stamp: str) -> None:
        self.microphone.stop()
        if not self._ptt_timed_out:
            self._ptt_held = False
            self.ptt_button.configure(text="Hold to Talk (PTT)")


    def _handle_event_app_send_finished(self, event: SessionEvent, stamp: str) -> None:
        result = event.fields.get("result")
        if not isinstance(result, SendResult):
            raise ValueError("send completion requires a SendResult")
        if not self._send_controller.is_current(result.operation.operation_id):
            return
        if result.error is not None:
            status = "FAILED"
        elif result.cancelled:
            status = "CANCELLED"
        elif result.completed:
            status = f"Re-Sent {time.strftime('%H:%M:%S')}" if self._send_is_resend else "SENT"
        else:
            status = "STOPPED"
        self._set_outgoing_status(self._active_outgoing_id, status)
        self._set_app_send_active(False, status)
        self._active_outgoing_id = None
        self._current_tx_mode = None
        if self._current_rx_mode and self._current_rx_callsign:
            details = self._current_rx_pic_tiles if self._current_rx_mode == "picture" else self._current_rx_frames
            self._set_rxtx_receiving(self._current_rx_callsign, self._current_rx_mode, details)
        else:
            self._set_rxtx_listening()
        if result.error is not None:
            messagebox.showerror("RS-MS1 send failed", str(result.error), parent=self.root)


    @cached_property
    def _event_handlers(self) -> dict[EventKind, Callable[[SessionEvent, str], None]]:
        """Bind an explicit event registry once, honoring subclass overrides."""
        return {
            EventKind.APP_SEND_FINISHED: self._handle_event_app_send_finished,
            EventKind.STATE: self._handle_event_state,
            EventKind.ERROR: self._handle_event_error,
            EventKind.STREAM_START: self._handle_event_stream_start,
            EventKind.VOICE_PROGRESS: self._handle_event_voice_progress,
            EventKind.STREAM_END: self._handle_event_stream_end,
            EventKind.APP_STREAM: self._handle_event_app_stream,
            EventKind.APP_TEXT: self._handle_event_app_text,
            EventKind.APP_GPS: self._handle_event_app_gps,
            EventKind.APP_PICTURE: self._handle_event_app_picture,
            EventKind.APP_STREAM_END: self._handle_event_app_stream_end,
            EventKind.OUTGOING_PICTURE: self._handle_event_outgoing_picture,
            EventKind.APP_TX_START: self._handle_event_app_tx_start,
            EventKind.APP_TX_PROGRESS: self._handle_event_app_tx_progress,
            EventKind.APP_TX_CANCEL: self._handle_event_app_tx_cancel,
            EventKind.APP_TX_END: self._handle_event_app_tx_end,
            EventKind.TX_START: self._handle_event_tx_start,
            EventKind.TX_PROGRESS: self._handle_event_tx_progress,
            EventKind.TX_END: self._handle_event_tx_end,
            EventKind.TX_TIMEOUT: self._handle_event_tx_timeout,
            EventKind.TX_CAPTURE_STOP: self._handle_event_tx_capture_stop,
        }


    def _dispatch_event(self, event: SessionEvent) -> None:
        metadata = event.fields.get("metadata", {})
        operation_id = event.fields.get("operation_id")
        if operation_id is None and isinstance(metadata, dict):
            operation_id = metadata.get("operation_id")
        if operation_id is not None:
            if not self._send_controller.is_current(str(operation_id)):
                return
            if (self._send_operation is not None
                    and self._send_operation.cancel_event.is_set()
                    and event.kind in {EventKind.OUTGOING_PICTURE, EventKind.APP_TX_START, EventKind.APP_TX_PROGRESS}):
                return
        stamp = event.timestamp.strftime("%H:%M:%S")
        should_log = (
            event.kind != EventKind.APP_PICTURE
            or bool(event.fields.get("complete"))
            or int(event.fields.get("received", 0)) == 1
            or int(event.fields.get("received", 0)) % 25 == 0
        )
        if should_log:
            self._append_log(stamp, event.message, event.kind)
        handler = self._event_handlers.get(event.kind)
        if handler is not None:
            handler(event, stamp)


    def _report_event_error(self, event: SessionEvent, error: Exception) -> None:
        self._append_log("ERROR", f"Could not handle {event.kind}: {error}", "error")


    def _drain_events(self) -> None:
        """Yield to Tk regularly and keep polling after individual handler errors."""
        if self._closing:
            return
        try:
            drain_events(self._events, self._dispatch_event, self._report_event_error)
            self._update_transmit_controls()
        finally:
            if not self._closing:
                self.root.after(50, self._drain_events)

