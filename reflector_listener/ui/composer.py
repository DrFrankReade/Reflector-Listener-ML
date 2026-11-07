"""Composer view methods for the application controller."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from reflector_listener.session import ConnectionState, SessionEvent
from reflector_listener.events import EventKind
from reflector_listener.sending import PreparedSend, SendOperation, SendResult
from reflector_listener.rsms1 import build_text_frame, normalize_payload_callsign, picture_preset, prepare_picture, resize_picture


class ComposerMixin:
    """UI concern shared by ListenerApp; all methods run on the Tk thread."""

    def _on_message_changed(self, *args: Any) -> None:
        text = self.message_var.get()
        if len(text) > 128:
            self.message_var.set(text[:128])
            text = text[:128]
        if hasattr(self, "char_counter_label"):
            self.char_counter_label.configure(text=f"{len(text)}/128")


    def _on_entry_keypress(self, ev: tk.Event[Any]) -> str | None:
        ch = ev.char
        if not ch:
            return None
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDBFF:
            self._pending_high_surrogate = ch
            return "break"
        if 0xDC00 <= cp <= 0xDFFF and self._pending_high_surrogate:
            high = self._pending_high_surrogate
            self._pending_high_surrogate = None
            full_char = (high + ch).encode("utf-16", "surrogatepass").decode("utf-16")
            self._insert_text_at_cursor(full_char)
            return "break"
        self._pending_high_surrogate = None
        return None


    def _insert_text_at_cursor(self, text: str) -> None:
        try:
            try:
                first = self.message_entry.index(tk.SEL_FIRST)
                last = self.message_entry.index(tk.SEL_LAST)
                self.message_entry.delete(first, last)
            except tk.TclError:
                pass
            idx = self.message_entry.index(tk.INSERT)
            self.message_entry.insert(idx, text)
        except Exception:
            curr = self.message_var.get()
            self.message_var.set(curr + text)


    def _setup_windows_emoji_input(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            imm32 = ctypes.windll.imm32

            WM_IME_COMPOSITION = 0x010F
            GCS_RESULTSTR = 0x0800
            GWLP_WNDPROC = -4

            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            )

            SetWindowLongPtr = (
                user32.SetWindowLongPtrW
                if hasattr(user32, "SetWindowLongPtrW")
                else user32.SetWindowLongW
            )
            SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, WNDPROC]
            SetWindowLongPtr.restype = ctypes.c_ssize_t

            CallWindowProc = user32.CallWindowProcW
            CallWindowProc.argtypes = [
                ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            CallWindowProc.restype = ctypes.c_ssize_t

            self.root.update_idletasks()
            hwnd = self.message_entry.winfo_id()

            def ime_wndproc(h: wintypes.HWND, msg: wintypes.UINT, wp: wintypes.WPARAM, lp: wintypes.LPARAM) -> int:
                if msg == WM_IME_COMPOSITION and (lp & GCS_RESULTSTR):
                    hIMC = imm32.ImmGetContext(h)
                    if hIMC:
                        try:
                            size = imm32.ImmGetCompositionStringW(hIMC, GCS_RESULTSTR, None, 0)
                            if size > 0:
                                buf = ctypes.create_unicode_buffer(size // 2 + 1)
                                imm32.ImmGetCompositionStringW(hIMC, GCS_RESULTSTR, buf, size + 2)
                                text_val = buf.value
                                if text_val:
                                    self.root.after_idle(lambda t=text_val: self._insert_text_at_cursor(t))
                        finally:
                            imm32.ImmReleaseContext(h, hIMC)
                        return 0
                return CallWindowProc(old_wndproc, h, msg, wp, lp)

            self._ime_wndproc_ref = WNDPROC(ime_wndproc)
            old_wndproc = SetWindowLongPtr(hwnd, GWLP_WNDPROC, self._ime_wndproc_ref)
            self._ime_old_wndproc = old_wndproc

            def cleanup_ime(_e: Any = None) -> None:
                try:
                    if hasattr(self, "_ime_old_wndproc") and self._ime_old_wndproc:
                        SetWindowLongPtr(hwnd, GWLP_WNDPROC, ctypes.cast(self._ime_old_wndproc, WNDPROC))
                        self._ime_old_wndproc = None
                except Exception:
                    pass

            self.message_entry.bind("<Destroy>", cleanup_ime, add="+")
        except Exception:
            pass


    def _choose_picture(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a picture",
            filetypes=(
                ("Pictures", "*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if not selected:
            return
        self._picture_path = Path(selected)
        self.picture_path_var.set(self._picture_path.name)
        self.picture_number_var.set((int(self.picture_number_var.get()) + 1) & 0xFF)
        self._refresh_picture_preview()
        self._set_controls_for_state(self.session.state)


    def _clear_picture(self) -> None:
        self._picture_path = None
        self.picture_path_var.set("No picture selected")
        self.picture_caption_var.set("")
        self._picture_preview_photo = None
        self.picture_preview_frame.grid_remove()
        self.picture_number_var.set((int(self.picture_number_var.get()) + 1) & 0xFF)
        self._set_controls_for_state(self.session.state)


    def _refresh_picture_preview(
        self, _event: tk.Event[Any] | None = None
    ) -> None:
        if self._picture_path is None:
            self.picture_preview_frame.grid_remove()
            return
        try:
            preset = picture_preset(
                self.picture_size_var.get(),
                int(self.picture_quality_var.get()),
            )
            with Image.open(self._picture_path) as opened:
                rendered = resize_picture(
                    opened,
                    preset,
                    self.picture_crop_var.get(),
                )
            preview = rendered.copy()
            if preset.label == "160x120" and preview.height == 128:
                preview = preview.crop((0, 0, 160, 120))
            preview.thumbnail((260, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
        except Exception as exc:
            self.picture_preview_label.configure(image="")
            self._picture_preview_photo = None
            self.picture_preview_details.configure(
                text=f"Preview unavailable\n{exc}"
            )
            self.picture_preview_frame.grid()
            return
        self._picture_preview_photo = photo
        self.picture_preview_label.configure(image=photo)
        canvas_note = (
            " (160x128 radio canvas)"
            if preset.label == "160x120"
            else ""
        )
        self.picture_preview_details.configure(
            text=(
                f"{self._picture_path.name}\n"
                f"{preset.label}{canvas_note}  Q{preset.quality}\n"
                f"{self.picture_crop_var.get()}"
            )
        )
        self.picture_preview_frame.grid()


    def _new_outgoing_id(self, kind: str) -> str:
        self._outgoing_counter += 1
        return f"tx-{kind}-{self._outgoing_counter}"


    def _chat_destination(self) -> str:
        destination = normalize_payload_callsign(
            self.chat_destination_var.get(), "CQCQCQ"
        )
        self.chat_destination_var.set(destination)
        return destination


    def _send_from_key(self, _event: tk.Event[Any]) -> str:
        self._on_send()
        return "break"


    def _on_send(self) -> None:
        if self.session.state is not ConnectionState.LINKED:
            messagebox.showinfo(
                "Not connected",
                "Connect to a reflector before sending.",
                parent=self.root,
            )
            return
        if self._app_send_active:
            return

        has_picture = self._picture_path is not None
        text_content = self.message_var.get().strip()

        if has_picture:
            if self._send_picture_with_caption(text_content):
                self._clear_picture()
                if self._picture_controls_visible:
                    self._toggle_picture_controls()
                self.message_var.set("")
        elif text_content:
            self._send_text()


    def _send_text(self) -> None:
        if self._app_send_active:
            return
        if self._receive_blocks_transmit():
            return
        if self.session.state is not ConnectionState.LINKED:
            messagebox.showinfo(
                "Not connected",
                "Connect to a reflector before sending.",
                parent=self.root,
            )
            return
        text = self.message_var.get().strip()
        if not text:
            return
        try:
            destination = self._chat_destination()
            frame = build_text_frame(self.callsign_var.get(), destination, text)
        except Exception as exc:
            messagebox.showerror("Cannot send text", str(exc), parent=self.root)
            return
        outgoing_id = self._new_outgoing_id("text")
        self._active_outgoing_id = outgoing_id
        self._append_chat_bubble(
            "tx",
            destination,
            text,
            status="SENDING",
            picture_id=outgoing_id,
        )
        if self._begin_send(
            outgoing_id,
            lambda operation: PreparedSend(frame, "text message", {
                "kind": "text",
                "destination": destination,
                "outgoing_id": outgoing_id,
                "operation_id": operation.operation_id,
            }),
            "Sending text...",
        ):
            self.message_var.set("")


    def _send_picture_with_caption(self, caption: str) -> bool:
        if self.session.state is not ConnectionState.LINKED or self._app_send_active:
            return False
        if self._receive_blocks_transmit():
            return False
        if self._picture_path is None:
            return False
        try:
            destination = self._chat_destination()
            number = int(self.picture_number_var.get()) & 0xFF
            source = normalize_payload_callsign(self.callsign_var.get(), "MYCALL")
            size = self.picture_size_var.get()
            quality = int(self.picture_quality_var.get())
            crop = self.picture_crop_var.get()
            picture_preset(size, quality)
        except (ValueError, TypeError, tk.TclError) as exc:
            messagebox.showerror("Cannot send picture", str(exc), parent=self.root)
            return False
        path = self._picture_path
        outgoing_id = self._new_outgoing_id("picture")

        def prepare(operation: SendOperation) -> PreparedSend:
            prepared = prepare_picture(path, source, destination, size, quality, crop, number, caption)
            metadata = {
                "kind": "picture", "destination": destination, "number": number,
                "size": prepared.preset.label, "quality": prepared.preset.quality,
                "crop": crop, "outgoing_id": outgoing_id,
                "operation_id": operation.operation_id,
            }
            if not operation.cancel_event.is_set():
                self._events.put(SessionEvent(
                    EventKind.OUTGOING_PICTURE, f"Prepared picture {number}",
                    fields={**metadata, "image": prepared.image, "caption": prepared.caption,
                            "prepared": prepared},
                ))
            return PreparedSend(prepared.payload, f"picture {number}", metadata)

        return self._begin_send(outgoing_id, prepare, "Preparing picture tiles...")


    def _resend_picture(self, pic_data: dict[str, Any]) -> None:
        if self.session.state is not ConnectionState.LINKED:
            messagebox.showinfo(
                "Not connected",
                "Connect to a reflector before sending.",
                parent=self.root,
            )
            return
        if self._app_send_active:
            messagebox.showinfo("Busy", "A transmission is already active.", parent=self.root)
            return
        if self._receive_blocks_transmit():
            return
        payload = pic_data.get("payload")
        if not payload:
            messagebox.showerror("Error", "Picture payload unavailable for re-send.", parent=self.root)
            return
        number = pic_data.get("number", 1)
        destination = str(pic_data.get("destination", "CQCQCQ"))
        caption = str(pic_data.get("caption", ""))
        orig_outgoing_id = pic_data.get("outgoing_id")
        outgoing_id = str(orig_outgoing_id) if orig_outgoing_id else self._new_outgoing_id("picture-resend")
        self._set_outgoing_status(outgoing_id, "RE-SENDING")
        self._begin_send(
            outgoing_id,
            lambda operation: PreparedSend(payload, f"picture {number}", {
                "kind": "picture", "destination": destination,
                "outgoing_id": outgoing_id, "operation_id": operation.operation_id,
                "number": number, "is_resend": True, "size": pic_data.get("size"),
            }),
            f"Re-sending picture {number}...",
            is_resend=True,
        )


    def _begin_send(
        self,
        outgoing_id: str,
        prepare: Callable[[SendOperation], PreparedSend],
        status: str,
        *,
        is_resend: bool = False,
    ) -> bool:
        if self._app_send_active:
            return False
        self._active_outgoing_id = outgoing_id
        self._send_is_resend = is_resend
        self._set_app_send_active(True, status)
        try:
            self._send_operation = self._send_controller.start(
                self._new_outgoing_id("operation"), prepare
            )
        except Exception as exc:
            self._set_outgoing_status(outgoing_id, "FAILED")
            self._set_app_send_active(False, f"Send failed: {exc}")
            self._active_outgoing_id = None
            messagebox.showerror("Cannot send", str(exc), parent=self.root)
            return False
        return True


    def _cancel_send(self) -> None:
        if self._send_controller.cancel():
            self._set_outgoing_status(self._active_outgoing_id, "CANCELLING")
            self._set_app_send_active(True, "Cancelling...")


    def _on_send_finished(self, result: SendResult) -> None:
        # Called on the worker thread; all UI changes happen during event delivery.
        self._events.put(SessionEvent(
            EventKind.APP_SEND_FINISHED, "Send worker finished",
            fields={"operation_id": result.operation.operation_id, "result": result},
        ))


    def _set_app_send_active(self, active: bool, status: str | None = None) -> None:
        self._app_send_active = active
        if status is not None:
            self.app_tx_status_var.set(status)
        if hasattr(self, "send_button") and hasattr(self, "cancel_button"):
            if active:
                self.send_button.grid_remove()
                self.cancel_button.grid()
            else:
                self.cancel_button.grid_remove()
                self.send_button.grid()
        if hasattr(self, "choose_picture_button"):
            self.choose_picture_button.configure(
                state="disabled" if active else "normal"
            )
        self._update_transmit_controls()

    def _receive_blocks_transmit(self) -> bool:
        if self.session.receiving:
            self.app_tx_status_var.set("Receiving — wait for active streams to end.")
            self._update_transmit_controls()
            return True
        return False

    def _update_transmit_controls(self) -> None:
        linked = self.session.state is ConnectionState.LINKED
        receiving = self.session.receiving
        active = self._app_send_active
        held = getattr(self, "_ptt_held", False)
        state = (linked, receiving, active, self.session.transmitting, held)
        if state == getattr(self, "_transmit_controls_state", None):
            return
        self._transmit_controls_state = state
        if hasattr(self, "ptt_button"):
            self.ptt_button.configure(state="normal" if linked and not receiving and not active else "disabled")
            if not held:
                self.ptt_button.configure(text="Receiving — wait to transmit" if receiving else "Hold to Talk (PTT)")
        if hasattr(self, "send_button"):
            self.send_button.configure(state="normal" if linked and not receiving and not active and not self.session.transmitting else "disabled")

