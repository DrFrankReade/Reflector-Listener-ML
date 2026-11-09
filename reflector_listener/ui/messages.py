"""Messages view methods for the application controller."""

from __future__ import annotations

import math
import time
from typing import Any, NamedTuple
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from reflector_listener.preferences import Preferences
from .timeline import MessageTimeline
from .tooltip import ToolTip
from .playback import PlaybackControl


class _BubbleTheme(NamedTuple):
    background: str
    foreground: str
    muted: str
    border: str
    border_width: int


class _PictureActions:
    """Own one label's stable bindings and the current, replaceable image.

    Rebinding on every received tile leaves old Python callbacks registered in
    Tcl, retaining their images until the row is destroyed. These callbacks read
    current state instead, and the label owns a single reusable context menu.
    """

    def __init__(self, owner: MessagesMixin, label: tk.Label) -> None:
        self.owner = owner
        self.label = label
        self.image: Image.Image | None = None
        self.prepared_data: dict[str, Any] | None = None
        self.menu: tk.Menu | None = None
        label.bind("<Button-1>", self._open)
        label.bind("<Button-3>", self._show_menu)
        label.bind("<Destroy>", self._release, add="+")

    def _open(self, _event: tk.Event[Any]) -> None:
        if self.image is not None:
            self.owner._open_image_lightbox(self.image)

    def _save(self) -> None:
        if self.image is not None:
            self.owner._save_picture_as(self.image)

    def _resend(self) -> None:
        if self.prepared_data:
            self.owner._resend_picture(self.prepared_data)

    def _show_menu(self, event: tk.Event[Any]) -> None:
        if self.menu is None:
            self.menu = tk.Menu(self.label, tearoff=0)
        self.menu.delete(0, "end")
        self.menu.add_command(label="Save As...", command=self._save)
        if self.prepared_data:
            self.menu.add_command(
                label=f"Re-Send Image (Pic #{self.prepared_data.get('number', 1)})",
                command=self._resend,
            )
        self.menu.tk_popup(event.x_root, event.y_root)

    def _release(self, event: tk.Event[Any]) -> None:
        if event.widget is self.label:
            self.image = None
            self.prepared_data = None
            # Tk destroys the menu with its owning image label.
            self.menu = None


class MessagesMixin:
    """UI concern shared by ListenerApp; all methods run on the Tk thread."""

    def _trim_message_history(self) -> None:
        history = getattr(self, "_message_history", {})
        limit = getattr(self, "preferences", Preferences()).message_history_limit
        while len(history) > limit:
            self._discard_message(next(iter(history.values())))

    def _clear_message_history(self) -> None:
        for message in tuple(getattr(self, "_message_history", {}).values()):
            self._discard_message(message)
        self._saved_tx_pictures.clear()
        self._last_call_summary = None

    def _discard_message(self, message) -> None:
        playback = message.get("playback")
        if playback is not None and hasattr(self, "stream_log"):
            if self.replay.state.token == playback.clip.token:
                self.replay.stop()
            self.stream_log.discard(playback.clip)
        message["row"].destroy()

    def _forget_message(self, message) -> None:
        """Release UI indexes when a row is evicted, merged, or discarded."""
        self._message_history.pop(message["row"], None)
        picture_id = message.get("picture_id")
        pictures = getattr(self, "_picture_bubbles", {})
        if picture_id and pictures.get(picture_id, {}).get("row") is message["row"]:
            pictures.pop(picture_id, None)
            self._picture_tooltips.pop(picture_id, None)
        voices = getattr(self, "_voice_bubbles", {})
        for key in [key for key, value in voices.items() if value is message]:
            del voices[key]
        if getattr(self, "_active_voice_bubble", None) is message:
            self._active_voice_bubble = None
            self._active_voice_segment = None
        if getattr(self, "_active_tx_voice_bubble", None) is message:
            self._active_tx_voice_bubble = None

    def _create_selectable_text(
        self,
        parent: tk.Widget,
        text: str,
        bg: str,
        fg: str,
    ) -> tk.Text:
        height, width = self._selectable_text_size(text)

        txt = tk.Text(
            parent,
            wrap="word",
            font=("Segoe UI Emoji", 11),
            background=bg,
            foreground=fg,
            height=height,
            width=width,
            highlightthickness=0,
            borderwidth=0,
            cursor="xterm",
            exportselection=True,
            takefocus=False,
        )
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        def copy_selection() -> None:
            try:
                sel = txt.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                sel = txt.get("1.0", "end-1c")
            if sel:
                self.root.clipboard_clear()
                self.root.clipboard_append(sel)

        menu = tk.Menu(txt, tearoff=0)
        menu.add_command(label="Copy", command=copy_selection)
        txt.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        txt.bind("<<Copy>>", lambda _e: (copy_selection(), "break"))
        txt.bind("<Control-c>", lambda _e: (copy_selection(), "break"))
        txt.bind("<Control-C>", lambda _e: (copy_selection(), "break"))
        txt.bind("<MouseWheel>", self._on_chat_mousewheel)
        return txt


    @staticmethod
    def _selectable_text_size(text: str) -> tuple[int, int]:
        """Use the same bounded wrap estimate for new and updated captions."""
        lines = text.splitlines() or [""]
        wrapped_count = sum(max(1, math.ceil(len(line) / 45)) for line in lines)
        return max(1, min(15, wrapped_count)), min(48, max(map(len, lines), default=1))

    def _update_selectable_text(self, txt: tk.Text, text: str) -> None:
        height, width = self._selectable_text_size(text)
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.insert("1.0", text)
        txt.configure(height=height, width=width, state="disabled")


    def _save_picture_as(self, image: Image.Image) -> None:
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Picture",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            try:
                if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
                    if image.mode in ("RGBA", "P"):
                        image = image.convert("RGB")
                    image.save(path, "JPEG", quality=95)
                else:
                    image.save(path)
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to save image:\n{exc}", parent=self.root)


    def _open_image_lightbox(self, image: Image.Image) -> None:
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.configure(background="#000000", highlightbackground="#000000", highlightthickness=1)
        top.transient(self.root)
        top.attributes("-topmost", True)

        root_w = max(400, self.root.winfo_width())
        root_h = max(300, self.root.winfo_height())
        max_w = max(320, root_w - 40)
        max_h = max(240, root_h - 40)
        w, h = image.size
        scale = 1.0
        if w < 480 and h < 360:
            scale = min(2.5, max_w / w, max_h / h)
        elif w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)

        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        resample_filter = Image.Resampling.NEAREST if scale > 1.5 else Image.Resampling.LANCZOS
        disp_img = image.resize((disp_w, disp_h), resample_filter)
        photo = ImageTk.PhotoImage(disp_img)

        border_width = 1
        win_w = disp_w + (border_width * 2)
        win_h = disp_h + (border_width * 2)

        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        cx = rx + max(0, (root_w - win_w) // 2)
        cy = ry + max(0, (root_h - win_h) // 2)
        top.geometry(f"{win_w}x{win_h}+{cx}+{cy}")

        lbl = tk.Label(top, image=photo, background="#000000", borderwidth=0, cursor="hand2")
        lbl.image = photo
        lbl.pack(padx=border_width, pady=border_width)

        ToolTip(lbl, "Click to close")

        def close_fn(_e: Any = None) -> None:
            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()

        def close_fn_event(e):
            close_fn()
        lbl.bind("<Button-1>", close_fn_event)
        top.bind("<Button-1>", close_fn_event)
        top.bind("<Escape>", close_fn_event)
        lbl.focus_set()


        top.grab_set()
        top.focus_set()


    @staticmethod
    def _chat_photo(image: Image.Image) -> ImageTk.PhotoImage:
        preview = image.copy()
        preview.thumbnail((380, 280), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(preview)


    def _get_bubble_theme(self, checksum_ok: bool, is_voice: bool, outgoing: bool) -> _BubbleTheme:
        if not checksum_ok:
            return _BubbleTheme("#ffebee", "#b71c1c", "#c62828", "#ef9a9a", 1)
        if is_voice:
            return _BubbleTheme("#ffdddd", "#282521", "#c62828", "#ef9a9a", 1)
        if outgoing:
            return _BubbleTheme("#0c6f68", "#ffffff", "#d5efeb", "#0c6f68", 0)
        return _BubbleTheme("#fffdf9", "#282521", "#7b746b", "#d7cec3", 1)


    def _append_chat_bubble(
        self,
        direction: str,
        party: str,
        text: str,
        *,
        image: Image.Image | None = None,
        status: str = "",
        picture_id: str | None = None,
        checksum_ok: bool = True,
        prepared_data: dict[str, Any] | None = None,
        timestamp: str | None = None,
        is_voice_rx: bool = False,
        is_voice_tx: bool = False,
        recording=None,
    ) -> dict[str, Any]:
        outgoing = direction == "tx"
        is_voice = is_voice_rx or is_voice_tx
        theme = self._get_bubble_theme(checksum_ok, is_voice, outgoing)
        message = self._create_bubble_frame(party, theme, outgoing=outgoing)
        message["picture_id"] = picture_id
        row, bubble = message["row"], message["bubble"]
        voice_mode = "RX" if is_voice_rx else ("TX" if is_voice_tx else None)
        header_label, header_tip, playback = self._create_bubble_header(
            message, theme, outgoing=outgoing, voice_mode=voice_mode,
            timestamp=timestamp, recording=recording,
        )
        if not checksum_ok:
            self._show_checksum_warning(bubble, theme.background)

        text_widget: tk.Text | None = None
        if text:
            text_widget = self._create_selectable_text(bubble, text, theme.background, theme.foreground)
            text_widget.pack(fill="x", pady=(4, 0))
            if not outgoing:
                self._message_tooltip(text_widget, message)
        image_label, photo = self._create_bubble_picture(
            message, image, theme, outgoing=outgoing, status=status, prepared_data=prepared_data,
        )

        status_label = tk.Label(
            bubble,
            text=status,
            background=theme.background,
            foreground="#000000" if is_voice else theme.muted,
            font=("Segoe UI", 8),
            anchor="e",
        )
        if not is_voice:
            status_label.pack(fill="x", pady=(4, 0))
            status_label.bind("<MouseWheel>", self._on_chat_mousewheel)

        if picture_id:
            self._picture_bubbles[picture_id] = {
                "row": row,
                "text": text_widget,
                "image": image_label,
                "status": status_label,
                "photo": photo,
                "prepared_data": prepared_data,
                "raw_image": image,
            }
        self.root.after_idle(self._scroll_chat_to_end)

        message.update({
            "row": row,
            "bubble": bubble,
            "text": text_widget,
            "header": header_label,
            "header_tip": header_tip,
            "status": status_label,
            "playback": playback,
            "picture_id": picture_id,
            "photo": photo,
        })
        if not hasattr(self, "_message_history"):
            self._message_history = {}
        self._message_history[row] = message
        row.bind("<Destroy>", lambda event: self._forget_message(message) if event.widget is row else None, add="+")
        self._trim_message_history()
        return message

    def _create_bubble_frame(
        self, party: str, theme: _BubbleTheme, *, outgoing: bool
    ) -> dict[str, Any]:
        """Attach an aligned message row and optional station badge to the timeline."""
        row = tk.Frame(self.chat_body, background="#f3eee6")
        if isinstance(self.chat_body, MessageTimeline):
            self.chat_body.add_row(row, padx=18, pady=5)
        else:
            row.pack(fill="x", padx=18, pady=5)
        row.bind("<MouseWheel>", self._on_chat_mousewheel)
        bubble = tk.Frame(
            row, background=theme.background, padx=12, pady=9,
            highlightbackground=theme.border, highlightthickness=theme.border_width,
        )
        bubble.pack(side="right" if outgoing else "left", padx=(90, 0) if outgoing else (0, 90))
        bubble.bind("<MouseWheel>", self._on_chat_mousewheel)
        message = {"party": party, "row": row, "bubble": bubble}
        if hasattr(self, "avatar_badges"):
            message["avatar"] = self.avatar_badges.attach(
                row, bubble, party, self._on_chat_mousewheel, outgoing=outgoing,
            )
        return message

    def _create_bubble_header(
        self,
        message: dict[str, Any],
        theme: _BubbleTheme,
        *,
        outgoing: bool,
        voice_mode: str | None,
        timestamp: str | None,
        recording,
    ) -> tuple[tk.Label, ToolTip, PlaybackControl | None]:
        """Render station/time identity and optional voice playback controls."""
        time_str = timestamp or time.strftime("%H:%M:%S")
        header_text = ("You to " if outgoing else "From ") + (message["party"] or "Unknown") + f" \u2022 {time_str}"
        if voice_mode:
            header_text += f" \u2022 {voice_mode} Voice 0:00"
        header = tk.Label(
            message["bubble"], text=header_text, background=theme.background,
            foreground="#000000" if voice_mode else theme.muted,
            font=("Segoe UI", 9) if voice_mode else ("Bahnschrift SemiBold", 9), anchor="w",
        )
        header.pack(fill="x")
        header.bind("<MouseWheel>", self._on_chat_mousewheel)
        playback = None
        if voice_mode == "RX" and recording is not None:
            playback = PlaybackControl(message["bubble"], header, recording, self.replay)
        tip = self._message_tooltip(header, message, "0 frames" if voice_mode else "", outgoing=outgoing)
        return header, tip, playback

    def _message_tooltip(
        self, widget: tk.Widget, message: dict[str, Any], text: str = "", *, outgoing: bool = False
    ) -> ToolTip:
        """Read station identity when shown so late headers/GPS remain visible."""
        tip = ToolTip(widget, text)
        if not outgoing:
            tip.suffix = lambda: self._distance_tooltip(message["party"])
        return tip

    def _show_checksum_warning(self, bubble: tk.Frame, background: str) -> None:
        warning = tk.Label(
            bubble, text="⚠️ CHECKSUM ERROR", foreground="#c62828", background=background,
            font=("Segoe UI", 8, "bold"), anchor="w",
        )
        warning.pack(fill="x", pady=(2, 0))
        ToolTip(warning, "This message had checksum errors over the air and may contain corrupted characters.")
        warning.bind("<MouseWheel>", self._on_chat_mousewheel)

    def _create_bubble_picture(
        self,
        message: dict[str, Any],
        image: Image.Image | None,
        theme: _BubbleTheme,
        *,
        outgoing: bool,
        status: str,
        prepared_data: dict[str, Any] | None,
    ) -> tuple[tk.Label | None, ImageTk.PhotoImage | None]:
        """Create the preview, current-image actions, and station-aware tooltip."""
        if image is None:
            return None, None
        photo = self._chat_photo(image)
        label = tk.Label(
            message["bubble"], image=photo, background=theme.background,
            borderwidth=0, cursor="hand2",
        )
        label.pack(pady=(7, 1))
        self._set_picture_actions(label, image, prepared_data if outgoing else None)
        label.bind("<MouseWheel>", self._on_chat_mousewheel)
        tip = self._message_tooltip(
            label, message, f"Resolution: {image.width}x{image.height}\n{status}", outgoing=outgoing,
        )
        if message["picture_id"]:
            self._picture_tooltips[message["picture_id"]] = tip
        return label, photo

    def _set_picture_actions(
        self, label: tk.Label, image: Image.Image, prepared_data: dict[str, Any] | None = None
    ) -> None:
        actions = getattr(label, "_picture_actions", None)
        if actions is None:
            actions = label._picture_actions = _PictureActions(self, label)
        actions.image = image
        actions.prepared_data = prepared_data

    def _set_message_party(self, message: dict, callsign: str) -> None:
        """Keep late station identity consistent for tooltips and its badge."""
        coord = message.get("coordinates")
        identity = (callsign, coord)
        if message.get("_identity") == identity:
            return
        message["_identity"] = identity
        message["party"] = callsign
        badge = message.get("avatar")
        if badge is not None:
            badge.set_callsign(callsign)
        if coord:
            self._remember_position(callsign, coord)
        tip = message.get("header_tip")
        if tip:
            lines = tip.text.splitlines()
            if coord:
                lines = [line for line in lines if not line.startswith("GPS:")]
                lines.append(f"GPS: {coord}")
            tip.set_text("\n".join(lines))


    def _show_received_picture(self, fields: dict[str, Any]) -> None:
        picture_id = str(fields.get("picture_id", ""))
        image = fields.get("image")
        if not picture_id or not isinstance(image, Image.Image):
            return
        caption = str(fields.get("caption", ""))
        status, tip_text = self._picture_progress_text(fields, image)
        existing = self._picture_bubbles.get(picture_id)
        if existing is None:
            self._append_chat_bubble(
                "rx",
                str(fields.get("source", "")),
                caption,
                image=image,
                status=status,
                picture_id=picture_id,
            )
        else:
            self._replace_picture_image(existing, image)

        tip = self._picture_tooltips.get(picture_id)
        if tip is not None:
            tip.set_text(tip_text)
        if existing is None:
            return

        self._update_picture_caption(existing, caption)
        status_label = existing.get("status")
        if status_label is not None:
            status_label.configure(text=status)
        self.root.after_idle(self._scroll_chat_to_end)

    @staticmethod
    def _picture_progress_text(fields: dict[str, Any], image: Image.Image) -> tuple[str, str]:
        """Format the inline summary and detailed tooltip from one progress update."""
        received = int(fields.get("received", 0))
        total = int(fields.get("total", 0))
        complete = bool(fields.get("complete"))
        number = fields.get("number", "?")
        size = fields.get("size", f"{image.width}x{image.height}")
        quality = fields.get("quality", "?")
        pct = round(received * 100 / max(1, total))
        status = (
            f"{size}  Q{quality}  "
            f"{received}/{total} tiles"
            + ("  complete" if complete else "")
        )
        tip_text = (
            f"Picture #{number}\n"
            f"Resolution: {size}\n"
            f"Quality: Q{quality}\n"
            f"Tiles: {received}/{total} ({pct}% complete)"
            + ("\nStatus: Complete" if complete else "")
        )
        return status, tip_text

    def _replace_picture_image(self, existing: dict[str, Any], image: Image.Image) -> None:
        """Replace preview references without retaining earlier tile snapshots."""
        photo = self._chat_photo(image)
        image_label = existing.get("image")
        if image_label is not None:
            image_label.configure(image=photo)
            self._set_picture_actions(image_label, image)
        existing["photo"] = photo
        existing["raw_image"] = image
        message = getattr(self, "_message_history", {}).get(existing.get("row"))
        if message is not None:
            message["photo"] = photo

    def _update_picture_caption(self, existing: dict[str, Any], caption: str) -> None:
        """Keep a late/updated caption before its image and in both UI indexes."""
        if not caption:
            return
        image_label = existing.get("image")
        text_widget = existing.get("text")
        message = getattr(self, "_message_history", {}).get(existing.get("row"))
        if text_widget is not None:
            self._update_selectable_text(text_widget, caption)
            text_widget.pack(fill="x", pady=(4, 0), before=image_label)
        elif image_label is not None and image_label.master is not None:
            theme = self._get_bubble_theme(True, False, False)
            text_widget = self._create_selectable_text(
                image_label.master, caption, theme.background, theme.foreground,
            )
            text_widget.pack(fill="x", pady=(4, 0), before=image_label)
            existing["text"] = text_widget
            if message is not None:
                self._message_tooltip(text_widget, message)
        if message is not None:
            message["text"] = text_widget


    def _set_outgoing_status(self, outgoing_id: str | None, status: str) -> None:
        if not outgoing_id:
            return
        bubble = self._picture_bubbles.get(outgoing_id)
        if bubble is not None and bubble.get("status") is not None:
            bubble["status"].configure(text=status)
        tip = self._picture_tooltips.get(outgoing_id)
        if tip is not None:
            prep = bubble.get("prepared_data") if bubble else None
            if prep:
                num = prep.get("number", 1)
                sz = prep.get("size", "")
                q = prep.get("quality", "")
                tot = prep.get("total", 300)
                tip.set_text(
                    f"Picture #{num}\n"
                    f"Resolution: {sz}\n"
                    f"Quality: Q{q}\n"
                    f"Tiles: {tot}/{tot}\n"
                    f"Status: {status}"
                )
            else:
                lines = tip.text.splitlines()
                new_lines = [l for l in lines if not l.startswith("Status:")]
                new_lines.append(f"Status: {status}")
                tip.set_text("\n".join(new_lines))

