"""Station badges and a compact avatar picker sharing bundled artwork."""

import logging
import math
import tkinter as tk
from tkinter import ttk
import weakref
from PIL import Image, ImageOps, ImageTk

from reflector_listener.stations import ANIMALS, COLORS, DEFAULT_AVATAR, AvatarAssignments, station_key
from .assets import asset_path
from .tooltip import ToolTip


def _badge_image(source, size, color):
    """Render a circular badge at its final pixel size, with smooth edges."""
    center = size / 2
    radius = center - 1
    # Radial coverage gives identical edges on both axes, even at odd sizes.
    mask = Image.frombytes("L", (size, size), bytes(
        round(255 * min(1, max(0, radius + 0.5 - math.hypot(x + 0.5 - center, y + 0.5 - center))))
        for y in range(size) for x in range(size)
    ))
    picture = Image.new("RGBA", (size, size), color)
    picture.putalpha(mask)
    # Preserve the portrait's proportions and place it on whole pixels.
    extent = round(size * 0.78)
    portrait = ImageOps.contain(source.convert("RGBA"), (extent, extent), Image.Resampling.LANCZOS)
    picture.alpha_composite(portrait, ((size - portrait.width) // 2, (size - portrait.height) // 2))
    return picture


class AvatarBadges:
    def __init__(self, root, assignments: AvatarAssignments, *, enabled=False, own_avatar=DEFAULT_AVATAR) -> None:
        self.root, self.assignments, self.enabled = root, assignments, enabled
        self.own_avatar = own_avatar
        self.size = round(root.winfo_fpixels("28p"))
        self._photos = {}
        self._badges = weakref.WeakSet()

    def attach(self, row, bubble, callsign, on_mousewheel, *, outgoing=False):
        badge = _StationBadge(self, row, bubble, callsign, on_mousewheel, outgoing=outgoing)
        self._badges.add(badge)
        badge.refresh()
        return badge

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        for badge in tuple(self._badges):
            badge.refresh()

    def shuffle(self, callsign):
        key = station_key(callsign)
        if self.assignments.shuffle(key) is not None:
            for badge in tuple(self._badges):
                if not badge.outgoing and badge.callsign == key:
                    badge.refresh()

    def set_own_avatar(self, assignment):
        self.own_avatar = assignment
        for badge in tuple(self._badges):
            if badge.outgoing:
                badge.refresh()

    def photo(self, animal, color):
        key = (animal, color)
        if key not in self._photos:
            try:
                with Image.open(asset_path(f"avatars/{animal}.png")) as source:
                    picture = _badge_image(source, self.size, color)
                    self._photos[key] = ImageTk.PhotoImage(picture, master=self.root)
            except OSError as exc:
                logging.getLogger(__name__).warning("Cannot load avatar %s: %s", animal, exc)
                self._photos[key] = None
        return self._photos[key]


class _StationBadge:
    def __init__(self, owner, row, bubble, callsign, on_mousewheel, *, outgoing=False):
        self.owner, self.row, self.bubble = owner, row, bubble
        self.outgoing = outgoing
        self.callsign = station_key(callsign)
        self.on_mousewheel = on_mousewheel
        self.widget = None
        self._menu = None
        row.bind("<Destroy>", self._destroy, add="+")

    def set_callsign(self, callsign):
        key = station_key(callsign)
        if key != self.callsign:
            self.callsign = key
            self.refresh()

    def refresh(self):
        assignment = (self.owner.own_avatar if self.outgoing else
                      self.owner.assignments.get(self.callsign) if self.owner.enabled else None)
        photo = self.owner.photo(assignment.animal, assignment.color) if assignment else None
        if photo is None:
            if self.widget is not None:
                self.widget.pack_forget()
            return
        size = self.owner.size
        if self.widget is None:
            self.widget = tk.Canvas(self.row, width=size, height=size,
                                    background=self.row.cget("background"),
                                    borderwidth=0, highlightthickness=0, takefocus=False)
            self.widget.bind("<MouseWheel>", self.on_mousewheel)
            if not self.outgoing:
                self.widget.bind("<Button-3>", self._show_menu)
            self._portrait = self.widget.create_image(0, 0, anchor="nw")
        self.widget.itemconfigure(self._portrait, image=photo)
        self.widget.pack(side="right" if self.outgoing else "left", anchor="n",
                         padx=(8, 0) if self.outgoing else (0, 8), pady=3, before=self.bubble)

    def _show_menu(self, event):
        if self._menu is None:
            self._menu = tk.Menu(self.widget, tearoff=False)
            self._menu.add_command(label="Shuffle this Avatar",
                                   command=lambda: self.owner.shuffle(self.callsign))
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()
        return "break"

    def _destroy(self, event):
        if event.widget == self.row:
            self.owner._badges.discard(self)


class AvatarPicker(ttk.Frame):
    """Compact draft selector sharing the message badge renderer and palette."""

    def __init__(self, parent, animal_var, color_var):
        super().__init__(parent)
        self.animal_var, self.color_var = animal_var, color_var
        self.size = round(self.winfo_fpixels("28p"))
        self._arrow("‹", animal_var, ANIMALS, -1, "Previous animal")
        self.preview = ttk.Label(self)
        self.preview.pack(side="left", padx=3)
        self._arrow("›", animal_var, ANIMALS, 1, "Next animal")
        ttk.Frame(self, width=12).pack(side="left")
        self._arrow("‹", color_var, COLORS, -1, "Previous color")
        self.swatch = tk.Canvas(self, width=round(self.size * .5), height=round(self.size * .5),
                                borderwidth=0, highlightthickness=0, takefocus=False)
        self.swatch.pack(side="left", padx=5)
        self._arrow("›", color_var, COLORS, 1, "Next color")
        self._traces = [(variable, variable.trace_add("write", self._refresh))
                        for variable in (animal_var, color_var)]
        self.bind("<Destroy>", self._destroy, add="+")
        self._refresh()

    def _arrow(self, text, variable, choices, step, tip):
        def change():
            variable.set(choices[(choices.index(variable.get()) + step) % len(choices)])
        button = ttk.Button(self, text=text, width=2, command=change)
        button.pack(side="left")
        ToolTip(button, tip)

    def _refresh(self, *_):
        animal, color = self.animal_var.get(), self.color_var.get()
        if animal not in ANIMALS or color not in COLORS:
            return
        try:
            with Image.open(asset_path(f"avatars/{animal}.png")) as source:
                self._photo = ImageTk.PhotoImage(_badge_image(source, self.size, color), master=self)
            self.preview.configure(image=self._photo)
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot load avatar %s: %s", animal, exc)
            self.preview.configure(image="")
        self.swatch.configure(background=color)

    def _destroy(self, event):
        if event.widget == self:
            for variable, trace in self._traces:
                variable.trace_remove("write", trace)
