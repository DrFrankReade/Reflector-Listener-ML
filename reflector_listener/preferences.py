"""Immutable, validated user preferences independent of Tk and networking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import re
import math
from typing import Any, Mapping

from .location import coordinates, grid_center
from .stations import ANIMALS, COLORS, DEFAULT_AVATAR, AvatarAssignment


@dataclass(frozen=True)
class Preferences:
    units: str = "miles"
    voice_suffix: str = "PC"
    station_message: str = ""
    picture_suffix: str = "SSTV"
    text_suffix: str = "TEXT"
    latitude: str = ""
    longitude: str = ""
    grid: str = ""
    send_dprs: bool = False
    picture_beep: bool = True
    stream_logging: bool = False
    ml_enhanced_decoder: bool = False
    transmit_device: str = ""
    receive_device: str = ""
    transmit_gain: float = 1.0
    packet_loss_concealment: bool = True
    incoming_avatars: bool = False
    own_avatar_animal: str = DEFAULT_AVATAR.animal
    own_avatar_color: str = DEFAULT_AVATAR.color
    message_history_limit: int = 100
    max_debug_mb: int = 100
    max_stream_mb: int = 100

    def __post_init__(self) -> None:
        self._validate_display_settings()
        self._normalize_suffixes()
        self._validate_limits_and_flags()
        self._normalize_location()

    def _validate_display_settings(self) -> None:
        """Validate the station identity and message presentation choices."""
        if self.own_avatar_animal not in ANIMALS or self.own_avatar_color not in COLORS:
            raise ValueError("Choose an avatar animal and color from the available options.")
        if not isinstance(self.station_message, str) or not re.fullmatch(r"[\x20-\x7e]{0,20}", self.station_message):
            raise ValueError("Call sign message allows up to 20 printable ASCII characters.")
        if type(self.message_history_limit) is not int or not 1 <= self.message_history_limit <= 500:
            raise ValueError("Message history must be a whole number from 1 to 500.")
        if self.units not in ("miles", "kilometers"):
            raise ValueError("Choose miles or kilometers.")

    def _normalize_suffixes(self) -> None:
        """Store each radio call sign suffix in its canonical on-air form."""
        for key in ("voice_suffix", "picture_suffix", "text_suffix"):
            value = getattr(self, key).strip().lstrip("/").upper()
            if not re.fullmatch(r"[A-Z0-9 ]{0,4}", value):
                raise ValueError("Call sign suffixes allow up to four letters or digits.")
            object.__setattr__(self, key, value)

    def _validate_limits_and_flags(self) -> None:
        """Reject coercion of persisted limits and on/off preferences."""
        for key in ("max_debug_mb", "max_stream_mb"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError("Log size must be a whole number from 1 to 100000 MB.")
        for key in ("send_dprs", "picture_beep", "stream_logging", "incoming_avatars", "ml_enhanced_decoder", "packet_loss_concealment"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be on or off.")
        for key in ("transmit_device", "receive_device"):
            value = getattr(self, key)
            if not isinstance(value, str) or len(value) > 4096 or "\x00" in value:
                raise ValueError("Choose a valid audio device.")
        if (type(self.transmit_gain) not in (int, float) or not math.isfinite(self.transmit_gain)
                or not 0 <= self.transmit_gain <= 6):
            raise ValueError("Microphone gain must be between 0 and 6.")

    def _normalize_location(self) -> None:
        """Validate location inputs before permitting D-PRS transmission."""
        if self.grid:
            grid_center(self.grid)
        if self.latitude or self.longitude:
            latitude, longitude = coordinates(self.latitude, self.longitude)
            for key, value in (("latitude", latitude), ("longitude", longitude)):
                object.__setattr__(self, key, f"{value:.6f}".rstrip("0").rstrip("."))
        if self.send_dprs and self.position is None:
            raise ValueError("Enter a location before enabling D-PRS transmission.")

    @property
    def own_avatar(self) -> AvatarAssignment:
        return AvatarAssignment(self.own_avatar_animal, self.own_avatar_color)

    @property
    def position(self) -> tuple[float, float] | None:
        return coordinates(self.latitude, self.longitude) if self.latitude and self.longitude else None

    def saved(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, document: Mapping[str, Any]) -> Preferences:
        """Keep valid siblings when a saved setting is damaged or outdated."""
        result = cls()
        for field in fields(cls):
            if field.name in ("latitude", "longitude", "send_dprs"):
                continue
            if field.name in document:
                try:
                    result = replace(result, **{field.name: document[field.name]})
                except (ValueError, TypeError, AttributeError):
                    pass
        try:
            result = replace(result, latitude=str(document.get("latitude", "")),
                             longitude=str(document.get("longitude", "")))
            result = replace(result, send_dprs=document.get("send_dprs", False))
        except (ValueError, TypeError):
            pass
        return result
