"""Coordinate validation, Maidenhead conversion, and great-circle navigation."""

from __future__ import annotations

import math
import re


def coordinates(latitude: object, longitude: object) -> tuple[float, float]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (ValueError, TypeError) as exc:
        raise ValueError("Enter both latitude and longitude in decimal degrees.") from exc
    if not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    return lat, lon


def grid_center(grid: str) -> tuple[float, float]:
    """Return the center, rather than a corner, of a 4/6-character locator."""
    grid = grid.strip().upper()
    if not re.fullmatch(r"[A-R]{2}[0-9]{2}(?:[A-X]{2})?", grid):
        raise ValueError("Use a 4 or 6 character Maidenhead locator, such as CM87 or CM87SS.")
    lon = -180 + (ord(grid[0]) - 65) * 20 + int(grid[2]) * 2
    lat = -90 + (ord(grid[1]) - 65) * 10 + int(grid[3])
    if len(grid) == 6:
        lon += (ord(grid[4]) - 65 + 0.5) / 12
        lat += (ord(grid[5]) - 65 + 0.5) / 24
    else:
        lon += 1
        lat += 0.5
    return lat, lon


def distance_bearing(origin: tuple[float, float], destination: tuple[float, float]) -> tuple[float, float]:
    """Return kilometers and initial bearing clockwise from true north."""
    lat1, lon1 = map(math.radians, coordinates(*origin))
    lat2, lon2 = map(math.radians, coordinates(*destination))
    delta = lon2 - lon1
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta / 2) ** 2
    km = 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))
    bearing = math.degrees(math.atan2(math.sin(delta) * math.cos(lat2),
                                    math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta))) % 360
    return km, bearing


def distance_text(origin: tuple[float, float] | None, remote: object, units: str) -> str:
    """Ignore missing, malformed and zero/zero remote fixes."""
    if origin is None:
        return ""
    try:
        lat, lon = coordinates(*str(remote).replace(",", " ").split())
        if lat == 0 and lon == 0:
            return ""
        km, bearing = distance_bearing(origin, (lat, lon))
    except (ValueError, TypeError):
        return ""
    value = km / 1.609344 if units == "miles" else km
    label = "miles" if units == "miles" else "km"
    return f"{value:.0f} {label} {round(bearing) % 360:03d}\N{DEGREE SIGN}"
