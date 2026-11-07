"""Load and query the bundled D-Star reflector host directory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Mapping

from .protocols import (
    Protocol,
    ReflectorTarget,
    parse_target,
)


HOSTS_FILENAME = "DStar_Hosts.json"
_DIRECTORY_NAME_RE = re.compile(r"^[A-Z0-9]{1,8}$")


class HostDirectoryError(ValueError):
    """The host directory exists but does not have the expected schema."""


class HostNotFoundError(LookupError):
    """An exact reflector name is absent from the loaded directory."""


@dataclass(frozen=True, slots=True)
class HostEntry:
    name: str
    reflector_type: str
    ipv4: str

    def __post_init__(self) -> None:
        name = self.name.strip().upper()
        reflector_type = self.reflector_type.strip().upper()
        if not _DIRECTORY_NAME_RE.fullmatch(name):
            raise HostDirectoryError(f"invalid directory name {self.name!r}")
        if reflector_type not in {"REF", "XRF", "DCS"}:
            raise HostDirectoryError(f"invalid reflector type {self.reflector_type!r}")
        # Entries named XLXnnn intentionally carry reflector_type DCS.  Other
        # non-prefixed REF records in the upstream directory are D-Plus
        # gateways and remain indexable, though parse_target() will not offer
        # them as reflector choices in this listener.
        if name.startswith("REF") and reflector_type != "REF":
            raise HostDirectoryError(f"{name} must have reflector type REF")
        if name.startswith("XRF") and reflector_type != "XRF":
            raise HostDirectoryError(f"{name} must have reflector type XRF")
        if name.startswith(("DCS", "XLX")) and reflector_type != "DCS":
            raise HostDirectoryError(f"{name} must have reflector type DCS")
        try:
            address = str(ipaddress.IPv4Address(self.ipv4.strip()))
        except ipaddress.AddressValueError as exc:
            raise HostDirectoryError(f"{name} has invalid IPv4 address {self.ipv4!r}") from exc

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reflector_type", reflector_type)
        object.__setattr__(self, "ipv4", address)

    @property
    def protocol(self) -> Protocol:
        return {
            "REF": Protocol.DPLUS,
            "XRF": Protocol.DEXTRA,
            "DCS": Protocol.DCS,
        }[self.reflector_type]

    def target(self, module: str = "C") -> ReflectorTarget:
        return parse_target(self.name, module)


@dataclass(frozen=True, slots=True)
class ResolvedHost:
    target: ReflectorTarget
    host: str
    port: int
    source: Path

    @property
    def endpoint(self) -> tuple[str, int]:
        return self.host, self.port


class HostDirectory:
    """An immutable, exact-name index over ``DStar_Hosts.json``."""

    def __init__(self, entries: Iterable[HostEntry], source: Path) -> None:
        ordered = tuple(entries)
        by_name: dict[str, HostEntry] = {}
        for entry in ordered:
            if entry.name in by_name:
                raise HostDirectoryError(f"duplicate reflector entry {entry.name}")
            by_name[entry.name] = entry
        if not ordered:
            raise HostDirectoryError("host directory contains no reflector entries")
        self._entries = ordered
        self._by_name: Mapping[str, HostEntry] = by_name
        self.source = source.resolve()

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "HostDirectory":
        source = (
            Path(path).expanduser().resolve()
            if path is not None
            else discover_hosts_path()
        )
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                document = json.load(handle)
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HostDirectoryError(f"unable to read host directory {source}: {exc}") from exc

        if not isinstance(document, dict):
            raise HostDirectoryError("host directory root must be a JSON object")
        records = document.get("reflectors")
        if not isinstance(records, list):
            raise HostDirectoryError("host directory must contain a 'reflectors' array")

        entries: list[HostEntry] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise HostDirectoryError(f"reflectors[{index}] must be an object")
            try:
                name = record["name"]
                reflector_type = record["reflector_type"]
                ipv4 = record["ipv4"]
            except KeyError as exc:
                raise HostDirectoryError(
                    f"reflectors[{index}] is missing {exc.args[0]!r}"
                ) from exc
            if not all(isinstance(value, str) for value in (name, reflector_type, ipv4)):
                raise HostDirectoryError(
                    f"reflectors[{index}] name, reflector_type, and ipv4 must be strings"
                )
            try:
                entries.append(HostEntry(name, reflector_type, ipv4))
            except HostDirectoryError as exc:
                raise HostDirectoryError(f"reflectors[{index}]: {exc}") from exc
        return cls(entries, source)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[HostEntry]:
        return iter(self._entries)

    @property
    def entries(self) -> tuple[HostEntry, ...]:
        return self._entries

    def find_exact(self, name: str | ReflectorTarget) -> HostEntry | None:
        """Find one complete six-character name, case-insensitively.

        Whitespace surrounding a string is ignored, but embedded modules,
        prefixes, and longer values do not match.
        """

        if isinstance(name, ReflectorTarget):
            key = name.name
        elif isinstance(name, str):
            key = name.strip().upper()
        else:
            raise TypeError("reflector name must be text or ReflectorTarget")
        return self._by_name.get(key)

    def require_exact(self, name: str | ReflectorTarget) -> HostEntry:
        entry = self.find_exact(name)
        if entry is None:
            raise HostNotFoundError(f"reflector {name!s} is not in {self.source}")
        return entry

    def resolve_target(
        self, target: str | ReflectorTarget, module: str | None = None
    ) -> ResolvedHost:
        selected = parse_target(target, module)
        entry = self.require_exact(selected)
        return ResolvedHost(selected, entry.ipv4, selected.port, self.source)


def host_path_candidates() -> tuple[Path, ...]:
    """Return the single portable directory location."""

    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        return (base_dir / HOSTS_FILENAME,)
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent
    return (project_dir / HOSTS_FILENAME,)


def discover_hosts_path() -> Path:
    """Locate the bundled host directory or raise ``FileNotFoundError``."""

    candidates = host_path_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(item.resolve(strict=False)) for item in candidates)
    raise FileNotFoundError(f"{HOSTS_FILENAME} was not found; searched:\n  {searched}")


@lru_cache(maxsize=1)
def load_default_host_directory() -> HostDirectory:
    """Load and cache the discovered directory for normal application use."""

    return HostDirectory.load()


def find_exact_host(
    name: str | ReflectorTarget,
    path: str | os.PathLike[str] | None = None,
) -> HostEntry | None:
    directory = HostDirectory.load(path) if path is not None else load_default_host_directory()
    return directory.find_exact(name)


__all__ = [
    "HOSTS_FILENAME",
    "HostDirectory",
    "HostDirectoryError",
    "HostEntry",
    "HostNotFoundError",
    "ResolvedHost",
    "discover_hosts_path",
    "find_exact_host",
    "host_path_candidates",
    "load_default_host_directory",
]
