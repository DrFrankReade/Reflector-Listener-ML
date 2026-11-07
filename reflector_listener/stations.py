"""Stable station identities and local avatar assignments; no UI or radio code."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random


ANIMALS = ("elephant", "giraffe", "hippo", "monkey", "panda", "parrot", "penguin", "pig", "rabbit", "snake")
COLORS = (
    "#76c78d", "#5C85A1", "#8e4fc1", "#eb8b62", "#dabb54",
    "#3e9086", "#d86384", "#8aa44c", "#5677be", "#98763c",
)
logger = logging.getLogger(__name__)


def station_key(callsign: str) -> str:
    """Ignore RF module/suffix metadata, preserving portable call signs."""
    return (callsign.strip().upper().split() or [""])[0]


@dataclass(frozen=True)
class AvatarAssignment:
    animal: str
    color: str


DEFAULT_AVATAR = AvatarAssignment("panda", COLORS[0])


class AvatarAssignments:
    """Keep assignments until shuffled, save atomically, and avoid repeats.

    Existing identities always win over duplicate avoidance. Merely loading the
    app or disabling avatars never creates or changes the assignments file.
    """

    def __init__(self, path: Path, *, rng=None) -> None:
        self.path = path
        self._rng = rng or random.SystemRandom()
        self._assignments: dict[str, AvatarAssignment] = {}
        self._seen: set[str] = set()
        self._can_save = True
        self._save_error_reported = False
        self._load()

    def _load(self) -> None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("version") != 1:
                raise ValueError("unsupported avatar assignments format")
            entries = document["assignments"]
            if not isinstance(entries, dict):
                raise ValueError("invalid avatar assignments")
            for call, value in entries.items():
                key = station_key(call)
                if (key and key != "UNKNOWN" and isinstance(value, dict)
                        and value.get("animal") in ANIMALS and value.get("color") in COLORS):
                    self._assignments[key] = AvatarAssignment(value["animal"], value["color"])
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Preserve an unreadable/corrupt file rather than overwriting identities.
            self._can_save = False
            logger.warning("Cannot load avatar assignments; keeping the file unchanged: %s", exc)

    def get(self, callsign: str) -> AvatarAssignment | None:
        key = station_key(callsign)
        if not key or key == "UNKNOWN":
            return None
        if key not in self._assignments:
            self._assignments[key] = self._choose(key)
            self._save()
        self._seen.add(key)
        return self._assignments[key]

    def shuffle(self, callsign: str) -> AvatarAssignment | None:
        """Persist a different animal for this station, preferring unused ones."""
        key = station_key(callsign)
        if not key or key == "UNKNOWN":
            return None
        previous = self._assignments.get(key)
        self._assignments[key] = self._choose(key, previous)
        self._seen.add(key)
        self._save()
        return self._assignments[key]

    def _choose(self, key, previous=None):
        visible = [self._assignments[call] for call in self._seen if call != key]
        animals = Counter(item.animal for item in visible)
        candidates = [animal for animal in ANIMALS if previous is None or animal != previous.animal]
        minimum = min(animals[animal] for animal in candidates)
        animal = self._rng.choice([animal for animal in candidates if animals[animal] == minimum])
        colors = Counter(item.color for item in visible if item.animal == animal)
        minimum = min(colors[color] for color in COLORS)
        color = self._rng.choice([color for color in COLORS if colors[color] == minimum])
        return AvatarAssignment(animal, color)

    def _save(self) -> None:
        if not self._can_save:
            return
        temporary = self.path.with_suffix(".json.tmp")
        try:
            document = {"version": 1, "assignments": {
                call: asdict(value) for call, value in sorted(self._assignments.items())
            }}
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
            self._save_error_reported = False
        except OSError as exc:
            if not self._save_error_reported:
                logger.warning("Avatar assignments are only available for this run; save failed: %s", exc)
                self._save_error_reported = True

