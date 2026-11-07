"""Replay a raw UDP receive log without sockets, audio, or the GUI."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from reflector_listener.protocols import MediaKind, Protocol, parse_control_packet, parse_media_packet
from reflector_listener.rsms1 import RsMs1Receiver


_RAW_LINE = re.compile(
    r"^(?P<timestamp>\S+) .*\[RX-RAW\] (?P<protocol>\S+) "
    r"packet=(?P<packet>\d+) connection=(?P<connection>\d+) .*"
    r"length=(?P<length>\d+) data=(?P<data>[0-9a-fA-F]*)$"
)


class _Diagnostics:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.entries: list[dict[str, Any]] = []

    def log_gap(self, direction, stream_id, expected, actual, note="") -> None:
        self.entries.append(dict(self.context, kind=direction, stream_id=f"{stream_id:04X}",
                                 expected=expected, actual=actual, message=note))

    def log_event(self, event_type, message, **fields) -> None:
        # Application events are collected from the receiver's return value;
        # retain transport diagnostics here without double-counting warnings.
        if event_type not in {"app_picture", "app_text", "app_gps", "app_warning"}:
            self.entries.append(dict(self.context, kind=event_type, message=message, fields=fields))


def replay_capture(raw_log: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Replay complete logged datagrams through the production parsers.

    Sequence discontinuities describe observations, not proven network loss:
    modulo-21 numbering cannot distinguish every duplicate, reorder, and gap.
    Diagnostic line numbers identify when the parser reported an issue; a
    damaged application frame can begin in an earlier UDP datagram.
    """
    source = Path(raw_log).resolve()
    destination = Path(output_dir).resolve()
    report_path = destination / "summary.json"
    if source == report_path:
        raise ValueError("output would overwrite the source capture")
    diagnostics = _Diagnostics()
    receiver = RsMs1Receiver(diagnostics)
    counts: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    event_counts: Counter[str] = Counter()
    streams: list[dict[str, Any]] = []
    active: dict[tuple[str, int], dict[str, Any]] = {}
    last_sequences: dict[tuple[str, int], tuple[int, int]] = {}
    pictures: dict[tuple[int, str], dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    connection: int | None = None
    datagrams = 0

    def consume(events) -> None:
        for event in events:
            event_counts[event.kind] += 1
            if event.kind == "app_warning":
                warnings.append(dict(diagnostics.context, kind=event.kind, message=event.message))
            elif event.kind == "app_text":
                texts.append(dict(diagnostics.context, **event.fields))
            elif event.kind == "app_picture":
                fields = event.fields
                key = (connection, fields["picture_id"])
                picture = pictures.setdefault(key, {"coordinates": set()})
                if fields["received"] < picture.get("received", 0):
                    picture["coordinates"].clear()
                picture.update({k: v for k, v in fields.items() if k != "image"})
                picture["connection"] = connection
                picture["reported_at"] = dict(diagnostics.context)
                picture["image"] = fields["image"]
                picture["coordinates"].add((fields["tile_x"], fields["tile_y"]))

    def finish_all(reason: str) -> None:
        for (_, stream_id), stream in list(active.items()):
            diagnostics.context["stream_id"] = f"{stream_id:04X}"
            consume(receiver.end_stream(stream_id))
            stream["end_reason"] = reason
        active.clear()
        last_sequences.clear()

    with source.open("r", encoding="utf-8") as capture:
        for line_number, line in enumerate(capture, 1):
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            match = _RAW_LINE.fullmatch(line)
            if match is None:
                warnings.append({"raw_line": line_number, "kind": "invalid_log_line",
                                 "message": "Line does not match the raw receive log format"})
                continue
            datagrams += 1
            try:
                payload = bytes.fromhex(match["data"])
                protocol = Protocol(match["protocol"])
            except ValueError as exc:
                warnings.append({"raw_line": line_number, "kind": "invalid_log_line", "message": str(exc)})
                continue
            incoming_connection = int(match["connection"])
            if connection is not None and incoming_connection != connection:
                finish_all("connection changed")
                receiver.reset()
            connection = incoming_connection
            diagnostics.context = {"raw_line": line_number, "timestamp": match["timestamp"],
                                   "packet": int(match["packet"]), "connection": connection}
            lengths[len(payload)] += 1
            if len(payload) != int(match["length"]):
                warnings.append(dict(diagnostics.context, kind="invalid_log_line",
                                     message="Logged length differs from captured bytes; datagram skipped"))
                continue
            try:
                packet = parse_media_packet(payload, protocol)
                control = parse_control_packet(payload, protocol) if packet is None else None
            except ValueError as exc:
                counts["parse_error"] += 1
                warnings.append(dict(diagnostics.context, kind="parse_error", message=str(exc)))
                continue
            if packet is None:
                counts["control" if control is not None else "unknown"] += 1
                continue
            counts[packet.kind.value] += 1
            key = (protocol.value, packet.stream_id)
            diagnostics.context["stream_id"] = f"{packet.stream_id:04X}"
            if key not in active:
                stream = {"stream_id": f"{packet.stream_id:04X}", "protocol": protocol.value,
                          "connection": connection, "first_raw_line": line_number,
                          "start_timestamp": match["timestamp"], "voice_datagrams": 0,
                          "header_datagrams": 0, "sequence_discontinuities": []}
                streams.append(stream)
                active[key] = stream
                receiver.start_stream(packet.stream_id)
            stream = active[key]
            stream["last_raw_line"] = line_number
            stream["end_timestamp"] = match["timestamp"]
            if packet.header is not None:
                stream["callsign"] = packet.header.my_call
                stream["header_crc_valid"] = packet.header.crc_valid
            if packet.kind is MediaKind.HEADER:
                stream["header_datagrams"] += 1
                continue
            if packet.kind is MediaKind.END:
                consume(receiver.end_stream(packet.stream_id))
                stream["end_reason"] = "end packet"
                active.pop(key)
                last_sequences.pop(key, None)
                continue
            stream["voice_datagrams"] += 1
            previous = last_sequences.get(key)
            sequence = packet.sequence
            if previous is not None:
                prior_sequence, prior_line = previous
                delta = (sequence - prior_sequence) % 21
                if delta != 1:
                    stream["sequence_discontinuities"].append({
                        "previous_raw_line": prior_line, "raw_line": line_number,
                        "previous_sequence": prior_sequence, "sequence": sequence,
                        "forward_distance_mod21": delta,
                        "observation": "repeated sequence" if delta == 0 else "nonconsecutive sequence",
                    })
            last_sequences[key] = (sequence, line_number)
            _, events = receiver.feed(packet.stream_id, sequence, packet.dv_payload)
            consume(events)
    finish_all("capture ended")

    destination.mkdir(parents=True, exist_ok=True)
    picture_summaries = []
    for index, picture in enumerate(pictures.values(), 1):
        image = picture.pop("image")
        coordinates = picture.pop("coordinates")
        image_path = destination / f"picture_{index:02d}.png"
        if image_path == source:
            raise ValueError("output would overwrite the source capture")
        image.save(image_path)
        tiles_x = picture["width"] // 16
        tiles_y = picture["total"] // tiles_x
        picture["missing_coordinates"] = [(x, y) for y in range(tiles_y)
                                          for x in range(tiles_x) if (x, y) not in coordinates]
        picture["image_path"] = str(image_path)
        picture_summaries.append(picture)
    with source.open("rb") as capture:
        digest = hashlib.file_digest(capture, "sha256").hexdigest()
    report = {"source_file": str(source), "source_sha256": digest,
              "datagrams": datagrams, "datagram_lengths": dict(lengths), "packet_kinds": dict(counts),
              "event_counts": dict(event_counts), "streams": streams, "pictures": picture_summaries,
              "warnings": warnings, "texts": texts, "diagnostics": diagnostics.entries,
              "notes": ["Sequence discontinuities alone do not prove network packet loss.",
                        "Diagnostic raw_line identifies when an issue was reported, not necessarily its first damaged byte.",
                        "Replay uses the production receiver; input bytes are never changed."]}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay_capture(args.raw_log, args.output_dir)
    print(f"Replayed {report['datagrams']} datagrams; {len(report['warnings'])} warnings.")
    for picture in report["pictures"]:
        print(f"Picture {picture['number']}: {picture['received']}/{picture['total']} tiles; {picture['image_path']}")
    print(f"Report: {(args.output_dir / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
