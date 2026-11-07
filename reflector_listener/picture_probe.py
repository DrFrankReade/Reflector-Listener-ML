"""Opt-in reflector -> RF -> Icom serial picture diagnostic.

Default execution only prepares a deterministic 320x240/50 picture. --live
uses the production sender and records COM receive bytes independently.
The serial port only sends a read-only CI-V frequency query in normal radio
mode, or the documented heartbeat in optional gateway mode. No picture,
PTT command, or tuning command is written to the radio.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import socket
import threading
import time

from PIL import Image, ImageDraw

from .protocols import (ControlKind, MediaKind, Protocol, build_link_packet,
                        build_login_packet, build_poll_packet, build_unlink_packet,
                        parse_control_packet, parse_media_packet, parse_target)
from .rsms1 import RsMs1Receiver, _TransportState, prepare_picture, unescape_bytes
from .icom_civ import CivDataReceiver

POLL = bytes.fromhex("02 02 ff")
FILL = bytes.fromhex("ff ff ff")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def test_pattern(path: Path, callsign: str):
    """Deterministic tile gradients, edges, and labels exercise JPEG escaping."""
    image = Image.new("RGB", (320, 240))
    pixels = image.load()
    for y in range(240):
        for x in range(320):
            pixels[x, y] = (x * 255 // 319, y * 255 // 239,
                            ((x // 16 + y // 16) % 2) * 120 + 50)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 311, 36), fill="black")
    draw.text((16, 17), f"{callsign}  REFLECTOR TEST  320x240 Q50", fill="white")
    for y in range(3, 15, 3):
        for x in range(0, 20, 4):
            draw.text((x * 16 + 1, y * 16 + 2), f"{x},{y}", fill="white")
    image.save(path)


class GatewayParser:
    """Length-framed IC-9700 External Gateway receive packets; FF is fill.

    Verified local reference: ambe-dstar-codex/tools/ic9700_hil.py.
    Payloads may contain FF, so never split a frame on its first FF byte.
    """
    TYPES = {0x03: {0x03, 0x21}, 0x04: {0x23}, 0x10: {0x12}, 0x2C: {0x10}}

    def __init__(self):
        self.buffer = bytearray()
        self.discarded = 0

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while self.buffer:
            length = self.buffer[0]
            if length == 0xFF:
                del self.buffer[0]
                continue
            if length not in self.TYPES:
                del self.buffer[0]
                self.discarded += 1
                continue
            if len(self.buffer) < 2:
                break
            if self.buffer[1] not in self.TYPES[length]:
                del self.buffer[0]
                self.discarded += 1
                continue
            if len(self.buffer) < length + 1:
                break
            if self.buffer[length] != 0xFF:
                del self.buffer[0]
                self.discarded += 1
                continue
            packets.append(bytes(self.buffer[:length + 1]))
            del self.buffer[:length + 1]
        return packets


def picture_records(data: bytes):
    """Independently validate tile checksum and retain exact escaped records."""
    tiles = {}
    invalid = []
    for record in data.split(b"\r"):
        start = record.rfind(b"$$Pic,")
        if start < 0:
            continue
        record = record[start:]
        try:
            _, source, dest, body = record.split(b",", 3)
            control = body[:4]
            address_check = (0x58 + sum(source) + sum(dest) + sum(control[:2])) & 255
            if int(control[2:], 16) != address_check:
                raise ValueError("address checksum")
            raw = unescape_bytes(body[4:])
            if len(raw) < 5 or sum(raw[:-1]) & 255 != raw[-1]:
                raise ValueError("tile checksum")
            key = (source.decode("ascii"), dest.decode("ascii"),
                   int(control[:2], 16), raw[0], raw[1], raw[2])
            tiles[key] = record + b"\r"
        except (ValueError, UnicodeError) as exc:
            invalid.append({"reason": str(exc), "record_hex": record.hex()})
    return tiles, invalid


def compare_tiles(expected: bytes, received: bytes):
    wanted, bad_expected = picture_records(expected)
    actual, bad_received = picture_records(received)
    missing = [list(key[-2:]) for key in wanted if key not in actual]
    different = [list(key[-2:]) for key in wanted if key in actual and wanted[key] != actual[key]]
    return {"expected_tiles": len(wanted), "received_expected_tiles": sum(key in actual for key in wanted),
            "byte_exact_tiles": sum(actual.get(key) == value for key, value in wanted.items()),
            "missing_coordinates": missing, "changed_coordinates": different,
            "invalid_received_records": bad_received, "invalid_expected_records": bad_expected,
            "complete_and_exact": bool(wanted) and not missing and not different and not bad_expected}


def recover_fast_bytes(packets):
    """Independent strict A/B extraction for recorded ordered DV frames.

    Never join a pair across a sequence gap, normal voice, or stream boundary.
    A missing byte-free sync between complete pairs is recorded without
    injecting application damage. This avoids production repair heuristics.
    """
    mask = bytes.fromhex("70 4f 93 40 64 74 6d 30 2b 70 4f 93")
    pending = None
    last = None
    recovered = bytearray()
    gaps = []
    for seq, dv in packets:
        if last is not None and seq != (last + 1) % 21:
            gaps.append({"previous": last, "next": seq})
            # This diagnostic maps the 20-byte A/B profile, with no
            # application bytes in slot zero. Losing only sync between
            # complete pairs must not invent a damaged picture record.
            sync_only = last == 20 and seq == 1 and pending is None
            pending = None
            if not sync_only:
                recovered.extend(b"\r")
        last = seq
        if seq == 0:
            continue
        plain = bytes(a ^ b for a, b in zip(dv, mask))
        if len(plain) != 12 or plain[4] != 2:
            pending = None
            continue
        if plain[9] != 0x83 and plain[9] & 0xF0 in (0x80, 0x90):
            pending = plain
        elif plain[9] == 0x83 and pending is not None:
            length = (pending[9] & 15) + (16 if pending[9] & 0xF0 == 0x90 else 0)
            block = (pending[10:12] + plain[10:12] + pending[:4] + pending[5:9]
                     + plain[:4] + plain[5:9])
            if length <= 20:
                recovered.extend(block[:length])
            pending = None
        else:
            pending = None
    return bytes(recovered), gaps


class SerialCapture:
    def __init__(self, port, baud, mode, directory):
        self.port, self.baud, self.mode, self.directory = port, baud, mode, directory
        self.records = []
        self.packets = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.last_pong = None
        self.cleanup = {}
        self.parser = GatewayParser()
        self.civ = CivDataReceiver()
        self.thread = threading.Thread(target=self.run, name="icom-capture", daemon=True)

    def run(self):
        handle = None
        try:
            import serial  # optional hardware-test dependency, not a GUI dependency
            handle = serial.Serial(port=None, baudrate=self.baud, timeout=.02, write_timeout=1)
            handle.dtr = False
            handle.rts = False
            handle.port = self.port
            handle.open()
            if self.mode == "gateway":
                handle.rts = True
            with (self.directory / "serial.jsonl").open("w", encoding="utf-8") as log:
                def record(direction, data):
                    item = {"time": time.perf_counter(), "direction": direction, "hex": data.hex()}
                    self.records.append(item)
                    log.write(json.dumps(item) + "\n")
                    log.flush()
                def send(data):
                    if handle.write(data) != len(data):
                        raise OSError("short serial heartbeat write")
                    record("tx", data)
                if self.mode == "gateway":
                    send(FILL)
                elif self.mode == "civ":
                    # Read-only frequency query confirms the actual radio,
                    # without tuning it or writing DV transmit data.
                    send(bytes.fromhex("fe fe a2 e0 03 fd"))
                else:
                    self.ready.set()
                next_poll = 0.0
                while not self.stop.is_set():
                    if self.mode == "gateway" and time.perf_counter() >= next_poll:
                        send(POLL)
                        next_poll = time.perf_counter() + .5
                    data = handle.read(max(1, min(4096, handle.in_waiting)))
                    if data:
                        record("rx", data)
                        if self.mode == "civ":
                            self.civ.feed(data)
                            if self.civ.frequency_hz is not None:
                                self.ready.set()
                        if self.mode == "gateway":
                            for packet in self.parser.feed(data):
                                self.packets.append((time.perf_counter(), packet))
                                if packet[:2] == b"\x03\x03":
                                    self.last_pong = time.perf_counter()
                                    self.ready.set()
        except Exception as exc:
            self.error = repr(exc)
            self.ready.set()
        finally:
            if handle is None:
                self.cleanup["not_opened"] = True
                return
            for line in ("rts", "dtr"):
                try:
                    setattr(handle, line, False)
                    self.cleanup[line + "_low"] = True
                except Exception as exc:
                    self.cleanup[line + "_error"] = repr(exc)
            try:
                handle.close()
                self.cleanup["closed"] = not handle.is_open
            except Exception as exc:
                self.cleanup["close_error"] = repr(exc)

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("COM port opened but radio did not reply to " +
                               ("CI-V frequency query" if self.mode == "civ" else "Icom gateway heartbeat"))
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("serial reader did not close")


class SilentAudio:
    generation = 0

    def start(self):
        pass

    retry = start

    def begin_stream(self):
        self.generation += 1
        return self.generation

    cancel_stream = begin_stream

    def submit(self, pcm, generation=None):
        return True


class ReflectorMonitor:
    """Independent D-Plus login; only link/login/poll/unlink go onto its socket.

    Retain received datagrams in memory during TX. A second client is not
    guaranteed to receive the sender's stream; absence is inconclusive.
    """
    def __init__(self, host, target, callsign):
        self.host, self.target, self.callsign = host, parse_target(target), callsign
        if self.target.protocol is not Protocol.DPLUS:
            raise ValueError("independent reflector monitor currently supports D-Plus only")
        self.records = []
        self.local_address = None
        self.error = None
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="reflector-monitor", daemon=True)

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
                handle.settimeout(.1)
                handle.connect((self.host, self.target.port))
                self.local_address = handle.getsockname()
                handle.send(build_link_packet(self.target, self.callsign))
                next_poll = time.monotonic() + 1
                try:
                    while not self.stop.is_set():
                        if time.monotonic() >= next_poll:
                            handle.send(build_poll_packet(self.target, self.callsign))
                            next_poll = time.monotonic() + 1
                        try:
                            data = handle.recv(65535)
                        except socket.timeout:
                            continue
                        self.records.append((time.perf_counter(), data))
                        if len(self.records) > 20000:
                            raise RuntimeError("reflector monitor capture limit reached")
                        control = parse_control_packet(data, Protocol.DPLUS)
                        if control is None:
                            continue
                        if control.kind is ControlKind.LINK:
                            handle.send(build_login_packet(self.callsign))
                        elif control.kind is ControlKind.ACK:
                            self.ready.set()
                        elif control.kind in (ControlKind.NAK, ControlKind.UNLINK):
                            raise ConnectionError("reflector monitor login rejected or disconnected")
                finally:
                    handle.send(build_unlink_packet(self.target, self.callsign))
        except Exception as exc:
            self.error = repr(exc)
            self.ready.set()

    def start(self):
        self.thread.start()
        if not self.ready.wait(8):
            raise RuntimeError("independent reflector monitor did not receive login ACK")
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("reflector monitor did not close")


def compare_reflector_return(expected, tx_records, rx_records):
    """Use RF headers to select our returned stream; never assume echo exists."""
    tx = [p for _, data in tx_records if (p := parse_media_packet(data)) is not None]
    rx = [p for _, data in rx_records if (p := parse_media_packet(data)) is not None]
    headers = [p for p in tx if p.kind is MediaKind.HEADER]
    if not headers:
        return {"status": "unobserved", "reason": "no outgoing RF header"}, b""
    sent_header = headers[0].header
    ids = {p.stream_id for p in rx if p.kind is MediaKind.HEADER
           and p.header.my_call == sent_header.my_call
           and p.header.my_suffix == sent_header.my_suffix
           and p.header.rpt2 == sent_header.rpt2}
    if len(ids) != 1:
        return {"status": "unobserved", "reason": "no unique matching returned stream",
                "matching_stream_ids": sorted(ids), "received_media_packets": len(rx)}, b""
    stream_id = next(iter(ids))
    sent = [(p.sequence, p.dv_payload) for p in tx
            if p.kind is MediaKind.VOICE and p.stream_id == headers[0].stream_id]
    received = [(p.sequence, p.dv_payload) for p in rx
                if p.kind is MediaKind.VOICE and p.stream_id == stream_id]
    recovered, gaps = recover_fast_bytes(received)
    changes = [{"kind": tag, "tx_start": i, "tx_end": j, "rx_start": k, "rx_end": l,
                "sent_sequences": [s for s, _ in sent[i:j]],
                "received_sequences": [s for s, _ in received[k:l]]}
               for tag, i, j, k, l in SequenceMatcher(None, sent, received, autojunk=False).get_opcodes()
               if tag != "equal"]
    result = {"status": "observed", "stream_id": stream_id,
              "sent_stream_id": headers[0].stream_id,
              "sent_voice_frames": len(sent), "received_voice_frames": len(received),
              "frame_order_and_payload_exact": sent == received,
              "frame_differences": changes,
              "application_bytes_exact": expected == recovered,
              "sequence_discontinuities": gaps,
              "tiles": compare_tiles(expected, recovered)}
    return result, recovered


def decode_image(data, directory, name):
    receiver = RsMs1Receiver()
    # The existing byte parser is shared with the on-air decoder. Keep this
    # diagnostic adapter separate from transport ordering and repair logic.
    events = receiver._feed_bytes(_TransportState(strict_text_checksum=True), data)
    pictures = [event for event in events if event.kind == "app_picture"]
    if pictures:
        pictures[-1].fields["image"].save(directory / name)
    return {"picture_events": len(pictures), "warnings": [event.message for event in events
                                                           if event.kind == "app_warning"]}


def analyze(directory, expected, tx_records, capture):
    parsed = [(stamp, parse_media_packet(data)) for stamp, data in tx_records]
    voice = [(stamp, packet) for stamp, packet in parsed if packet is not None and packet.kind is MediaKind.VOICE]
    outgoing, gaps = recover_fast_bytes((p.sequence, p.dv_payload) for _, p in voice)
    (directory / "tx_recovered.bin").write_bytes(outgoing)
    raw_serial = b"".join(bytes.fromhex(r["hex"]) for r in capture.records if r["direction"] == "rx")
    (directory / "serial_rx.bin").write_bytes(raw_serial)
    if capture.mode == "gateway":
        # One stream per header; do not pair Fast Data from unrelated callers.
        chunks, group = [], []
        selected = False
        source = expected.split(b",", 2)[1].decode("ascii").strip()
        radio_gaps = []
        for _, packet in capture.packets:
            if packet[:2] == b"\x2c\x10":
                if group:
                    data, discontinuities = recover_fast_bytes(group)
                    chunks.append(data + b"\r")
                    radio_gaps.extend(discontinuities)
                    group = []
                selected = packet[29:37].decode("ascii", errors="replace").strip() == source
            elif packet[:2] == b"\x10\x12" and selected:
                if not packet[3] & 0x40:
                    group.append((packet[3] & 0x1F, packet[4:16]))
        if group:
            data, discontinuities = recover_fast_bytes(group)
            chunks.append(data)
            radio_gaps.extend(discontinuities)
        incoming = b"".join(chunks)
    elif capture.mode == "civ":
        civ = CivDataReceiver()
        incoming = b"".join(civ.feed(raw_serial))
        radio_gaps = []
    else:
        incoming, radio_gaps = raw_serial, []
    (directory / "radio_recovered.bin").write_bytes(incoming)
    intervals = [(b[0] - a[0]) * 1000 for a, b in zip(voice, voice[1:])]
    result = {"transmit": compare_tiles(expected, outgoing),
              "radio": compare_tiles(expected, incoming),
              "tx_sequence_discontinuities": gaps,
              "radio_sequence_discontinuities": radio_gaps,
              "socket_send_intervals_ms": {"count": len(intervals), "min": min(intervals, default=0),
                  "max": max(intervals, default=0), "over_25ms": sum(x > 25 for x in intervals)},
              "serial": {"rx_bytes": len(raw_serial), "error": capture.error,
                         "packet_types": dict(Counter(p[:2].hex() for _, p in capture.packets)),
                         "framing_discarded_bytes": capture.parser.discarded,
                         "civ": capture.civ.summary() if capture.mode == "civ" else None,
                         "radio_frequency_hz": capture.civ.frequency_hz,
                         "cleanup": capture.cleanup},
              "decoded_radio": decode_image(incoming, directory, "received.png")}
    decode_image(outgoing, directory, "expected-decoded.png")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="actually transmit one picture")
    parser.add_argument("--port", default="COM11")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--expect-frequency", type=int,
                        help="require this CI-V receive frequency in Hz before TX")
    parser.add_argument("--serial-mode", choices=("civ", "gateway", "data"), default="civ")
    parser.add_argument("--callsign", required=True)
    parser.add_argument("--reflector", default="REF012")
    parser.add_argument("--module", default="A")
    parser.add_argument("--local-module", default="T")
    parser.add_argument("--host")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--number", type=int, default=201)
    parser.add_argument("--no-beep", action="store_true")
    parser.add_argument("--reflector-monitor", action="store_true",
                        help="capture a second receive-only D-Plus connection")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.expect_frequency is not None and args.serial_mode != "civ":
        parser.error("--expect-frequency requires --serial-mode civ")
    directory = args.output or Path("logs/picture_probe") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True, exist_ok=False)
    image_path = args.image or directory / "source.png"
    if not args.image:
        test_pattern(image_path, args.callsign)
    prepared = prepare_picture(image_path, args.callsign, "CQCQCQ", "320x240", 50,
                               "Fit / letterbox", args.number, f"{args.callsign} picture path diagnostic")
    (directory / "expected.bin").write_bytes(prepared.payload)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "image_bytes": len(prepared.payload), "tiles": prepared.preset.tile_count,
                "sha256": hashlib.sha256(prepared.payload).hexdigest(), "live": args.live}
    write_json(directory / "manifest.json", manifest)
    decode_image(prepared.payload, directory, "expected-decoded.png")
    print(f"Prepared {len(prepared.payload)} bytes / 300 tiles in {directory.resolve()}", flush=True)
    if not args.live:
        return 0

    from .decoder import AmbeDecoder
    from .hosts import find_exact_host
    from .preferences import Preferences
    from .session import ConnectionConfig, ConnectionState, ReflectorSession

    tx_records = []
    events = []
    cancel = threading.Event()

    class RecordingSession(ReflectorSession):
        def _send(self, sock, packet):
            super()._send(sock, packet)
            tx_records.append((time.perf_counter(), bytes(packet)))

    last_progress = [-1]

    def on_event(event):
        events.append({"time": time.perf_counter(), "kind": event.kind,
                       "message": event.message, "fields": dict(event.fields)})
        if event.kind == "app_tx_progress":
            progress = int(event.fields.get("percent", 0)) // 10
            if progress == last_progress[0]:
                return
            last_progress[0] = progress
        if event.kind in {"state", "network", "error", "warning", "app_tx_start", "app_tx_progress"}:
            print(event.message, flush=True)

    capture = SerialCapture(args.port, args.baud, args.serial_mode, directory)
    session = None
    monitor = None
    decoder = None
    completed = False
    error = None
    try:
        capture.start()
        confirmation = (f"; CI-V radio reply at {capture.civ.frequency_hz} Hz" if args.serial_mode == "civ"
                        else "; Icom PONG confirmed" if args.serial_mode == "gateway"
                        else " (data mode has no PONG)")
        print("Serial capture ready" + confirmation, flush=True)
        if args.expect_frequency is not None and capture.civ.frequency_hz != args.expect_frequency:
            raise RuntimeError(f"radio frequency {capture.civ.frequency_hz} Hz does not match required {args.expect_frequency} Hz")
        decoder = AmbeDecoder()
        session = RecordingSession(decoder, SilentAudio(), on_event, app_receiver=RsMs1Receiver())
        session.set_preferences(Preferences(picture_beep=not args.no_beep, station_message=f"{args.callsign} picture test"[:20]))
        session.enable_packet_logging(directory)
        entry = find_exact_host(args.reflector)
        host = args.host or (entry.ipv4 if entry else None)
        if not host:
            raise ValueError("No reflector address available")
        if args.reflector_monitor:
            monitor = ReflectorMonitor(host, parse_target(args.reflector, args.module), args.callsign)
            monitor.start()
            print(f"Independent reflector monitor linked on {monitor.local_address}", flush=True)
        session.connect(ConnectionConfig(args.callsign, args.reflector, args.module, host,
                                         local_module=args.local_module))
        deadline = time.monotonic() + 25
        while session.state != ConnectionState.LINKED:
            if time.monotonic() > deadline or session.state == ConnectionState.ERROR:
                raise RuntimeError("reflector link did not complete")
            time.sleep(.1)
        idle_since = time.monotonic()
        deadline = idle_since + 60
        while time.monotonic() - idle_since < 3:
            if session.receiving:
                idle_since = time.monotonic()
            if time.monotonic() > deadline:
                raise RuntimeError("reflector remained busy; no picture sent")
            time.sleep(.1)
        if capture.error:
            raise RuntimeError(capture.error)
        if monitor is not None and monitor.error:
            raise RuntimeError(monitor.error)
        if args.serial_mode == "gateway" and (capture.last_pong is None or time.perf_counter() - capture.last_pong > 3):
            raise RuntimeError("Icom heartbeat lost before transmission")

        def watch_capture():
            deadline = time.monotonic() + 180
            while not cancel.wait(.1):
                heartbeat_lost = (args.serial_mode == "gateway" and
                    (capture.last_pong is None or time.perf_counter() - capture.last_pong > 3))
                monitor_error = monitor.error if monitor is not None else None
                if capture.error or monitor_error or heartbeat_lost or time.monotonic() > deadline:
                    print(f"Stopping TX: {capture.error or monitor_error or ('heartbeat lost' if heartbeat_lost else '180s limit')}", flush=True)
                    cancel.set()
                    return

        watchdog = threading.Thread(target=watch_capture, daemon=True)
        watchdog.start()
        try:
            completed = session.transmit_application(prepared.payload, label="picture diagnostic",
                                                      metadata={"kind": "picture", "number": args.number},
                                                      cancel_event=cancel)
        finally:
            cancel.set()
            watchdog.join(1)
        time.sleep(5)  # collect the repeater/radio tail before closing capture
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc) or type(exc).__name__
        print(f"Diagnostic stopped: {error}", flush=True)
    finally:
        cancel.set()
        try:
            if session is not None:
                session.disconnect()
                session.disable_packet_logging()
        finally:
            try:
                capture.close()
            finally:
                try:
                    if monitor is not None:
                        monitor.close()
                finally:
                    if decoder is not None:
                        decoder.close()
        # Preserve hardware failures before analysis so a report bug cannot
        # erase evidence of a disconnect or a failed cleanup operation.
        write_json(directory / "serial_status.json", {"error": capture.error, "cleanup": capture.cleanup})
        with (directory / "tx_datagrams.jsonl").open("w", encoding="utf-8") as out:
            for stamp, data in tx_records:
                out.write(json.dumps({"time": stamp, "hex": data.hex()}) + "\n")
        write_json(directory / "events.json", events)
        result = analyze(directory, prepared.payload, tx_records, capture)
        if monitor is not None:
            with (directory / "reflector_rx_datagrams.jsonl").open("w", encoding="utf-8") as out:
                for stamp, data in monitor.records:
                    out.write(json.dumps({"time": stamp, "hex": data.hex()}) + "\n")
            comparison, incoming = compare_reflector_return(prepared.payload, tx_records, monitor.records)
            comparison.update(error=monitor.error, local_address=monitor.local_address)
            result["reflector_return"] = comparison
            (directory / "reflector_recovered.bin").write_bytes(incoming)
            decode_image(incoming, directory, "reflector-received.png")
        result.update(completed=completed, error=error)
        write_json(directory / "summary.json", result)
        print(json.dumps({"completed": completed, "error": error,
                          "radio_tiles": result["radio"]["byte_exact_tiles"],
                          "serial_rx_bytes": result["serial"]["rx_bytes"],
                          "summary": str((directory / "summary.json").resolve())}), flush=True)
    return 0 if completed and result["radio"]["complete_and_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
