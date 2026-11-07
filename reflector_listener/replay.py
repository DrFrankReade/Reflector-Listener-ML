"""Independent, seekable archive playback with optional live-audio arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from .receive_audio import ReceiveAudioPlayer
from .decoder import AmbeDecoder, BitOrder, DEFAULT_GAIN
from .protocols import parse_media_packet
from .stream_log import BinaryStreamLog, Recording, RecordingGroup, RecordRef, voice_index


@dataclass(frozen=True)
class PlaybackState:
    token: str = ""
    position: float = 0.0
    duration: float = 0.0
    playing: bool = False
    loading: bool = False
    error: str = ""


class ReplayController:
    """One replay at a time, on its own decoder, queue, and sound-device handle.

    Commands never decode or perform disk I/O on Tk's thread. A revision token
    prevents a slow load/seek from submitting stale audio after a newer command.
    """

    def __init__(self, archive: BinaryStreamLog, *, decoder_factory: Callable = AmbeDecoder,
                 player_factory: Callable = ReceiveAudioPlayer, on_playing: Callable[[bool], None] | None = None,
                 on_warning: Callable[[str], None] | None = None) -> None:
        self.archive = archive
        self._decoder_factory = decoder_factory
        self._player_factory = player_factory
        self._on_playing = on_playing or (lambda playing: None)
        self._on_warning = on_warning or (lambda message: None)
        self._ml_enabled = False
        self._announced_playing = False
        self._condition = threading.Condition()
        self._clip: Recording | RecordingGroup | None = None
        self._state = PlaybackState()
        self._revision = 0
        self._closed = False
        self._gain = DEFAULT_GAIN
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> PlaybackState:
        with self._condition:
            return self._state

    def set_gain(self, gain: float) -> None:
        with self._condition:
            self._gain = gain

    def set_ml_enabled(self, enabled: bool) -> None:
        """Choose the receive mode for the next playback or seek."""
        with self._condition:
            self._ml_enabled = bool(enabled)

    def _publish_playing(self) -> None:
        """Called with the condition held; callback must not call back into replay."""
        playing = self._state.playing and not self._closed
        if playing != self._announced_playing:
            self._announced_playing = playing
            self._on_playing(playing)

    def play(self, clip: Recording | RecordingGroup, position: float = 0.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._clip = clip
            self._state = PlaybackState(clip.token, max(0.0, position), playing=True, loading=True)
            self._publish_playing()
            self._revision += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="archive-playback", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            s = self._state
            self._state = PlaybackState(s.token, s.position, s.duration)
            self._publish_playing()
            self._revision += 1
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            s = self._state
            self._state = PlaybackState(s.token, 0.0, s.duration)
            self._publish_playing()
            self._revision += 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._publish_playing()
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        """Own the playback devices and process one revision at a time."""
        decoder = player = None
        loaded_token = ""
        index: dict[int, RecordRef] = {}
        try:
            decoder = self._decoder_factory(gain=self._gain)
            options = dict(max_device_frames=3, max_queued_frames=8, on_error=self._audio_error)
            if self._player_factory is ReceiveAudioPlayer:
                options["on_warning"] = self._on_warning
            player = self._player_factory(**options)
            while True:
                command = self._wait_for_playback(player)
                if command is None:
                    return
                clip, state, revision = command
                try:
                    self._check_recording_available(clip)
                    if loaded_token != clip.token:
                        loaded_index = voice_index(self.archive, clip, lambda: self._closed or self._revision != revision)
                        with self._condition:
                            if self._closed or self._revision != revision:
                                continue
                            # Keep the token and index paired. A cancelled load
                            # must not overwrite the previous recording's cache.
                            index, loaded_token = loaded_index, clip.token
                    self._play_recording(decoder, player, clip, index, state.position, revision)
                except (OSError, ValueError, RuntimeError) as exc:
                    self._recording_error(clip, revision, str(exc))
                    player.cancel_stream()
        except Exception as exc:
            self._audio_error(str(exc))
        finally:
            self._close_playback_devices(player, decoder)

    def _wait_for_playback(self, player) -> tuple[Recording | RecordingGroup, PlaybackState, int] | None:
        """Snapshot a play/seek command without holding the lock during I/O."""
        with self._condition:
            while not self._closed and not self._state.playing:
                player.cancel_stream()
                self._condition.wait()
            if self._closed:
                return None
            assert self._clip is not None  # Only play() publishes playing=True.
            return self._clip, self._state, self._revision

    @staticmethod
    def _check_recording_available(clip: Recording | RecordingGroup) -> None:
        if clip.expired:
            raise FileNotFoundError("Recording expired from the rolling stream log.")

    def _play_recording(self, decoder, player, clip: Recording | RecordingGroup,
                        index: dict[int, RecordRef], position: float, revision: int) -> None:
        """Warm the codec for a seek, then submit frames while this command owns playback."""
        length = max(index, default=-1) + 1
        if not length:
            raise ValueError("No recorded voice frames are available.")
        cursor = min(int(position / 0.02), length - 1)
        if set_ml_enabled := getattr(player, "set_ml_enabled", None):
            with self._condition:
                set_ml_enabled(self._ml_enabled)
        generation = player.begin_stream()
        decoder.begin_stream(1)
        # Warm codec state before the seek point, without playing it.
        for slot in range(max(0, cursor - 10), cursor):
            self._decode(decoder, clip, index.get(slot))
        deadline = time.monotonic()
        while cursor < length:
            self._check_recording_available(clip)
            with self._condition:
                if self._closed or self._revision != revision:
                    break
                decoder.set_gain(self._gain)
            result = self._decode(decoder, clip, index.get(cursor))
            with self._condition:
                if self._closed or self._revision != revision:
                    break
                if submit_decoded := getattr(player, "submit_decoded", None):
                    submit_decoded(result, generation)
                else:
                    player.submit(result.pcm, generation)
                cursor += 1
                self._state = PlaybackState(clip.token, cursor * 0.02, length * 0.02, True)
                deadline += 0.02
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
        if cursor >= length:
            if finish := getattr(player, "finish_stream", None):
                drained = finish(generation)
                with self._condition:
                    while not drained.is_set():
                        if self._closed or self._revision != revision:
                            player.cancel_stream()
                            return
                        self._condition.wait(timeout=0.02)
            self._finish_recording(clip, revision, length * 0.02)

    def _finish_recording(self, clip: Recording | RecordingGroup, revision: int, duration: float) -> None:
        """Allow queued audio to drain before publishing completion for this command."""
        with self._condition:
            if self._revision == revision:
                self._condition.wait(timeout=0.08)
                if self._revision == revision:
                    self._state = PlaybackState(clip.token, duration, duration)
                    self._publish_playing()

    def _recording_error(self, clip: Recording | RecordingGroup, revision: int, message: str) -> None:
        with self._condition:
            if self._revision == revision:
                self._state = PlaybackState(clip.token, error=message)
                self._publish_playing()

    def _close_playback_devices(self, player, decoder) -> None:
        """Release both devices and worker ownership even if one close fails."""
        try:
            if player:
                player.close()
        finally:
            try:
                if decoder:
                    decoder.close()
            finally:
                with self._condition:
                    self._thread = None

    def _decode(self, decoder, clip, ref):
        if ref is None:
            return decoder.conceal_frame()
        packet = parse_media_packet(self.archive.read(ref), clip.protocol)
        if packet is None or packet.ambe is None:
            raise ValueError("Invalid archived voice frame")
        return decoder.decode_frame(packet.ambe, BitOrder.LSB_FIRST)

    def _audio_error(self, message: str) -> None:
        with self._condition:
            self._state = PlaybackState(self._state.token, error=message)
            self._publish_playing()
            self._revision += 1
            self._condition.notify_all()
