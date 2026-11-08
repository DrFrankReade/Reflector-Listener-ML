# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Isolated asymmetric future-context experiment for frozen Pilot5C; no decoder/audio device I/O.

Caller supplies successive stock-decoder observations. The harness deliberately
does not reset the neural noise timeline at incoming chunk boundaries. It tests
the model's actual block/context contract, while leaving hardware capture,
packet concealment, and device buffering to separate integration qualification.
"""
from __future__ import annotations

import time
from collections import deque
import numpy as np
import torch

from .observations import STATE_DIM, normalize_features
from ._model_v2 import status_features
from ._model_v4 import BLOCK_FRAMES, CONTEXT_FRAMES


class FutureContextStream:
    """Receive arbitrary frame chunks, emit fixed blocks once future context exists.

    No model-owned recurrent state or decoder state is reset here. Every input
    frame remains on its original timeline. ``finish`` declares the real end
    and flushes every final sample with explicit unavailable-future padding.
    """

    def __init__(self, model, normalization, state_normalization, *, device='cpu', seed=6110704, future_frames=18):
        if future_frames not in (36,18,9):raise ValueError('Use36,18,9 future codec frames')
        self.past_frames=36
        self.future_frames=future_frames
        self.emissions=deque(maxlen=256)
        self.flushing=False
        self.model = model.eval()
        self.device = device
        self.mean = np.asarray(normalization['mean'], np.float32)
        self.std = np.asarray(normalization['std'], np.float32)
        if self.mean.shape != (74,) or self.std.shape != (74,) or np.any(self.std <= 0):
            raise ValueError('Invalid frozen compact-feature normalization')
        self.state_normalization = state_normalization
        self.rng = np.random.default_rng(seed)
        self.buffer = {'pcm8': np.empty((0, 160), np.float32), 'features': np.empty((0, 74), np.float32),
                       'flags': np.empty((0,), np.int64), 'errors': np.empty((0,), np.int32),
                       'state': np.empty((0, STATE_DIM), np.float32), 'noise': np.empty((0, 960), np.float32)}
        self.base = self.received = self.next_frame = self.emitted_samples = self.max_buffered_frames = 0
        self.finished = False
        self.processing_seconds = deque(maxlen=256)
        self.total_blocks = 0
        self.maximum_processing_seconds = 0.
        self._pending_processing = 0.

    def _append(self, inputs, state):
        started = time.perf_counter()
        count = len(inputs['pcm8'])
        normalized = np.clip((np.asarray(inputs['features'], np.float32) - self.mean) / self.std, -12, 12)
        additions = {'pcm8': np.asarray(inputs['pcm8'], np.float32), 'features': normalized,
                     'flags': np.asarray(inputs['flags'], np.int64), 'errors': np.asarray(inputs['errors'], np.int32),
                     'state': normalize_features(state, self.state_normalization),
                     'noise': self.rng.standard_normal(count * 960).astype(np.float32).reshape(count, 960)}
        for name, value in additions.items():
            self.buffer[name] = np.concatenate((self.buffer[name], value))
        self.received += count
        self.max_buffered_frames = max(self.max_buffered_frames, len(self.buffer['pcm8']))
        if len(self.buffer['pcm8']) > self.past_frames + self.future_frames + BLOCK_FRAMES:
            raise RuntimeError('Streaming retained more than the declared context/block bound')
        self._pending_processing += time.perf_counter() - started

    def _render(self, stop):
        started = time.perf_counter()
        start = self.next_frame
        indices = np.arange(start - self.past_frames, stop + self.future_frames)
        present = (indices >= 0) & (indices < self.received)
        offsets = indices[present] - self.base
        if np.any(offsets < 0) or np.any(offsets >= len(self.buffer['pcm8'])):
            raise RuntimeError('Required past/future decoder context was discarded')

        def tensor(name):
            source = self.buffer[name]
            value = np.zeros((len(indices),) + source.shape[1:], source.dtype)
            value[present] = source[offsets]
            return torch.from_numpy(value[None]).to(self.device)

        flags, errors = tensor('flags'), tensor('errors')
        status = status_features(flags, errors, torch.from_numpy(present[None]).to(self.device))
        arguments = (tensor('pcm8'), tensor('features'), status, tensor('noise').flatten(1), tensor('state'))
        with torch.inference_mode():
            output = self.model(*arguments)[0, self.past_frames * 960:(self.past_frames + stop - start) * 960]
            output = output.detach().cpu().numpy().copy()
        if output.shape != ((stop - start) * 960,) or not np.isfinite(output).all():
            raise ValueError('Invalid streaming output extent or nonfinite sample')
        self.emissions.append({'start':start,'stop':stop,'received':self.received,'end_flush':self.flushing})
        self.next_frame = stop
        self.emitted_samples += len(output)
        retain_from = max(0, self.next_frame - self.past_frames)
        discarded = retain_from - self.base
        if discarded:
            for name in self.buffer:
                self.buffer[name] = self.buffer[name][discarded:].copy()
            self.base = retain_from
        elapsed = self._pending_processing + time.perf_counter() - started
        self.processing_seconds.append(elapsed)
        self.total_blocks += 1
        self.maximum_processing_seconds = max(self.maximum_processing_seconds, elapsed)
        self._pending_processing = 0.
        return output

    def feed(self, inputs, decoder_state):
        if self.finished:
            raise ValueError('Stream has already been flushed')
        count = len(inputs['pcm8'])
        shapes = {'pcm8': (count, 160), 'features': (count, 74), 'flags': (count,), 'errors': (count,)}
        if any(np.asarray(inputs[name]).shape != shape for name, shape in shapes.items()):
            raise ValueError('Streaming decoder input geometry differs')
        if np.asarray(decoder_state).shape != (count, STATE_DIM):
            raise ValueError('Streaming richer-state geometry differs')
        outputs = []
        position = 0
        while position < count:
            # Split even a huge incoming chunk before retaining it, so input
            # delivery size cannot turn this into whole-utterance buffering.
            needed = self.next_frame + BLOCK_FRAMES + self.future_frames - self.received
            take = min(count - position, needed)
            if take <= 0:
                raise RuntimeError('Streaming emission did not advance')
            stop = position + take
            self._append({name: value[position:stop] for name, value in inputs.items() if name in shapes},
                         decoder_state[position:stop])
            position = stop
            if self.received >= self.next_frame + BLOCK_FRAMES + self.future_frames:
                outputs.append(self._render(self.next_frame + BLOCK_FRAMES))
        return outputs

    def finish(self):
        if self.finished:
            raise ValueError('Stream has already been flushed')
        outputs = []
        self.flushing=True
        while self.next_frame < self.received:
            outputs.append(self._render(min(self.next_frame + BLOCK_FRAMES, self.received)))
        self.finished = True
        if self.emitted_samples != self.received * 960:
            raise RuntimeError('Final word/sample tail was lost or duplicated')
        return outputs

    def report(self):
        milliseconds = np.asarray(self.processing_seconds, np.float64) * 1000
        maximum = 1000 * self.maximum_processing_seconds if self.total_blocks else None
        context_ms = (self.future_frames + BLOCK_FRAMES) * 20
        return {'past_context_ms':720,'future_context_ms':self.future_frames*20,'block_ms':160,'recent_emissions':list(self.emissions),'input_frames': self.received, 'output_samples48': self.emitted_samples,
                'maximum_buffered_frames': self.max_buffered_frames,
                'buffer_limit_frames': self.past_frames + self.future_frames + BLOCK_FRAMES,
                'input_chunk_boundaries_reset_noise': False, 'all_final_samples_flushed': self.finished,
                'measured_blocks': self.total_blocks, 'percentile_window_blocks': len(milliseconds),
                'processing_ms': {'median': float(np.median(milliseconds)) if len(milliseconds) else None,
                                  'p95': float(np.quantile(milliseconds, .95)) if len(milliseconds) else None,
                                  'maximum': maximum},
                'nominal_context_and_block_ms': context_ms,
                'nominal_plus_observed_max_processing_ms': None if maximum is None else context_ms + maximum,
                'remaining_1000ms_margin_before_unmeasured_io': None if maximum is None else 1000 - context_ms - maximum,
                'model_throughput_below160ms_on_all_observed_blocks': maximum is not None and maximum < 160,
                'audio_device_qualified': False, 'native_decoder_integrated': False,
                'latency_scope': 'Measured normalization, copies, noise generation, model, CPU output and buffer retention; excludes native decoder, scheduler stalls, input transport and audio-device queues. No worst-case deadline guarantee.'}

