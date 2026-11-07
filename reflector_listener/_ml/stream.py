"""Bounded 360 ms future-context inference, extracted from FutureContextStream.

Diagnostic per-block history is omitted so live calls have bounded memory.
The neural observations, padding, block boundaries and noise timeline are unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
from .observations import STATE_DIM, normalize_features
from .model import BLOCK_FRAMES, status_features

class FutureContextStream:
    """Receive arbitrary frame chunks, emit fixed blocks once future context exists.

    No model-owned recurrent state or decoder state is reset here. Every input
    frame remains on its original timeline. ``finish`` declares the real end
    and flushes every final sample with explicit unavailable-future padding.
    """

    def __init__(self, model, normalization, state_normalization, *, device='cpu', seed=6110704, future_frames=36):
        if future_frames not in (36, 18, 9):
            raise ValueError('Use36,18,9 future codec frames')
        self.past_frames = 36
        self.future_frames = future_frames
        self.flushing = False
        self.model = model.eval()
        self.device = device
        self.mean = np.asarray(normalization['mean'], np.float32)
        self.std = np.asarray(normalization['std'], np.float32)
        if self.mean.shape != (74,) or self.std.shape != (74,) or np.any(self.std <= 0):
            raise ValueError('Invalid frozen compact-feature normalization')
        self.state_normalization = state_normalization
        self.rng = np.random.default_rng(seed)
        self.buffer = {'pcm8': np.empty((0, 160), np.float32), 'features': np.empty((0, 74), np.float32), 'flags': np.empty((0,), np.int64), 'errors': np.empty((0,), np.int32), 'state': np.empty((0, STATE_DIM), np.float32), 'noise': np.empty((0, 960), np.float32)}
        self.base = self.received = self.next_frame = self.emitted_samples = self.max_buffered_frames = 0
        self.finished = False

    def _append(self, inputs, state):
        count = len(inputs['pcm8'])
        normalized = np.clip((np.asarray(inputs['features'], np.float32) - self.mean) / self.std, -12, 12)
        additions = {'pcm8': np.asarray(inputs['pcm8'], np.float32), 'features': normalized, 'flags': np.asarray(inputs['flags'], np.int64), 'errors': np.asarray(inputs['errors'], np.int32), 'state': normalize_features(state, self.state_normalization), 'noise': self.rng.standard_normal(count * 960).astype(np.float32).reshape(count, 960)}
        for name, value in additions.items():
            self.buffer[name] = np.concatenate((self.buffer[name], value))
        self.received += count
        self.max_buffered_frames = max(self.max_buffered_frames, len(self.buffer['pcm8']))
        if len(self.buffer['pcm8']) > self.past_frames + self.future_frames + BLOCK_FRAMES:
            raise RuntimeError('Streaming retained more than the declared context/block bound')

    def _render(self, stop):
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
        flags, errors = (tensor('flags'), tensor('errors'))
        status = status_features(flags, errors, torch.from_numpy(present[None]).to(self.device))
        arguments = (tensor('pcm8'), tensor('features'), status, tensor('noise').flatten(1), tensor('state'))
        with torch.inference_mode():
            output = self.model(*arguments)[0, self.past_frames * 960:(self.past_frames + stop - start) * 960]
            output = output.detach().cpu().numpy().copy()
        if output.shape != ((stop - start) * 960,) or not np.isfinite(output).all():
            raise ValueError('Invalid streaming output extent or nonfinite sample')
        self.next_frame = stop
        self.emitted_samples += len(output)
        retain_from = max(0, self.next_frame - self.past_frames)
        discarded = retain_from - self.base
        if discarded:
            for name in self.buffer:
                self.buffer[name] = self.buffer[name][discarded:].copy()
            self.base = retain_from
        return output

    def feed(self, inputs, decoder_state):
        if self.finished:
            raise ValueError('Stream has already been flushed')
        count = len(inputs['pcm8'])
        shapes = {'pcm8': (count, 160), 'features': (count, 74), 'flags': (count,), 'errors': (count,)}
        if any((np.asarray(inputs[name]).shape != shape for name, shape in shapes.items())):
            raise ValueError('Streaming decoder input geometry differs')
        if np.asarray(decoder_state).shape != (count, STATE_DIM):
            raise ValueError('Streaming richer-state geometry differs')
        outputs = []
        position = 0
        while position < count:
            needed = self.next_frame + BLOCK_FRAMES + self.future_frames - self.received
            take = min(count - position, needed)
            if take <= 0:
                raise RuntimeError('Streaming emission did not advance')
            stop = position + take
            self._append({name: value[position:stop] for name, value in inputs.items() if name in shapes}, decoder_state[position:stop])
            position = stop
            if self.received >= self.next_frame + BLOCK_FRAMES + self.future_frames:
                outputs.append(self._render(self.next_frame + BLOCK_FRAMES))
        return outputs

    def finish(self):
        if self.finished:
            raise ValueError('Stream has already been flushed')
        outputs = []
        self.flushing = True
        while self.next_frame < self.received:
            outputs.append(self._render(min(self.next_frame + BLOCK_FRAMES, self.received)))
        self.finished = True
        if self.emitted_samples != self.received * 960:
            raise RuntimeError('Final word/sample tail was lost or duplicated')
        return outputs
