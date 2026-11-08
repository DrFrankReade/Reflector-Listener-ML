# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Optional stock265 speech reconstruction. No encoder or native decoder changes.

A dense 5 ms acoustic predictor drives phase-anchored low-band repair and
complex filtering of harmonic/nonperiodic excitation above the codec band.
All normalization is per time point; there is no global temporal normalization.
Caller supplies continuous noise. Reference audio is never a runtime input.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

CONTEXT_FRAMES = 35
BLOCK_FRAMES = 8
HOP = 240


def interpolation_kernel():
    k = torch.arange(-60, 61, dtype=torch.float32)
    h = torch.sinc(k / 6) * torch.kaiser_window(121, periodic=False, beta=8.6)
    return (h * (6 / h.sum())).reshape(1, 1, -1)


def upsample8(x):
    """Centered 121 tap reference interpolation; x [B,N8]."""
    y = x.new_zeros(x.shape[0], x.shape[1] * 6)
    y[:, ::6] = x
    return F.conv1d(y[:, None], interpolation_kernel().to(x), padding=60)[:, 0]


class PointNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = PointNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=dilation,
                                   dilation=dilation, groups=channels)
        self.gates = nn.Conv1d(channels, channels * 2, 1)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        a, b = self.gates(self.depthwise(self.norm(x))).chunk(2, 1)
        return x + .2 * self.out(torch.tanh(a) * torch.sigmoid(b))


class SpeechReconstructorV2(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.channels = channels
        self.input = nn.Conv1d(129 + 74 + 4, channels, 1)
        self.blocks = nn.Sequential(*(TemporalBlock(channels, d) for d in (1,2,4,8,16,32,64)))
        self.head = nn.Conv1d(channels, 513 * 5, 1)
        nn.init.normal_(self.head.weight, std=1e-4)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('analysis_window', torch.hann_window(256))
        self.register_buffer('synthesis_window', torch.hann_window(960))
        f = torch.linspace(0, 24000, 513)
        high = ((f - 3400) / 1000).clamp(0, 1)
        low = (1 - high) * ((f - 80) / 120).clamp(0, 1)
        self.register_buffer('high', high[None, :, None])
        self.register_buffer('low', low[None, :, None])
        k = torch.arange(-960, 961, dtype=torch.float32)
        lp = (2 * 65 / 48000) * torch.sinc(2 * 65 * k / 48000)
        lp *= torch.kaiser_window(1921, periodic=False, beta=8.6)
        lp /= lp.sum()
        hp = -lp; hp[960] += 1
        self.register_buffer('residual_highpass', hp.reshape(1, 1, -1))

    def specification(self):
        return {'architecture':'dense-excitation-filter-v2-pilot1', 'channels':self.channels,
                'input_features':74, 'sample_rate':48000, 'analysis_hop_ms':5,
                'context_frames_each_side':CONTEXT_FRAMES, 'block_frames':BLOCK_FRAMES,
                'nominal_block_wait_plus_future_ms':860,
                'residual_highpass_hz':65, 'original_audio_runtime_input':False,
                'low_band':'bounded real mask, stock phase',
                'high_band':'dense complex filters on nonlinear harmonic and continuous-noise excitation'}

    def spectrum(self, x):
        return torch.stft(x, 1024, HOP, win_length=960, window=self.synthesis_window,
                          center=True, pad_mode='constant', return_complex=True)

    def forward(self, pcm8, features, status, noise):
        # pcm8 [B,frames,160], features [B,frames,74], status [B,frames,4].
        b, frames, _ = pcm8.shape
        pcm = pcm8.reshape(b, -1)
        base = upsample8(pcm)
        if noise.shape != base.shape:
            raise ValueError('Noise must share the exact 48 kHz input timeline')
        s8 = torch.stft(pcm, 256, 40, window=self.analysis_window,
                        center=True, pad_mode='constant', return_complex=True)
        acoustic = torch.log1p(s8.abs() * 8)
        side = torch.cat((features, status), -1).repeat_interleave(4, 1)
        side = F.pad(side.transpose(1, 2), (0, 1), mode='replicate')
        if side.shape[-1] != acoustic.shape[-1]:
            raise ValueError('Feature/spectrum time grids differ')
        h = self.blocks(F.silu(self.input(torch.cat((acoustic, side), 1))))
        heads = self.head(h).reshape(b, 5, 513, -1)
        # A local envelope is a synthesizer scale, never a target or score.
        envelope = (F.avg_pool1d(base.square()[:, None], 961, 1, 480)[:, 0] + 1e-12).sqrt()
        harmonic = envelope * torch.tanh(3 * base / envelope.clamp_min(1e-6))
        stochastic = envelope * noise
        harmonic_s, noise_s = self.spectrum(harmonic), self.spectrum(stochastic)
        base_s = self.spectrum(base)
        low_delta = .5 * torch.tanh(heads[:, 0]) * self.low * base_s
        periodic = torch.complex(torch.tanh(heads[:, 1]), torch.tanh(heads[:, 2]))
        aperiodic = torch.complex(torch.tanh(heads[:, 3]), torch.tanh(heads[:, 4]))
        delta_s = low_delta + self.high * (4 * periodic * harmonic_s + .5 * aperiodic * noise_s)
        delta = torch.istft(delta_s, 1024, HOP, win_length=960,
                            window=self.synthesis_window, center=True, length=base.shape[-1])
        # Fixed zero-DC residual projection. The frozen stock path is not filtered.
        delta = F.conv1d(delta[:, None], self.residual_highpass, padding=960)[:, 0]
        return base + delta


def status_features(flags, errors, present):
    flags = flags.to(torch.int64)
    return torch.stack(((flags & 0x4f0 != 0).float(), errors.clamp(0, 8).float()/8,
                        (flags & 0x80000000 != 0).float(), present.float()), -1)