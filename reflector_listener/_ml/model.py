"""Frozen Pilot5C inference model; training and historical forward paths omitted.

Operations/state names retained from restoration_v2*_model.py for exact checkpoint
and numerical compatibility. Only the final Pilot5C forward path is callable.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .observations import SCALAR_NAMES, STATE_DIM

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
        self.blocks = nn.Sequential(*(TemporalBlock(channels, d) for d in (1, 2, 4, 8, 16, 32, 64)))
        self.head = nn.Conv1d(channels, 513 * 5, 1)
        nn.init.normal_(self.head.weight, std=0.0001)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('analysis_window', torch.hann_window(256))
        self.register_buffer('synthesis_window', torch.hann_window(960))
        f = torch.linspace(0, 24000, 513)
        high = ((f - 3400) / 1000).clamp(0, 1)
        low = (1 - high) * ((f - 80) / 120).clamp(0, 1)
        self.register_buffer('high', high[None, :, None])
        self.register_buffer('low', low[None, :, None])
        k = torch.arange(-960, 961, dtype=torch.float32)
        lp = 2 * 65 / 48000 * torch.sinc(2 * 65 * k / 48000)
        lp *= torch.kaiser_window(1921, periodic=False, beta=8.6)
        lp /= lp.sum()
        hp = -lp
        hp[960] += 1
        self.register_buffer('residual_highpass', hp.reshape(1, 1, -1))

    def spectrum(self, x):
        return torch.stft(x, 1024, HOP, win_length=960, window=self.synthesis_window, center=True, pad_mode='constant', return_complex=True)


def status_features(flags, errors, present):
    flags = flags.to(torch.int64)
    return torch.stack(((flags & 0x4f0 != 0).float(), errors.clamp(0, 8).float()/8,
                        (flags & 0x80000000 != 0).float(), present.float()), -1)
def linear_fft_filter(x, kernel):
    n=x.shape[-1];m=kernel.shape[-1]
    size=1 << (n+m-2).bit_length()
    # FIR is symmetric, so convolution equals the frozen conv1d correlation.
    result=torch.fft.irfft(torch.fft.rfft(x,size)*torch.fft.rfft(kernel.reshape(-1),size),size)
    return result[:,m//2:m//2+n]
CONTEXT_FRAMES = 36


BLOCK_FRAMES = 8


PCM_HOP = 40


EXTRA_WINDOWS = (64, 128, 512)


ACF_START = 16


ACF_STOP = 161


EXTRA_MAG_CHANNELS = sum(n//2+1 for n in EXTRA_WINDOWS)


DERIVATIVE_CHANNELS = 129 + EXTRA_MAG_CHANNELS


PERIODICITY_CHANNELS = ACF_STOP-ACF_START


EXTRA_CHANNELS = EXTRA_MAG_CHANNELS + DERIVATIVE_CHANNELS + PERIODICITY_CHANNELS


class SpeechReconstructorV2P3(SpeechReconstructorV2):
    """A compatible backbone with zero-initialized richer observations/repair."""

    def __init__(self, channels=96):
        super().__init__(channels=channels)
        for n in EXTRA_WINDOWS:
            self.register_buffer('analysis_window_' + str(n), torch.hann_window(n))
        self.multi_projection = nn.Conv1d(EXTRA_CHANNELS, channels, 1)
        self.repair_head = nn.Conv1d(channels, 4 * 513, 1)
        self.reset_new_branches()

    def reset_new_branches(self):
        """Return only new trainable branches to the pilot1-equivalent state."""
        nn.init.zeros_(self.multi_projection.weight)
        nn.init.zeros_(self.multi_projection.bias)
        nn.init.zeros_(self.repair_head.weight)
        nn.init.zeros_(self.repair_head.bias)

    def acoustic_features(self, pcm):
        """Return inherited32ms observation plus source-free new observations."""
        inherited = torch.stft(pcm, 256, PCM_HOP, window=self.analysis_window, center=True, pad_mode='constant', return_complex=True)
        acoustic = torch.log1p(inherited.abs() * 8)
        magnitudes = []
        long_spectrum = None
        for n in EXTRA_WINDOWS:
            window = getattr(self, 'analysis_window_' + str(n))
            spectrum = torch.stft(pcm, n, PCM_HOP, window=window, center=True, pad_mode='constant', return_complex=True)
            magnitudes.append(torch.log1p(spectrum.abs() * (8 * 256 / n)))
            if n == 512:
                long_spectrum = spectrum
        all_magnitudes = torch.cat((acoustic, *magnitudes), 1)
        differences = F.pad(all_magnitudes[:, :, 1:] - all_magnitudes[:, :, :-1], (1, 0))
        power = long_spectrum.abs().square().transpose(1, 2)
        acf = torch.fft.irfft(power, n=512, dim=-1)
        acf = acf / acf[..., :1].clamp_min(1e-08)
        periodicity = acf[..., ACF_START:ACF_STOP].clamp(-1, 1).transpose(1, 2)
        extra = torch.cat((*magnitudes, differences, periodicity), 1)
        if extra.shape[1] != EXTRA_CHANNELS or extra.shape[-1] != acoustic.shape[-1]:
            raise ValueError('Multiresolution feature grid differs')
        return (acoustic, extra)
SCALAR_OFFSET = 336


SCALAR_INDEX = {name: SCALAR_OFFSET + i for i, name in enumerate(SCALAR_NAMES)}


INVALID_CLASSES = ('flag_tone', 'flag_erasure', 'flag_repeat', 'flag_mute',
                   'flag_silence', 'flag_c0_uncorrectable', 'flag_python_silence')


def masked_decoder_state(state, present):
    """Torch equivalent of the sidecar validity mask, applied after centering.

Invalid acoustic estimates must remain exactly zero after normalization. Their
status/error observations and decoded PCM power are still useful to the model.
Correctable ECC counts are not automatically treated as uncorrectable speech.
All columns in absent/padded frames are zero. Nonfinite *usable* observations
are an input error; masked nonfinite stale decoder state cannot contaminate a
valid frame or the stock bypass.
"""
    if state.ndim != 3 or state.shape[-1] != STATE_DIM:
        raise ValueError('Expected decoder state [batch,frames,' + str(STATE_DIM) + ']')
    if present.shape != state.shape[:2]:
        raise ValueError('Decoder state/presence timelines differ')
    mask = torch.ones_like(state, dtype=torch.bool)
    invalid = torch.stack([state[..., SCALAR_INDEX[name]] > .5 for name in INVALID_CLASSES], -1).any(-1)
    for label, amp_start, voice_start, harmonic_start in (('pre', 0, 112, 224), ('post', 56, 168, 280)):
        valid = (state[..., SCALAR_INDEX[label + '_model_valid']] > .5) & ~invalid
        harmonic = (state[..., harmonic_start:harmonic_start + 56] > .5) & valid[..., None]
        mask[..., amp_start:amp_start + 56] = harmonic
        mask[..., voice_start:voice_start + 56] = harmonic
        mask[..., harmonic_start:harmonic_start + 56] = valid[..., None]
        for name in ('L_fraction', 'K_fraction', 'gamma', 'log_harmonic_power', 'voiced_fraction', 'model_valid'):
            mask[..., SCALAR_INDEX[label + '_' + name]] = valid
        pitch_valid = valid & (state[..., SCALAR_INDEX[label + '_pitch_valid']] > .5)
        mask[..., SCALAR_INDEX[label + '_log_f0']] = pitch_valid
        mask[..., SCALAR_INDEX[label + '_pitch_valid']] = pitch_valid
    mask &= (present > .5)[..., None]
    clean = torch.where(mask, state, torch.zeros_like(state))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError('Nonfinite usable decoder-state observation')
    return clean


class SpeechReconstructorV2P4(SpeechReconstructorV2P3):
    """Pilot3 plus optional 20 ms decoder observations; no added lookahead."""

    def __init__(self, channels=96):
        super().__init__(channels=channels)
        self.state_projection = nn.Conv1d(STATE_DIM, channels, 1, bias=False)
        self.reset_state_branch()

    def reset_state_branch(self):
        nn.init.zeros_(self.state_projection.weight)
GAIN_ANCHORS_HZ = (4000., 6000., 8000., 12000., 16000., 24000.)


def gain_interpolation_weights(frequencies_hz):
    """Convex interpolation between fixed anchors on the log-frequency axis.

Below 4 kHz the first gain is held constant; Pilot4's existing 3.4--4.4 kHz
crossfade remains responsible for the transition. At/above Nyquist the last
anchor is held. This function does not add an analysis window or lookahead.
"""
    if frequencies_hz.ndim != 1 or not frequencies_hz.is_floating_point():
        raise ValueError('Expected a floating-point frequency vector')
    if not bool(torch.isfinite(frequencies_hz).all()) or bool((frequencies_hz < 0).any()):
        raise ValueError('Frequencies must be finite and nonnegative')
    anchors = frequencies_hz.new_tensor(GAIN_ANCHORS_HZ).log()
    positions = frequencies_hz.clamp(GAIN_ANCHORS_HZ[0], GAIN_ANCHORS_HZ[-1]).log()
    right = torch.searchsorted(anchors, positions, right=True).clamp(1, len(anchors) - 1)
    left = right - 1
    fraction = ((positions - anchors[left]) / (anchors[right] - anchors[left])).clamp(0, 1)
    weights = frequencies_hz.new_zeros((len(frequencies_hz), len(anchors)))
    weights.scatter_(1, left[:, None], (1 - fraction)[:, None])
    weights.scatter_add_(1, right[:, None], fraction[:, None])
    return weights


class SpeechReconstructorV2P5(SpeechReconstructorV2P4):
    """Pilot4 plus bounded coarse high-band gain; no added temporal context."""

    def __init__(self, channels=96):
        super().__init__(channels=channels)
        self.gain_head = nn.Conv1d(channels, len(GAIN_ANCHORS_HZ), 1)
        self.register_buffer('gain_interpolation', gain_interpolation_weights(torch.linspace(0.0, 24000.0, 513)))
        self.reset_gain_branch()

    def reset_gain_branch(self):
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)

    def forward(self, pcm8, features, status, noise, decoder_state):
        if pcm8.ndim != 3 or pcm8.shape[-1] != 160:
            raise ValueError('Expected PCM8 as [batch,frames,160]')
        b, frames, _ = pcm8.shape
        if features.shape != (b, frames, 74) or status.shape != (b, frames, 4):
            raise ValueError('Decoder feature/status dimensions differ')
        if decoder_state.shape != (b, frames, STATE_DIM):
            raise ValueError('Rich decoder state dimensions differ')
        pcm = pcm8.reshape(b, -1)
        base = upsample8(pcm)
        if noise.shape != base.shape:
            raise ValueError('Noise must share the exact48kHz input timeline')
        acoustic, extra = self.acoustic_features(pcm)
        side = torch.cat((features, status), -1).repeat_interleave(4, 1)
        side = F.pad(side.transpose(1, 2), (0, 1), mode='replicate')
        if side.shape[-1] != acoustic.shape[-1]:
            raise ValueError('Decoder/spectrum time grids differ')
        state = masked_decoder_state(decoder_state, status[..., 3])
        state = F.pad(state.repeat_interleave(4, 1).transpose(1, 2), (0, 1), mode='replicate')
        hidden = self.input(torch.cat((acoustic, side), 1)) + self.multi_projection(extra)
        hidden = hidden + self.state_projection(state)
        hidden = self.blocks(F.silu(hidden))
        heads = self.head(hidden).reshape(b, 5, 513, -1)
        repair = self.repair_head(hidden).reshape(b, 4, 513, -1)
        high_gain = self.gains_from_logits(self.gain_head(hidden))
        envelope = (F.avg_pool1d(base.square()[:, None], 961, 1, 480)[:, 0] + 1e-12).sqrt()
        harmonic = envelope * torch.tanh(3 * base / envelope.clamp_min(1e-06))
        stochastic = envelope * noise
        harmonic_s, noise_s = (self.spectrum(harmonic), self.spectrum(stochastic))
        base_s = self.spectrum(base)
        low_delta = 0.5 * torch.tanh(heads[:, 0]) * self.low * base_s
        periodic = torch.complex(torch.tanh(heads[:, 1]), torch.tanh(heads[:, 2]))
        aperiodic = torch.complex(torch.tanh(heads[:, 3]), torch.tanh(heads[:, 4]))
        inherited_delta = low_delta + self.high * (high_gain * (4 * periodic * harmonic_s + 0.5 * aperiodic * noise_s))
        complex_repair = torch.complex(0.5 * torch.tanh(repair[:, 0]), 0.5 * torch.tanh(repair[:, 1]))
        aperiodic_repair = torch.complex(0.15 * torch.tanh(repair[:, 2]), 0.15 * torch.tanh(repair[:, 3]))
        delta_s = inherited_delta + self.low * (complex_repair * base_s + aperiodic_repair * noise_s)
        delta = torch.istft(delta_s, 1024, HOP, win_length=960, window=self.synthesis_window, center=True, length=base.shape[-1])
        return base + linear_fft_filter(delta, self.residual_highpass)
POST_TANH_GAIN_SCALE = .5


LOG_HIGH_GAIN_BOUND = POST_TANH_GAIN_SCALE * math.log(8.)


class SpeechReconstructorV2P5C(SpeechReconstructorV2P5):
    """Identical Pilot5 controls with a conservative fixed global gain range."""

    def gains_from_logits(self, logits):
        if logits.ndim != 3 or logits.shape[1] != len(GAIN_ANCHORS_HZ):
            raise ValueError('Expected six high-band gain logits per time step')
        bounded_log_gain = torch.einsum('fa,bat->bft', self.gain_interpolation, torch.tanh(logits))
        return torch.exp(LOG_HIGH_GAIN_BOUND * bounded_log_gain)
