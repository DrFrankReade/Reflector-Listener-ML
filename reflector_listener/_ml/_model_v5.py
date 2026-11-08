# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Provisional Pilot5: smoothly relax Pilot4's high-band synthesis amplitude.

This challenger must be justified by the offline synthesis-ceiling experiment
before training. It retains the frozen Pilot4 observations, low-band repair,
carriers and temporal context. Six pointwise log-gain controls scale only the
existing high-band branch. Reference speech is never a runtime observation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .observations import STATE_DIM
from ._model_v2 import upsample8
from ._filters import linear_fft_filter
from ._model_v4 import (
    BLOCK_FRAMES, CONTEXT_FRAMES, HOP, SpeechReconstructorV2P4, masked_decoder_state)

GAIN_ANCHORS_HZ = (4000., 6000., 8000., 12000., 16000., 24000.)
MAX_HIGH_GAIN = 8.


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
        self.register_buffer('gain_interpolation', gain_interpolation_weights(torch.linspace(0., 24000., 513)))
        self.reset_gain_branch()

    def reset_gain_branch(self):
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)

    @staticmethod
    def gain_branch_keys():
        return ['gain_head.weight', 'gain_head.bias', 'gain_interpolation']

    def load_pilot4_state(self, state):
        """Accept exactly Pilot4 state, preserving all inherited tensors."""
        expected_new = set(self.gain_branch_keys())
        expected_old = set(self.state_dict()) - expected_new
        if set(state) != expected_old:
            raise ValueError('Pilot4 warm-start keys differ: missing=' + str(sorted(expected_old - set(state)))
                             + ', unexpected=' + str(sorted(set(state) - expected_old)))
        self.reset_gain_branch()
        self.gain_interpolation.copy_(gain_interpolation_weights(
            torch.linspace(0., 24000., 513, device=self.gain_interpolation.device,
                           dtype=self.gain_interpolation.dtype)))
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != expected_new or result.unexpected_keys:
            raise ValueError('Unexpected Pilot4 warm-start loading result')
        return {'missing_keys': sorted(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys),
                'gain_head_zero_initialized': True, 'initial_high_gain': 1.,
                'inherited_parameters_changed': False}

    def gains_from_logits(self, logits):
        """Map six controls [batch,anchors,time] to bounded per-bin gains."""
        if logits.ndim != 3 or logits.shape[1] != len(GAIN_ANCHORS_HZ):
            raise ValueError('Expected six high-band gain logits per time step')
        bounded_log_gain = torch.einsum('fa,bat->bft', self.gain_interpolation, torch.tanh(logits))
        return torch.exp(math.log(MAX_HIGH_GAIN) * bounded_log_gain)

    def specification(self):
        result = super().specification()
        result.update(architecture='coarse-high-band-gain-v2-pilot5',
                      inherited_architecture='decoder-state-conditioned-v2-pilot4',
                      provisional_ceiling_evidence_required=True,
                      additional_parameters=(self.channels + 1) * len(GAIN_ANCHORS_HZ),
                      gain_anchors_hz=list(GAIN_ANCHORS_HZ),
                      high_gain_range=[1 / MAX_HIGH_GAIN, MAX_HIGH_GAIN],
                      gain_interpolation='convex linear in log frequency and bounded log gain',
                      gain_scope='existing high-band periodic plus aperiodic branch only',
                      gain_initialization='zero weights and bias; exact unit multiplier',
                      acoustic_frontend_changed=False, synthesis_changed=True, loss_changed=False,
                      low_band_repair_changed=False, excitation_carriers_changed=False,
                      additional_lookahead_ms=0,
                      warm_start='all Pilot4 state keys retained; gain head zero',
                      comparison_scope='high-band synthesis amplitude only; unchanged decoder conditioning')
        return result

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
        harmonic = envelope * torch.tanh(3 * base / envelope.clamp_min(1e-6))
        stochastic = envelope * noise
        harmonic_s, noise_s = self.spectrum(harmonic), self.spectrum(stochastic)
        base_s = self.spectrum(base)
        low_delta = .5 * torch.tanh(heads[:, 0]) * self.low * base_s
        periodic = torch.complex(torch.tanh(heads[:, 1]), torch.tanh(heads[:, 2]))
        aperiodic = torch.complex(torch.tanh(heads[:, 3]), torch.tanh(heads[:, 4]))
        inherited_delta = low_delta + self.high * (high_gain * (4 * periodic * harmonic_s + .5 * aperiodic * noise_s))
        complex_repair = torch.complex(.5 * torch.tanh(repair[:, 0]), .5 * torch.tanh(repair[:, 1]))
        aperiodic_repair = torch.complex(.15 * torch.tanh(repair[:, 2]), .15 * torch.tanh(repair[:, 3]))
        delta_s = inherited_delta + self.low * (complex_repair * base_s + aperiodic_repair * noise_s)
        delta = torch.istft(delta_s, 1024, HOP, win_length=960, window=self.synthesis_window,
                            center=True, length=base.shape[-1])
        return base + linear_fft_filter(delta, self.residual_highpass)


class SpeechReconstructorV2P5FastCPU(SpeechReconstructorV2P5):
    """Same parameters and numerical forward; explicit CPU-facing type."""
    pass
