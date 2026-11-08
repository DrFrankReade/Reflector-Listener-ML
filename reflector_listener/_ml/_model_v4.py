# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Pilot4: read-only stock-decoder state conditioning of the frozen Pilot3 design.

The only model change is a zero-initialized, bias-free state projection before
the existing temporal network. Acoustic frontend, synthesis, context and losses
are unchanged. Runtime observations never include original/reference speech.
The caller preserves binary masks while normalizing continuous state columns
using training-only statistics from decoder_state_v4.normalize_features().
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .observations import SCALAR_NAMES, STATE_DIM
from ._model_v2 import upsample8
from ._filters import linear_fft_filter
from ._model_v3 import BLOCK_FRAMES, CONTEXT_FRAMES, HOP, SpeechReconstructorV2P3

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

    @staticmethod
    def state_branch_keys():
        return ['state_projection.weight']

    def load_pilot3_state(self, state):
        """Accept exactly the inherited Pilot3 keys; never partially warm-start."""
        expected_new = set(self.state_branch_keys())
        expected_old = set(self.state_dict()) - expected_new
        if set(state) != expected_old:
            raise ValueError('Pilot3 warm-start keys differ: missing=' + str(sorted(expected_old - set(state)))
                             + ', unexpected=' + str(sorted(set(state) - expected_old)))
        self.reset_state_branch()
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != expected_new or result.unexpected_keys:
            raise ValueError('Unexpected Pilot3 warm-start loading result')
        return {'missing_keys': sorted(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys),
                'state_projection_zero_initialized': True, 'inherited_parameters_changed': False}

    def specification(self):
        result = super().specification()
        result.update(architecture='decoder-state-conditioned-v2-pilot4',
                      inherited_architecture='multiresolution-fullerband-repair-v2-pilot3',
                      decoder_state_dim=STATE_DIM, decoder_state_projection='bias-free pointwise, zero initialized',
                      additional_parameters=STATE_DIM * self.channels,
                      decoder_state_rate_hz=50, decoder_state_alignment='20ms frame repeated on existing5ms grid',
                      decoder_state_masking='pre/post harmonic and model validity; pitch validity; invalid frame class; padding',
                      state_normalization='training-only continuous columns; binary/status masks unchanged; invalid values zero',
                      acoustic_frontend_changed=False, synthesis_changed=False, loss_changed=False,
                      additional_lookahead_ms=0, warm_start='all Pilot3 state keys retained; state projection zero',
                      optional_pitch_excitation_enabled=False,
                      comparison_scope='conditioning only; no pitch-excitation challenger in this comparison')
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
        envelope = (F.avg_pool1d(base.square()[:, None], 961, 1, 480)[:, 0] + 1e-12).sqrt()
        harmonic = envelope * torch.tanh(3 * base / envelope.clamp_min(1e-6))
        stochastic = envelope * noise
        harmonic_s, noise_s = self.spectrum(harmonic), self.spectrum(stochastic)
        base_s = self.spectrum(base)
        low_delta = .5 * torch.tanh(heads[:, 0]) * self.low * base_s
        periodic = torch.complex(torch.tanh(heads[:, 1]), torch.tanh(heads[:, 2]))
        aperiodic = torch.complex(torch.tanh(heads[:, 3]), torch.tanh(heads[:, 4]))
        inherited_delta = low_delta + self.high * (4 * periodic * harmonic_s + .5 * aperiodic * noise_s)
        complex_repair = torch.complex(.5 * torch.tanh(repair[:, 0]), .5 * torch.tanh(repair[:, 1]))
        aperiodic_repair = torch.complex(.15 * torch.tanh(repair[:, 2]), .15 * torch.tanh(repair[:, 3]))
        delta_s = inherited_delta + self.low * (complex_repair * base_s + aperiodic_repair * noise_s)
        delta = torch.istft(delta_s, 1024, HOP, win_length=960, window=self.synthesis_window,
                            center=True, length=base.shape[-1])
        return base + linear_fft_filter(delta, self.residual_highpass)


class SpeechReconstructorV2P4FastCPU(SpeechReconstructorV2P4):
    """Same parameters and numerical forward; explicit CPU-facing type."""
    pass
