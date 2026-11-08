# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Pilot3: multiresolution decoded-waveform conditioning and fuller-band repair.

This optional post-decoder model never changes or calls the native 8 kHz decoder.
All runtime observations come from stock PCM8, the frozen74 decoder features,
status and caller-owned continuous noise. Original audio/phones are not inputs.

All pilot1 state keys remain unchanged. New conditioning/repair projections start
at zero, so a checked pilot1 warm start retains its output to numerical precision.
Both training and CPU inference use the same differentiable linear FFT FIR.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._model_v2 import SpeechReconstructorV2, upsample8
from ._filters import linear_fft_filter

CONTEXT_FRAMES = 36
BLOCK_FRAMES = 8
HOP = 240
PCM_HOP = 40
EXTRA_WINDOWS = (64, 128, 512)  # 8,16,64ms at the actual 8kHz input rate.
ACF_START = 16                # 2ms at 8kHz.
ACF_STOP = 161                # Include20ms; these are descriptors, not F0 labels.
EXTRA_MAG_CHANNELS = sum(n//2+1 for n in EXTRA_WINDOWS)
DERIVATIVE_CHANNELS = 129 + EXTRA_MAG_CHANNELS
PERIODICITY_CHANNELS = ACF_STOP-ACF_START
EXTRA_CHANNELS = EXTRA_MAG_CHANNELS + DERIVATIVE_CHANNELS + PERIODICITY_CHANNELS


class SpeechReconstructorV2P3(SpeechReconstructorV2):
    """A compatible backbone with zero-initialized richer observations/repair."""

    def __init__(self, channels=96):
        super().__init__(channels=channels)
        for n in EXTRA_WINDOWS:
            self.register_buffer('analysis_window_'+str(n), torch.hann_window(n))
        self.multi_projection = nn.Conv1d(EXTRA_CHANNELS, channels, 1)
        # Two complex controls: stock-spectrum correction and aperiodic repair.
        self.repair_head = nn.Conv1d(channels, 4*513, 1)
        self.reset_new_branches()

    def reset_new_branches(self):
        """Return only new trainable branches to the pilot1-equivalent state."""
        nn.init.zeros_(self.multi_projection.weight)
        nn.init.zeros_(self.multi_projection.bias)
        nn.init.zeros_(self.repair_head.weight)
        nn.init.zeros_(self.repair_head.bias)

    def new_state_keys(self):
        prefixes = ('multi_projection.', 'repair_head.', 'analysis_window_')
        return sorted(k for k in self.state_dict() if k.startswith(prefixes))

    def load_pilot1_state(self, state):
        """Load exactly a pilot1 model state; fail on hidden/missing old fields.

        Returns the explicit expected missing/new keys for the run receipt.
        Full pilot3 checkpoint resumes should instead use load_state_dict(...,
        strict=True), preserving trained new branches and their optimizer state.
        """
        expected_new = set(self.new_state_keys())
        expected_old = set(self.state_dict())-expected_new
        supplied = set(state)
        if supplied != expected_old:
            raise ValueError('Pilot1 warm-start keys differ: missing='+str(sorted(expected_old-supplied))
                             +', unexpected='+str(sorted(supplied-expected_old)))
        self.reset_new_branches()
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != expected_new or result.unexpected_keys:
            raise ValueError('Unexpected pilot1 warm-start loading result')
        return {'expected_new_keys': sorted(expected_new),
                'missing_keys': sorted(result.missing_keys),
                'unexpected_keys': list(result.unexpected_keys),
                'new_trainable_branches_zero_initialized': True}

    def specification(self):
        return {
            'architecture': 'multiresolution-fullerband-repair-v2-pilot3',
            'channels': self.channels, 'sample_rate':48000,
            'input_sample_rate':8000, 'input_features':74, 'status_features':4,
            'analysis_windows_ms':[8,16,32,64], 'analysis_hop_ms':5,
            'additional_conditioning_channels':EXTRA_CHANNELS,
            'conditioning': 'fixed-scale logmagnitudes, backward5ms spectral differences, normalized64ms-window circular ACF lags2..20ms',
            'spectral_amplitude_scaling': 'extra FFT magnitudes scaled by256/window_samples; no per-utterance level normalization',
            'periodicity_descriptor': 'windowed circular autocorrelation normalized by zero-lag energy; not a reference pitch target',
            'context_frames_each_side':CONTEXT_FRAMES, 'block_frames':BLOCK_FRAMES,
            'nominal_block_wait_plus_future_ms':880,
            'low_band': 'pilot1 gain plus bounded complex stock correction and small complex-filtered continuous-noise repair',
            'extra_low_complex_real_bound':.5, 'extra_low_complex_imag_bound':.5,
            'extra_aperiodic_complex_component_bound':.15,
            'high_band': 'inherited dense complex harmonic/noise filters, conditioned on richer features',
            'residual_highpass_hz':65,
            'fixed_fir_implementation': 'zero-padded linear FFT convolution; identical coefficients to frozen pilot1',
            'original_audio_runtime_input':False,
            'default8_decoder_called_or_modified':False,
            'warm_start': 'all pilot1 state keys retained; new branches zero',
            'latency_claim': '880ms algorithmic block/context budget; actual execution/device buffering must be measured separately',
        }

    @staticmethod
    def baseline(pcm8):
        """Exact stock48 interpolation bypass, independent of model parameters."""
        if pcm8.ndim != 3 or pcm8.shape[-1] != 160:
            raise ValueError('Expected PCM8 as [batch,frames,160]')
        return upsample8(pcm8.flatten(1))

    def acoustic_features(self, pcm):
        """Return inherited32ms observation plus source-free new observations."""
        inherited = torch.stft(pcm,256,PCM_HOP,window=self.analysis_window,
                              center=True,pad_mode='constant',return_complex=True)
        acoustic = torch.log1p(inherited.abs()*8)
        magnitudes = []
        long_spectrum = None
        for n in EXTRA_WINDOWS:
            window = getattr(self,'analysis_window_'+str(n))
            spectrum = torch.stft(pcm,n,PCM_HOP,window=window,
                                  center=True,pad_mode='constant',return_complex=True)
            # Normalize only the known analysis-window scale, not speech level.
            magnitudes.append(torch.log1p(spectrum.abs()*(8*256/n)))
            if n == 512:
                long_spectrum = spectrum
        all_magnitudes = torch.cat((acoustic,*magnitudes),1)
        differences = F.pad(all_magnitudes[:,:,1:]-all_magnitudes[:,:,:-1],(1,0))
        # A fixed local descriptor. No source-derived F0 or voice decision.
        power = long_spectrum.abs().square().transpose(1,2)
        acf = torch.fft.irfft(power,n=512,dim=-1)
        acf = acf/acf[...,:1].clamp_min(1e-8)
        periodicity = acf[...,ACF_START:ACF_STOP].clamp(-1,1).transpose(1,2)
        extra = torch.cat((*magnitudes,differences,periodicity),1)
        if extra.shape[1] != EXTRA_CHANNELS or extra.shape[-1] != acoustic.shape[-1]:
            raise ValueError('Multiresolution feature grid differs')
        return acoustic,extra

    def forward(self, pcm8, features, status, noise):
        if pcm8.ndim != 3 or pcm8.shape[-1] != 160:
            raise ValueError('Expected PCM8 as [batch,frames,160]')
        b,frames,_ = pcm8.shape
        if features.shape != (b,frames,74) or status.shape != (b,frames,4):
            raise ValueError('Decoder feature/status dimensions differ')
        pcm = pcm8.reshape(b,-1)
        base = upsample8(pcm)
        if noise.shape != base.shape:
            raise ValueError('Noise must share the exact48kHz input timeline')
        acoustic,extra = self.acoustic_features(pcm)
        side = torch.cat((features,status),-1).repeat_interleave(4,1)
        side = F.pad(side.transpose(1,2),(0,1),mode='replicate')
        if side.shape[-1] != acoustic.shape[-1]:
            raise ValueError('Decoder/spectrum time grids differ')
        hidden = self.input(torch.cat((acoustic,side),1)) + self.multi_projection(extra)
        hidden = self.blocks(F.silu(hidden))
        heads = self.head(hidden).reshape(b,5,513,-1)
        repair = self.repair_head(hidden).reshape(b,4,513,-1)
        envelope = (F.avg_pool1d(base.square()[:,None],961,1,480)[:,0]+1e-12).sqrt()
        harmonic = envelope*torch.tanh(3*base/envelope.clamp_min(1e-6))
        stochastic = envelope*noise
        harmonic_s,noise_s = self.spectrum(harmonic),self.spectrum(stochastic)
        base_s = self.spectrum(base)
        low_delta = .5*torch.tanh(heads[:,0])*self.low*base_s
        periodic = torch.complex(torch.tanh(heads[:,1]),torch.tanh(heads[:,2]))
        aperiodic = torch.complex(torch.tanh(heads[:,3]),torch.tanh(heads[:,4]))
        inherited_delta = low_delta + self.high*(4*periodic*harmonic_s+.5*aperiodic*noise_s)
        complex_repair = torch.complex(.5*torch.tanh(repair[:,0]),.5*torch.tanh(repair[:,1]))
        aperiodic_repair = torch.complex(.15*torch.tanh(repair[:,2]),.15*torch.tanh(repair[:,3]))
        delta_s = inherited_delta + self.low*(complex_repair*base_s + aperiodic_repair*noise_s)
        delta = torch.istft(delta_s,1024,HOP,win_length=960,window=self.synthesis_window,
                            center=True,length=base.shape[-1])
        return base + linear_fft_filter(delta,self.residual_highpass)


class SpeechReconstructorV2P3FastCPU(SpeechReconstructorV2P3):
    """Explicit CPU-facing name; same weights and FFT-FIR forward as training."""
    pass