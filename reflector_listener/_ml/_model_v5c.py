# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
"""Conservative Pilot5 candidate supported by the global half-log-gain oracle.

The frozen wide-range Pilot5/oracle implementation remains untouched. This
subclass changes only the gain exponent to half its original value: exactly
the globally scaled0.5 post-tanh-anchor family, with amplitude bounds1/sqrt(8)
through sqrt(8). No new weights, runtime observations or lookahead are added.
"""
from __future__ import annotations

import math

import torch

from ._model_v5 import GAIN_ANCHORS_HZ, SpeechReconstructorV2P5

POST_TANH_GAIN_SCALE = .5
LOG_HIGH_GAIN_BOUND = POST_TANH_GAIN_SCALE * math.log(8.)
MAX_HIGH_GAIN = math.sqrt(8.)


class SpeechReconstructorV2P5C(SpeechReconstructorV2P5):
    """Identical Pilot5 controls with a conservative fixed global gain range."""

    def gains_from_logits(self, logits):
        if logits.ndim != 3 or logits.shape[1] != len(GAIN_ANCHORS_HZ):
            raise ValueError('Expected six high-band gain logits per time step')
        bounded_log_gain = torch.einsum('fa,bat->bft', self.gain_interpolation, torch.tanh(logits))
        return torch.exp(LOG_HIGH_GAIN_BOUND * bounded_log_gain)

    def specification(self):
        result = super().specification()
        result.update(architecture='conservative-coarse-high-band-gain-v2-pilot5c',
                      inherited_gain_architecture='coarse-high-band-gain-v2-pilot5',
                      high_gain_range=[1 / MAX_HIGH_GAIN, MAX_HIGH_GAIN],
                      post_tanh_gain_scale=POST_TANH_GAIN_SCALE,
                      oracle_family='global0.5 scaling of frozen Pilot5 post-tanh gain anchors',
                      additional_parameters_over_pilot5=0,
                      comparison_scope='conservative high-band gain amplitude only; unchanged decoder conditioning',
                      safety_claim='restricted gain reachability; learned texture/quality still require held-out evaluation')
        return result


class SpeechReconstructorV2P5CFastCPU(SpeechReconstructorV2P5C):
    """Same conservative parameters and numerical forward; explicit CPU type."""
    pass
