Experimental ML Enhanced 48kHz Decoder runtime

Frozen Pilot5C candidate, step 15900, selected from the ML Enhanced 48kHz
Decoder work in the decoder research project. The selected setting is
360 ms future context, 720 ms past context and 160 ms output blocks: nominal
buffering of 520 ms plus native/model processing, network and output-device queues.

candidate.pt contains only the candidate's tensor state dictionary. Optimizers,
comparison models, training data, references, captures, evaluations and training
code are not included. The two normalization files contain only inference
statistics/schema fields. Inference-only Python lives in reflector_listener/_ml.
CPU PyTorch and NumPy are required; no GPU, training server or source checkout
is used at runtime. This is a best-effort test model, not a quality-qualified
production replacement. Its source record explicitly says quality_pass=false.

stock265.dll is the separately pinned mbelib-neo 2.6.5 decoder used by the model.
It does not replace native/bin/mbe-neo.dll or alter the ordinary 8 kHz path.
Upstream: https://github.com/arancormonk/mbelib-neo
Source tag v2.6.5; commit 77a9a33b93eff910bc2bcade18c1e76fce1b7105.
Source: https://github.com/arancormonk/mbelib-neo/tree/77a9a33b93eff910bc2bcade18c1e76fce1b7105
License: GPL-2.0-or-later; see LICENSE.txt. The Python decoder binding and
observation extraction retain the same SPDX notice as their source binding.
MANIFEST.json records source checkpoint provenance and runtime asset hashes.
