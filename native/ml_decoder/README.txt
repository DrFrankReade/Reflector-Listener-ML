Experimental ML Enhanced 48kHz Decoder runtime — full360-v1

Accepted Pilot5C full360 model, step 473130. The inference-only weights and
qualified Python API come from the decoder research project's full360-v1
export. Context remains 720 ms past, 360 ms future and 160 ms output blocks:
nominal buffering is 520 ms plus processing, network and device queues.

weights.pt contains only the accepted model tensor dictionary. No optimizers,
training checkpoints, corpus, references, captures or evaluations are bundled.
CPU PyTorch and NumPy are required. No GPU or research checkout is needed.
This remains an experimental best-effort decoder; training completion does
not establish audible improvement or production quality.

reflector_listener/_ml is the unmodified hash-qualified inference package.
All its Python sources are retained because ModelBundle verifies its exact
source inventory; the standalone WAV CLI is not used by this application.
No soundfile dependency is needed for streaming playback. ml_runtime.py adapts
ModelBundle/StockDecoder to the audio worker. ml_transport.py adds explicit
network erasures and resynchronization without editing the qualified package.
Native workers are exclusively leased and reused across talkspurts to isolate
native RNG state and bound thread-local FFT caches. PLC muting is applied on
the delayed output timeline. The existing playback gain/PCM16 saturation remains.

MANIFEST.json pins the checkpoint, runtime source, weights and normalization.
SOURCE_PROVENANCE.json records the evaluated architecture's source hashes.
Preserve these files byte-for-byte; line-ending conversion invalidates hashes.

stock265.dll remains the pinned mbelib-neo 2.6.5 decoder used by this model.
It does not replace native/bin/mbe-neo.dll or change ordinary 8 kHz reception
or microphone encoding. Correctable received channel errors still use FEC.
Upstream: https://github.com/arancormonk/mbelib-neo
Source commit: 77a9a33b93eff910bc2bcade18c1e76fce1b7105 (v2.6.5).
License: GPL-2.0-or-later; see LICENSE.txt and native/vendor codec sources.
