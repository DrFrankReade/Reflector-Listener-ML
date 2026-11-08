"""Load the accepted model's small, pinned inference-only bundle."""
from pathlib import Path

from .artifacts import checked_file, read_json, canonical_hash
from .native import STOCK_SHA256

MODEL_ID = 'v2-full360-step473130'
CHECKPOINT_SHA256 = '7ebb87ad5465d102c104dedce3968131b5ebdbefc046f952928d92b56b42d768'
RECIPE_ID = '873251d1aef6dd038c3382a44e80ca08ce3bbef190b1c7621e135ac80fc318c0'
DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / 'models/full360-v1'


class ModelBundle:
    """Share immutable weights across streams; each stream owns its context/noise.

    Torch is optional for the stock path and imported only here. This class
    does not change process-global Torch thread counts or global random seeds.
    Set CPU thread policy in the hosting application before constructing it.
    """

    def __init__(self, path=DEFAULT_BUNDLE, *, device='cpu'):
        self.path = Path(path).resolve()
        m = read_json(self.path / 'MANIFEST.json')
        expected = dict(format='ambe-ml-inference-v1', model_id=MODEL_ID,
                        architecture='SpeechReconstructorV2P5C', updates=473130,
                        source_checkpoint_sha256=CHECKPOINT_SHA256, training_recipe_identity=RECIPE_ID,
                        stock_decoder_sha256=STOCK_SHA256, past_frames=36, future_frames=18,
                        block_frames=8, input_rate=8000, output_rate=48000, pcm_scale=7/32768)
        if any(m.get(k) != v for k, v in expected.items()):
            raise ValueError('Unsupported model identity/architecture/context contract')
        package = Path(__file__).resolve().parent
        actual_sources = {p.name for p in package.glob('*.py')}
        if {b['path'] for b in m['runtime_code']} != actual_sources:
            raise ValueError('Runtime source inventory differs from model bundle')
        for binding in m['runtime_code']:
            checked_file(package, binding)
        weights = checked_file(self.path, m['weights'])
        self.normalization = read_json(checked_file(self.path, m['normalization']))
        self.state_normalization = read_json(checked_file(self.path, m['state_normalization']))
        if canonical_hash(self.state_normalization) != 'bc32edca97e69f2211f86a35c715769af3a430b56112646a996113d6312fb846':
            raise ValueError('State normalization differs from accepted training')
        if m['normalization']['sha256'] != '582554c98f16a6f07ea82a26d976ada42cd60ae6101247eb72a8169d10c0d4bb':
            raise ValueError('Compact normalization differs from accepted training')
        import numpy as np
        import torch
        from .observations import normalize_features
        from ._model_v5c import SpeechReconstructorV2P5C
        mean, std = (np.asarray(self.normalization[n]) for n in ('mean', 'std'))
        if mean.shape != (74,) or std.shape != (74,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError('Invalid compact normalization')
        normalize_features(np.zeros((0, 371), np.float32), self.state_normalization)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError('Invalid inference weights')
        # Constructor initialization must not consume the host's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = SpeechReconstructorV2P5C()
        model.load_state_dict(state, strict=True)
        if sum(p.numel() for p in model.parameters()) != 798159:
            raise ValueError('Model parameter count differs')
        self.device = str(torch.device(device))
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.manifest = m

    def stream(self, *, seed=6110704):
        from ._context import FutureContextStream
        return FutureContextStream(self.model, self.normalization, self.state_normalization,
                                   device=self.device, seed=seed, future_frames=18)
