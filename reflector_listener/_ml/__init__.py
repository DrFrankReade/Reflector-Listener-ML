"""Stock mbelib-neo2.6.5 plus optional full-trained360ms ML enhancement.

Importing the package and using DecoderStream() require only NumPy.
ModelBundle (and Torch) are imported lazily for explicit enhancement.
"""
from .decoder import AudioChunk, DecoderStream
from .native import StockDecoder, DecodedFrames

__all__ = ['AudioChunk', 'DecodedFrames', 'DecoderStream', 'StockDecoder', 'ModelBundle']
__version__ = '1.0.0'


def __getattr__(name):
    if name == 'ModelBundle':
        from .model import ModelBundle
        return ModelBundle
    raise AttributeError(name)
