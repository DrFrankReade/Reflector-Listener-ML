"""Transport loss/resynchronization adapter for the pinned full360 native API.

Keep app-specific erasures outside the hashed runtime, on its exclusively
leased native worker. Never substitute synthetic channel packets for loss.
"""
import ctypes as C
import numpy as np
from ._ml._native_abi import Result
from ._ml.native import StockDecoder, DecodedFrames, _release_worker
from ._ml.observations import _snapshot, frame_conditioning, model_features


class TransportDecoder(StockDecoder):
    def _transport_call(self, action):
        with self._gate:
            if self._closed:
                raise RuntimeError('Native decoder is closed')
            try:
                return self._pool.submit(action).result()
            except BaseException:
                self._closed = True
                _release_worker(self._pool)
                raise

    def erase(self):
        return self._transport_call(self._erase)

    def _erase(self):
        pcm, bits = (C.c_float * 160)(), (C.c_char * 49)()
        result = Result(c0_errors=4, total_errors=4, flags=0x0402)
        process = self._lib.mbe_processAmbe2400Dataf
        process.argtypes = [C.c_void_p] * 6
        process.restype = C.c_int
        code = process(pcm, C.byref(result), bits, *(C.byref(p) for p in self._parameters))
        if code < 0:
            raise RuntimeError(f'ML native erasure failed with status {code}')
        waveform = np.frombuffer(bytes(pcm), dtype='<f4').copy()
        if not np.isfinite(waveform).all():
            raise ValueError('Nonfinite native erasure PCM')
        compact = model_features(self._parameters[2], waveform)[:74]
        current, pre, post = (_snapshot(p) for p in self._parameters)
        state = frame_conditioning(pre, post, current, waveform, result, int(result.flags))
        return DecodedFrames(waveform.reshape(1, 160), np.array([result.flags], np.uint32),
                             np.array([result.total_errors], np.int32), np.array([code], np.int32),
                             compact.reshape(1, 74), state.reshape(1, 371))

    def resynchronize(self):
        def reset():
            self._lib.mbe_initMbeParms(*(C.byref(p) for p in self._parameters))
            for _ in range(4):
                decoded = self._erase()
            return decoded
        return self._transport_call(reset)
