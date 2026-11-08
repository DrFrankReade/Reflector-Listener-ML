# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
import ctypes as C
from pathlib import Path
from .artifacts import sha256_file

class Params(C.Structure):
    _fields_ = [('w0', C.c_float), ('L', C.c_int), ('K', C.c_int), ('Vl', C.c_int * 57)] + [
        (n, C.c_float * 57) for n in ('Ml', 'log2Ml', 'PHIl', 'PSIl')
    ] + [('gamma', C.c_float), ('tonePhase', C.c_uint32), ('swn', C.c_int),
         ('localEnergy', C.c_float), ('amplitudeThreshold', C.c_int), ('errorRate', C.c_float),
         ('errorCountTotal', C.c_int), ('errorCount4', C.c_int), ('repeatCount', C.c_int),
         ('mutingThreshold', C.c_float), ('previousUw', C.c_float * 256), ('noiseSeed', C.c_float),
         ('noiseOverlap', C.c_float * 96)]


class Result(C.Structure):
    _fields_ = [(n, C.c_int) for n in ('c0_errors', 'protected_errors', 'c4_errors', 'total_errors')] + [('flags', C.c_uint)]


def load_decoder(path, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError("Frozen decoder hash mismatch")
    lib = C.CDLL(str(Path(path).resolve()))
    lib.mbe_setThreadRngSeed.argtypes = [C.c_uint32]
    lib.mbe_setThreadRngSeed.restype = None
    lib.mbe_initMbeParms.argtypes = [C.POINTER(Params)] * 3
    lib.mbe_initMbeParms.restype = None
    lib.mbe_decodeDStarDVData.argtypes = [C.c_void_p, C.c_void_p]
    lib.mbe_decodeDStarDVData.restype = C.c_int
    lib.mbe_processAmbe3600x2400Framef.argtypes = [C.c_void_p] * 7
    lib.mbe_processAmbe3600x2400Framef.restype = C.c_int
    lib.mbe_decodeAmbe3600x2400Frame.argtypes = [C.c_void_p] * 3
    lib.mbe_decodeAmbe3600x2400Frame.restype = C.c_int
    return lib
