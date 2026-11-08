"""Decode packed9-byte D-STAR/AMBE2400 to WAV; ML is explicitly opt-in."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import DecoderStream
from .artifacts import atomic_json, canonical_hash, read_json, sha256_file
from .native import DEFAULT_LIBRARY, STOCK_SHA256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Raw concatenated9-byte frames, no container/header/slow-data')
    parser.add_argument('output', type=Path, help='WAV output (never replaces an unrelated file)')
    parser.add_argument('--ml48', action='store_true', help='Enable accepted full-trained48kHz model')
    parser.add_argument('--bundle', type=Path, help='Relocated inference bundle; requires --ml48')
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4, help='Torch CPU threads for this CLI process only')
    parser.add_argument('--seed', type=int, default=6110704, help='ML excitation seed (native seed remains12345)')
    parser.add_argument('--headroom-db', type=float, default=0, help='Fixed attenuation on WAV only; no AGC/limiter')
    parser.add_argument('--format', choices=('float32', 'pcm16'), default='float32')
    args = parser.parse_args(argv)
    if args.bundle and not args.ml48:
        parser.error('--bundle requires --ml48')
    if args.threads < 1 or args.seed < 0 or not math.isfinite(args.headroom_db) or args.headroom_db < 0:
        parser.error('Invalid threads, seed or headroom')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must differ')
    size = args.input.stat().st_size
    if not size or size % 9:
        parser.error('Input must contain a nonempty exact multiple of9 bytes')
    import soundfile as sf
    bundle = None
    if args.ml48:
        import torch
        from .model import ModelBundle
        torch.set_num_threads(args.threads)
        bundle = ModelBundle(args.bundle, device=args.device) if args.bundle else ModelBundle(device=args.device)
    rate = 48000 if args.ml48 else 8000
    request = dict(input_sha256=sha256_file(args.input), input_bytes=size, output_rate=rate,
                   native_sha256=STOCK_SHA256, model=canonical_hash(bundle.manifest) if bundle else None,
                   seed=args.seed, headroom_db=args.headroom_db, format=args.format,
                   wav_pcm8_scale=7/32768, package_code={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                   execution=dict(device=args.device if args.ml48 else None,
                                  torch=torch.__version__ if args.ml48 else None, numpy=np.__version__,
                                  threads=args.threads if args.ml48 else None))
    receipt = args.output.with_suffix(args.output.suffix + '.json')
    staging = args.output.with_suffix(args.output.suffix + '.part')
    recovery = staging.with_suffix(staging.suffix + '.json')
    # Recover a crash between final WAV publication and its receipt publication.
    if args.output.exists() and not receipt.exists() and recovery.exists():
        old = read_json(recovery)
        if (old.get('complete') is True and old.get('request') == request
                and old['output_sha256'] == sha256_file(args.output)):
            atomic_json(receipt, old)
    if args.output.exists():
        if receipt.exists():
            old = read_json(receipt)
            if (old.get('complete') is True and old.get('request') == request
                    and old['output_sha256'] == sha256_file(args.output)):
                print('Already complete; verified output and recipe:', args.output)
                return
        raise ValueError('Output already exists without a matching complete receipt; choose a new filename')
    if receipt.exists():
        raise ValueError('Receipt exists without output; preserve it and choose a new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix(args.output.suffix + '.lock')
    # Exclusive creation prevents two processes writing the same destination.
    # A crash leaves this small evidence file: see runtime instructions before removing.
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()) + '\n')
    peak = 0.; total = 0; above = 0; last = 0.; decoded_frames = 0
    attenuation = np.float32(10**(-args.headroom_db/20))
    try:
        with DecoderStream(enhance=args.ml48, bundle=bundle, library=args.library, seed=args.seed) as decoder:
            with args.input.open('rb') as source, sf.SoundFile(staging, 'w', samplerate=rate, channels=1,
                    format='WAV', subtype='FLOAT' if args.format == 'float32' else 'PCM_16') as dest:
                def write(chunk):
                    nonlocal peak, total, above
                    values = chunk.enhanced48 if args.ml48 else chunk.pcm8_native * np.float32(7/32768)
                    values = values * attenuation
                    if len(values):
                        peak = max(peak, float(np.max(np.abs(values))))
                        above += int(np.count_nonzero((values < -1) | (values > 32767/32768)))
                        if args.format == 'pcm16' and above:
                            raise ValueError('PCM16 would clip; rerun with more --headroom-db or --format float32')
                        dest.write(values)
                        total += len(values)
                for raw in iter(lambda: source.read(8*9), b''):
                    chunk = decoder.feed(raw)
                    write(chunk)
                    decoded_frames += len(raw)//9
                    now = time.monotonic()
                    if now-last >= 1:
                        fraction = decoded_frames/(size//9)
                        print(f'\r[{"#"*int(fraction*30):30}] {100*fraction:6.2f}% {decoded_frames}/{size//9} frames', end='', flush=True)
                        last = now
                write(decoder.finish())
                dest.flush()
                report = decoder.report()
        if total != (size//9)*(rate//50):
            raise RuntimeError('Output sample count differs')
        if sha256_file(args.input) != request['input_sha256']:
            raise ValueError('Input changed during decoding')
        with staging.open('rb+') as f:
            os.fsync(f.fileno())
        # Keep a recovery receipt before publication. Never silently overwrite a WAV.
        result = dict(complete=True, request=request, samples=total, peak=peak,
                      samples_outside_pcm16=above, output_sha256=sha256_file(staging), runtime=report)
        atomic_json(staging.with_suffix(staging.suffix + '.json'), result)
        if args.output.exists():
            raise ValueError('Output appeared while decoding; staging preserved')
        os.replace(staging, args.output)
        atomic_json(receipt, result)
        print('\r[##############################] 100.00% Complete:', args.output)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun the same command to deterministically replay the file.', file=sys.stderr)
        raise SystemExit(130)
