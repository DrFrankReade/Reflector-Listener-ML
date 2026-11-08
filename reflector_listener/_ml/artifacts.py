"""Small standard-library artifact utilities; no training dependencies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as dest:
            dest.write(data)
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def checked_file(root, binding):
    root = Path(root).resolve()
    path = (root / binding['path']).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes bundle')
    if path.stat().st_size != binding['bytes'] or sha256_file(path) != binding['sha256']:
        raise ValueError('Artifact integrity mismatch: ' + str(path))
    return path
