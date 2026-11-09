# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ml_runtime = Path(SPECPATH) / 'native' / 'ml_decoder'
ml_binaries = [(str(path), 'native/ml_decoder') for path in ml_runtime.glob('*.dll')]
ml_data = [(str(path), 'native/ml_decoder') for path in ml_runtime.iterdir()
           if path.is_file() and path.suffix.lower() != '.dll']
# The qualified model verifies the exact runtime source inventory at startup.
ml_sources = [(str(path), 'reflector_listener/_ml')
              for path in (Path(SPECPATH) / 'reflector_listener' / '_ml').glob('*.py')]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[('native/bin/mbe-neo.dll', 'native/bin'), *ml_binaries],
    datas=[('assets', 'assets'), ('buffers.toml', '.'),
           ('native/CODEC_VERSION.txt', 'native'), ('native/vendor', 'native/vendor'), *ml_data, *ml_sources],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='reflectorListen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
