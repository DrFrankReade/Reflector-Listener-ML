# Reflector Listener

Windows D-STAR reflector voice, text and picture client. Download the EXE from Releases; Python and a GPU are not required.

**The optional ML decoder is extremely early and hardly trained at all.** This is a best-effort experiment; expect artifacts and uneven quality, not a finished voice model.

Enable **Settings → Audio → ML Enhanced 48kHz Decoder**. It defaults off and applies to the next reception or replay. Pilot5C step 15,900 uses 360 ms future context (about 520 ms buffering plus processing). The standard decoder remains available, with automatic fallback on ML failure.

Use your own licensed callsign. Settings and logs are stored locally beside the application; none are supplied in this repository.

## Build

Windows x64, Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt pyinstaller==6.22.3
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm reflectorListen.spec
```

CPU PyTorch, NumPy, model weights and both native codecs are bundled in the EXE. Training data, training tools and optimizer state are excluded. Third-party notices and corresponding codec sources are under `native/vendor/`; ML provenance is under `native/ml_decoder/`.
