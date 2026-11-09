# Reflector Listener

Windows D-STAR reflector voice, text and picture client. Download the EXE from Releases; Python and a GPU are not required.

**The optional ML decoder remains experimental; its enhancement only helps the upper bands.** It now uses the accepted 473,130-step model; expect artifacts and uneven quality, not a finished voice model.

**Settings → Audio → ML Enhanced 48kHz Decoder** defaults on and applies to the next reception or replay. Full360 step 473,130 uses 360 ms future context (about 520 ms buffering plus processing). The standard decoder remains available, with automatic fallback on ML failure.

Download `DStar_Hosts.json` from [Pi-Star Downloads](https://www.pistar.uk/downloads/) and place it beside the EXE. Use the star to save a reflector and module under **Favorites**. Right-click a completed replay recording to save it as WAV.

Use your own licensed callsign. Settings and logs are stored locally beside the application; none are supplied in this repository.

**Settings → Audio** selects a microphone and playback device, with automatic Windows-default fallback if absent. Microphone gain and a live meter help set levels. Device choices and gains are saved; Packet Loss Concealment defaults on and AMBE FEC remains enabled.

## Build

Windows x64, Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt pyinstaller==6.22.3
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm reflectorListen.spec
```

CPU PyTorch, NumPy, model weights and both native codecs are bundled in the EXE. Training data, training tools and optimizer state are excluded. Third-party notices and corresponding codec sources are under `native/vendor/`; ML provenance is under `native/ml_decoder/`.
