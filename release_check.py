"""Offline packaged-runtime diagnostic; does not transmit or load user settings."""
import json
from pathlib import Path

def run(destination):
    from reflector_listener.decoder import AmbeDecoder, BitOrder
    from reflector_listener.ml_runtime import Runtime
    from reflector_listener.protocols import NULL_AMBE_FRAME
    import numpy as np
    import torch
    import tkinter as tk
    from PIL import Image
    from reflector_listener.audio import WaveOutPlayer
    from reflector_listener.audio_devices import enumerate_audio_devices
    from reflector_listener.preferences import Preferences
    devices = {direction: enumerate_audio_devices(direction) for direction in ('input', 'output')}
    assert Preferences().packet_loss_concealment
    codec=AmbeDecoder()
    assert len(codec.decode_frame(NULL_AMBE_FRAME,BitOrder.LSB_FIRST).pcm)==320
    codec.close()
    runtime=Runtime()
    stream=runtime.new_stream()
    parts=[]
    for _ in range(29):parts.extend(stream.decode(NULL_AMBE_FRAME))
    parts.extend(stream.finish())
    assert sum(map(len,parts))==29*960
    assert all(np.isfinite(p).all() for p in parts)
    stream.close()
    root=tk.Tk(); root.withdraw(); root.update(); root.destroy()
    errors=[]
    player=WaveOutPlayer(sample_rate=48000,on_error=errors.append,device_id='missing-self-test-device')
    generation=player.begin_stream()
    for _ in range(3):player.submit(bytes(1920),generation)
    assert player.wait_drained(5)
    player.close()
    assert not errors,errors
    Path(destination).write_text(json.dumps({'passed':True,'torch':torch.__version__,'cuda':torch.version.cuda,'samples48':29*960,'tk':True,'waveout48':True,'device_fallback':True,'devices':{direction:len(items) for direction,items in devices.items()}}),encoding='utf-8')
