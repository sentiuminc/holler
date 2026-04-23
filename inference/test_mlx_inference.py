"""Test mlx-audio inference on our fine-tuned ivi voice."""
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from mlx_audio.tts import load
import mlx.core as mx

MODEL = str(Path.home() / "Downloads/ivi-v6-epoch1")
OUT = str(Path.home() / "Downloads/ivi-mlx-samples")
Path(OUT).mkdir(exist_ok=True)

print(f"[1/2] loading from {MODEL}", flush=True)
t_load = time.time()
model = load(MODEL)
print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)

texts = [
    ("greeting", "Hello! I am your personal AI assistant."),
    ("short", "Sure, let me check that for you."),
    ("question", "What kind of music are you in the mood for?"),
    ("long", "Your CPU usage is at 47 percent and you have three meetings this afternoon."),
    ("emotion", "Oh wow, that's really exciting news!"),
]

for name, text in texts:
    t0 = time.time()
    chunks = []
    first_chunk_time = None
    for result in model.generate(
        text=text,
        voice="ivi_female",
        language="english",
        temperature=0.6,
        stream=False,
    ):
        if hasattr(result, 'audio'):
            chunks.append(result.audio)
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
    elapsed = time.time() - t0
    audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
    path = f"{OUT}/e01_{name}.wav"
    sf.write(path, audio, 24000)
    audio_len = len(audio)/24000
    flag = " <-- HIT MAX?" if audio_len > 19 else ""
    print(f"  [{name}] total={elapsed:.2f}s ttfa={first_chunk_time:.2f}s len={audio_len:.2f}s{flag}", flush=True)

print("[2/2] DONE", flush=True)
