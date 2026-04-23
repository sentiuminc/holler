"""Test mlx-audio STREAMING inference for real TTFA."""
import time
from pathlib import Path
import numpy as np
import soundfile as sf

from mlx_audio.tts import load
import mlx.core as mx

MODEL = str(Path.home() / "Downloads/ivi-v6-epoch1")

print(f"loading bf16...", flush=True)
model = load(MODEL)
print("loaded", flush=True)

texts = [
    "Hello! I am your personal AI assistant.",
    "Sure, let me check that for you.",
    "What kind of music are you in the mood for?",
]

for text in texts:
    t0 = time.time()
    chunks = []
    ttfa = None
    for result in model.generate(
        text=text,
        voice="ivi_female",
        language="english",
        temperature=0.6,
        stream=True,
        streaming_interval=0.1,
    ):
        if hasattr(result, 'audio'):
            if ttfa is None:
                ttfa = (time.time() - t0) * 1000
            chunks.append(result.audio)
    total_ms = (time.time() - t0) * 1000
    audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
    audio_len_ms = len(audio) / 24000 * 1000
    print(f"[{text[:40]!r}] TTFA={ttfa:.0f}ms  total={total_ms:.0f}ms  audio={audio_len_ms:.0f}ms", flush=True)
