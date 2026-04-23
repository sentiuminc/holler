"""Local mlx-audio inference on the multi-voice v7 checkpoint.
Generates 10 test clips per voice (katie, joe) using the same texts used
remotely on the Vast instance, so we can A/B v7-local vs v7-remote.
"""
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

CHECKPOINT = "/Users/nagy/Downloads/ivi-v7-katie-joe/checkpoint-epoch-1"
OUT_DIR = Path("/Users/nagy/Downloads/ivi-v7-katie-joe/samples-mlx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_TEXTS = [
    "What time is it in Tokyo right now?",
    "The quick brown fox jumps over the lazy dog.",
    "Before we begin, let me make sure I have this right.",
    "I think we should grab dinner somewhere downtown.",
    "There's a package waiting for you at the front desk.",
    "Would you prefer coffee or tea with your breakfast?",
    "The train arrives at the station in eight minutes.",
    "Don't worry, I'll handle everything from here.",
    "Sorry to interrupt, but I have an urgent question.",
    "It's been a long day. Let's call it a night.",
]

print(f"Loading checkpoint: {CHECKPOINT}")
t0 = time.time()
model = load(CHECKPOINT)
print(f"Loaded in {time.time() - t0:.1f}s")

for voice in ["katie", "joe"]:
    (OUT_DIR / voice).mkdir(exist_ok=True)
    print(f"\n=== {voice} ===")
    for i, text in enumerate(TEST_TEXTS, 1):
        label = f"test_{i:02d}"
        t = time.time()
        chunks = []
        for result in model.generate(
            text=text, voice=voice, language="english", temperature=0.6, stream=False
        ):
            chunks.append(np.asarray(result.audio).squeeze())
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        peak = np.abs(audio).max()
        clipped_count = int(np.sum(np.abs(audio) >= 0.999))
        out_path = OUT_DIR / voice / f"{label}.wav"
        sf.write(str(out_path), audio, 24000)
        dur = len(audio) / 24000
        flag = "⚠ CLIP" if clipped_count > 0 else ""
        print(f"  {label}: {dur:.1f}s  peak={peak:.3f}  gen {time.time()-t:.1f}s {flag} → {out_path.name}")

print(f"\nDone. Samples in {OUT_DIR}")
