#!/usr/bin/env python3
# Clone nora09_v6_enhanced using 1.7B-Base to get cleaner ref candidates.

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

OUT_DIR = Path("/Users/nagy/Downloads/nora-candidates-r2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

REF_AUDIO = str(OUT_DIR / "nora09_v6_enhanced.wav")
REF_TEXT = (
    "Look, I already checked the numbers twice. They don't add up, and I'm not gonna "
    "sugarcoat it for you. We need to fix this before the meeting, or we're in trouble."
)

TARGET_TEXT = (
    "Oh wow, that actually worked! Can you believe it? I honestly thought we were going to "
    "have to start over but no, it just clicked."
)

NUM_CLONES = 6

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

for i in range(1, NUM_CLONES + 1):
    name = f"nora09_clone_{i:02d}"
    print(f"→ {name}")
    t = time.time()
    chunks = []
    for result in model.generate(
        text=TARGET_TEXT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        language="english",
        temperature=0.85,
        stream=False,
    ):
        chunks.append(result.audio)

    if not chunks:
        print("   !! no audio chunks produced")
        continue

    audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
    sr = 24000
    out_path = OUT_DIR / f"{name}.wav"
    sf.write(str(out_path), audio, sr)
    dur = len(audio) / sr
    elapsed = time.time() - t
    print(f"   ✓ {dur:.1f}s audio, gen {elapsed:.1f}s → {out_path}")
    print()

print(f"Done — {NUM_CLONES} clones in {OUT_DIR}")
