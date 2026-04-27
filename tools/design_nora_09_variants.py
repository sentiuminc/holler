#!/usr/bin/env python3
# Nora09 — run the same prompt multiple times to get variants.

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

OUT_DIR = Path("/Users/nagy/Downloads/nora-candidates-r2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Look, I already checked the numbers twice. They don't add up, and I'm not gonna "
    "sugarcoat it for you. We need to fix this before the meeting, or we're in trouble."
)

INSTRUCT = (
    "Low-pitched Boston woman, early thirties. Rich alto register, warm but direct. "
    "Clear non-rhotic accent — natural R-dropping. Unhurried, confident delivery. "
    "Sounds like a woman who runs the room without raising her voice. No breathiness."
)

NUM_VARIANTS = 6

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

for i in range(1, NUM_VARIANTS + 1):
    name = f"nora09_v{i}"
    print(f"→ {name}")
    t = time.time()
    chunks = []
    for result in model.generate_voice_design(
        text=TARGET_TEXT,
        instruct=INSTRUCT,
        language="english",
        temperature=0.9,
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

print("Done — 6 variants of nora09")
