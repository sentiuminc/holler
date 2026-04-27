#!/usr/bin/env python3
# Generate Nora voice candidates using Qwen3-TTS-VoiceDesign.
# Boston female, sharp, no-nonsense — drops her R's. Think smart, fast, direct.

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

OUT_DIR = Path("/Users/nagy/Downloads/nora-candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Look, I already checked the numbers twice. They don't add up, and I'm not gonna "
    "sugarcoat it for you. We need to fix this before the meeting, or we're in trouble."
)

CANDIDATES = [
    ("nora01_sharp_boston",
     "Sharp, fast-talking American woman from Boston, early thirties. Clear non-rhotic accent — "
     "drops her R's naturally. Direct, assertive tone with a no-nonsense edge. Medium-high register, "
     "crisp consonants, clipped sentences. Sounds like a prosecutor in casual conversation."),

    ("nora02_warm_boston",
     "Boston woman in her late twenties, warm but direct. Soft non-rhotic accent — subtle R-dropping, "
     "not exaggerated. Quick tempo, confident delivery, friendly underneath the sharpness. "
     "Like a smart friend who always tells you the truth."),

    ("nora03_newsroom",
     "American woman from Massachusetts, early thirties. Newsroom energy — brisk, articulate, "
     "authoritative. Slight Boston vowels (broad A, dropped R's) but polished. Sounds like she "
     "grew up in Southie but went to a good school. No vocal fry."),

    ("nora04_tough_tender",
     "Boston woman, mid-thirties. Lower register for a woman, a little gravelly but not raspy. "
     "Tough exterior, warm underneath. Non-rhotic accent, unhurried but efficient delivery. "
     "Sounds like a detective who's seen everything but still cares."),

    ("nora05_quick_wit",
     "Fast, witty American woman from Boston, late twenties. Higher register, bright timbre, "
     "rapid-fire delivery. Strong non-rhotic accent. Sounds like she'd win any argument — not mean, "
     "just quicker than everyone else. Sharp sibilants, clean enunciation."),

    ("nora06_dry_professional",
     "Dry, professional Boston woman, early forties. Medium register, controlled pace, "
     "understated delivery. Subtle non-rhotic features. Conveys intelligence through restraint — "
     "every word chosen carefully. Like a senior partner at a law firm."),

    ("nora07_energetic_boston",
     "Energetic Boston woman, late twenties. Bright, punchy delivery, noticeable Boston accent — "
     "clear R-dropping, broad vowels. Friendly but no-nonsense, like a sports commentator or "
     "a bartender who runs the place. Fast tempo, strong presence."),

    ("nora08_cool_collected",
     "Cool, collected American woman from Boston, early thirties. Mid-register, smooth, steady. "
     "Subtle Boston accent — more in the vowels than the consonants. Unflappable tone, slight "
     "sardonic edge. Like a trauma surgeon giving you instructions."),
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

results = []
for name, instruct in CANDIDATES:
    print(f"→ {name}")
    print(f"   instruct: {instruct[:90]}...")
    t = time.time()
    chunks = []
    for result in model.generate_voice_design(
        text=TARGET_TEXT,
        instruct=instruct,
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
    results.append((name, instruct, out_path, dur))

index_path = OUT_DIR / "index.txt"
with open(index_path, "w") as f:
    f.write(f"Nora voice candidates — generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Target text: {TARGET_TEXT}\n\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n")
        f.write(f"  → {path.name}\n")
        f.write(f"  instruct: {instruct}\n")
print(f"\nWrote index to {index_path}")
print(f"Done — {len(results)} candidates in {OUT_DIR}")
