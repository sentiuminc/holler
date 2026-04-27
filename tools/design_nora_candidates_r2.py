#!/usr/bin/env python3
# Nora round 2 — lower register, less bright, still Boston sharp.

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

CANDIDATES = [
    ("nora09_low_boston",
     "Low-pitched Boston woman, early thirties. Rich alto register, warm but direct. "
     "Clear non-rhotic accent — natural R-dropping. Unhurried, confident delivery. "
     "Sounds like a woman who runs the room without raising her voice. No breathiness."),

    ("nora10_contralto_sharp",
     "Deep-voiced American woman from Boston, mid-thirties. Contralto register, smooth and controlled. "
     "Subtle Boston vowels, dropped R's. Sharp intellect in the tone — direct, slightly sardonic. "
     "Like a jazz singer who became a lawyer."),

    ("nora11_husky_direct",
     "Husky-voiced Boston woman, late twenties. Lower than typical female register, "
     "slightly gravelly edge but clean. Fast, direct delivery. Strong non-rhotic accent. "
     "No sweetness, no softening — just straight talk. Sounds like she drinks black coffee."),

    ("nora12_warm_low",
     "Warm, low-pitched woman from Massachusetts, early thirties. Rich lower midrange, "
     "relaxed Boston accent. Direct but not cold — there's warmth underneath. Measured pace, "
     "clear articulation. Sounds trustworthy and grounded."),

    ("nora13_smoky_boston",
     "Smoky-voiced Boston woman, early thirties. Dark, rich timbre in the lower female range. "
     "Subtle non-rhotic accent. Deliberate, confident pacing. No vocal fry — just naturally deep. "
     "Like a late-night radio host from Cambridge."),

    ("nora14_alto_newsroom",
     "Alto-register Boston woman, early thirties. Lower pitch, authoritative, brisk delivery. "
     "Professional newsroom tone with clear Massachusetts vowels and R-dropping. "
     "Polished but not precious. Like a seasoned reporter who grew up in Dorchester."),

    ("nora15_velvet_tough",
     "Velvety low female voice, Boston accent, mid-thirties. Rich, smooth, slightly dark timbre. "
     "Non-rhotic, unhurried but efficient. Tough without being harsh — the velvet glove. "
     "Sounds like she could talk anyone into anything."),

    ("nora16_grounded_alto",
     "Grounded, steady alto. American woman from Boston, early thirties. Lower pitch, "
     "even tone, clean and direct. Subtle Boston accent — more attitude than exaggerated vowels. "
     "No upspeak, no filler, no fluff. Competent and real."),
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
    f.write(f"Nora voice candidates round 2 (lower register) — generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Target text: {TARGET_TEXT}\n\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n")
        f.write(f"  → {path.name}\n")
        f.write(f"  instruct: {instruct}\n")
print(f"\nWrote index to {index_path}")
print(f"Done — {len(results)} candidates in {OUT_DIR}")
