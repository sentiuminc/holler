#!/usr/bin/env python3
# Iterate on joe08_modern_assistant — faster tempo, more animation.
# Same target text so differences are purely voice.

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

OUT_DIR = Path("/Users/nagy/Downloads/joe-candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Good morning. I've reviewed your schedule for today, and everything is in order. "
    "Your first meeting begins in forty-five minutes. "
    "Would you like me to prepare the car?"
)

# Base (joe08) was: "Modern American male assistant voice, early to mid thirties. Clean midrange,
# natural contemporary American accent, friendly-but-professional tone. Concise delivery, clear
# consonants, neutral emotion. Feels like a smart, reliable colleague."
# Variations: faster pace, more animation, keep the modern-assistant DNA.

VARIANTS = [
    ("joe08b_quick_animated",
     "Modern American male assistant voice, early thirties. Clean midrange, natural contemporary "
     "American accent, friendly and professional. Quicker speech tempo — brisk but never rushed. "
     "Lively inflection and natural expressiveness, with clear consonants and engaged energy. "
     "Feels like a smart colleague who's on top of the day."),

    ("joe08c_faster_warm",
     "Modern American male assistant voice, early thirties. Clean midrange, warm contemporary "
     "American accent, upbeat and confident. Faster pace, with natural rhythmic variation and "
     "subtle melodic inflection. Feels engaged and present — warm, capable, never monotone."),

    ("joe08d_bright_energetic",
     "Modern American male assistant voice, mid-thirties. Clean midrange, bright contemporary "
     "American accent. Energetic delivery with lively pitch contour and animated phrasing. "
     "Crisp consonants, brisk tempo. Conveys sharp intelligence and genuine enthusiasm while "
     "staying professional."),

    ("joe08e_punchy_expressive",
     "Modern American male assistant voice, early thirties. Clean midrange, contemporary "
     "American accent. Punchy, expressive delivery — dynamic stress on key words, natural ups "
     "and downs in intonation. Fast but clearly articulated. Feels like a trusted, sharp-witted "
     "executive assistant."),

    ("joe08f_snappy_friendly",
     "Modern American male assistant voice, early thirties. Clean midrange, friendly contemporary "
     "American accent. Snappy, efficient tempo with natural warmth. Playful melodic variation "
     "without being theatrical. Crisp, engaged, and quick-witted — like a brilliant coworker who "
     "moves at speed."),

    ("joe08g_lively_pro",
     "Modern American male assistant voice, mid-thirties. Clean midrange, polished contemporary "
     "American accent. Lively and professional — moderate-to-fast pace with expressive intonation, "
     "clear articulation, and a smile in the voice. Conveys competence plus genuine personality. "
     "Never flat, never overdone."),
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

results = []
for name, instruct in VARIANTS:
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

# Append to the index file (don't overwrite the joe01-08 index)
index_path = OUT_DIR / "index.txt"
with open(index_path, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"joe08 variations — generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Iterating on joe08_modern_assistant (selected as direction). Faster + more animated.\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n")
        f.write(f"  → {path.name}\n")
        f.write(f"  instruct: {instruct}\n")
print(f"\nAppended to {index_path}")
print(f"Done — {len(results)} variants in {OUT_DIR}")
