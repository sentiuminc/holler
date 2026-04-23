#!/usr/bin/env python3
# Generate Joe voice candidates using Qwen3-TTS-VoiceDesign.
# American male, Jarvis-inspired assistant — calm, confident, refined, measured.
# Same ~10s target text for every candidate so differences are purely voice.

import os
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

# Each instruct is a distinct voice direction. All American male, assistant-core,
# varying register / warmth / crispness / tempo. Jarvis DNA but relocated to the US.
CANDIDATES = [
    ("joe01_measured_baritone",
     "A calm, measured American male voice in his late thirties. Warm baritone register, "
     "clear articulation, unhurried pace. Speaks with quiet confidence and understated authority, "
     "like a trusted chief of staff. No vocal fry. Even tonal flow."),

    ("joe02_crisp_mid_register",
     "Mid-register American male voice, early thirties, crisp and articulate. Polished, "
     "precise diction. Professional, slightly formal, but approachable. Clean breath control, "
     "steady tempo. No nasal quality."),

    ("joe03_warm_advisor",
     "Warm, grounded American male voice in his forties. Rich lower midrange, relaxed cadence, "
     "thoughtful pauses between phrases. Sounds like a seasoned advisor — wise, reassuring, "
     "never patronizing. Slight smile in the voice."),

    ("joe04_refined_butler",
     "Refined American male voice, late thirties. Slightly lower register, polished enunciation, "
     "poised delivery. Subtly formal, like a discreet personal assistant. Controlled, elegant pacing. "
     "Neutral American accent, no regional flavor."),

    ("joe05_intelligent_dry",
     "Intelligent, dry American male voice in his late thirties to early forties. Medium register, "
     "smooth, unflustered. Minimalist — no theatrical inflection. Conveys competence and mild wit. "
     "Like a senior engineer who never raises his voice."),

    ("joe06_soft_confident",
     "Soft-spoken yet confident American male voice in his thirties. Medium-low register, "
     "warm timbre, gentle delivery. No breathiness. Conveys trust and quiet intelligence. "
     "Moderate tempo, natural phrasing."),

    ("joe07_classic_narrator",
     "Classic American male narrator voice, forties. Resonant lower midrange, crisp articulation, "
     "steady and deliberate pacing. Rich but not dramatic. The kind of voice you'd hear on a well-produced "
     "documentary. Balanced and engaging."),

    ("joe08_modern_assistant",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural contemporary "
     "American accent, friendly-but-professional tone. Concise delivery, clear consonants, neutral emotion. "
     "Feels like a smart, reliable colleague."),
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
    # generate_voice_design is a generator that yields GenerationResult objects
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

    # Concatenate chunks into single waveform
    audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
    sr = 24000
    out_path = OUT_DIR / f"{name}.wav"
    sf.write(str(out_path), audio, sr)
    dur = len(audio) / sr
    elapsed = time.time() - t
    print(f"   ✓ {dur:.1f}s audio, gen {elapsed:.1f}s → {out_path}")
    print()
    results.append((name, instruct, out_path, dur))

# Write an index.txt so the user can read descriptions alongside the samples
index_path = OUT_DIR / "index.txt"
with open(index_path, "w") as f:
    f.write(f"Joe voice candidates — generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"Target text: {TARGET_TEXT}\n\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n")
        f.write(f"  → {path.name}\n")
        f.write(f"  instruct: {instruct}\n")
print(f"\nWrote index to {index_path}")
print(f"Done — {len(results)} candidates in {OUT_DIR}")
