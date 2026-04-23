#!/usr/bin/env python3
# Subtle nudges on joe08 — just slightly faster + slightly more natural inflection.
# Keep the original joe08 DNA, don't push into "animated" territory.

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

OUT_DIR = Path("/Users/nagy/Downloads/joe-candidates")
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Good morning. I've reviewed your schedule for today, and everything is in order. "
    "Your first meeting begins in forty-five minutes. "
    "Would you like me to prepare the car?"
)

SUBTLE_VARIANTS = [
    ("joe08h_subtle_nudge",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
     "consonants, natural conversational rhythm — slightly faster pace than deliberate, with "
     "subtle melodic inflection. Feels like a smart, reliable colleague who's engaged and present."),

    ("joe08i_slightly_quicker",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Moderate-to-brisk pace — "
     "efficient without sounding rushed. Clear consonants, neutral emotion with a trace of warmth. "
     "Feels like a smart, reliable colleague."),

    ("joe08j_natural_flow",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
     "consonants, natural conversational flow — subtle rise and fall in pitch on longer phrases. "
     "Neutral emotion. Feels like a smart, reliable colleague."),

    ("joe08k_touch_more_alive",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
     "consonants. Calm and composed, but with a touch of natural engagement in the voice — "
     "not flat, not theatrical. Feels like a smart, reliable colleague who's paying attention."),
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

results = []
for name, instruct in SUBTLE_VARIANTS:
    print(f"→ {name}")
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
        print("   !! no audio")
        continue

    audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
    out_path = OUT_DIR / f"{name}.wav"
    sf.write(str(out_path), audio, 24000)
    dur = len(audio) / 24000
    print(f"   ✓ {dur:.1f}s audio, gen {time.time()-t:.1f}s → {out_path.name}")
    results.append((name, instruct, out_path, dur))

index_path = OUT_DIR / "index.txt"
with open(index_path, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"joe08 subtle variants — {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write("Tiny nudges on original joe08 (previous b-g were too expressive).\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n  → {path.name}\n  instruct: {instruct}\n")

print(f"\nDone — {len(results)} subtle variants")
