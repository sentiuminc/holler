#!/usr/bin/env python3
# Expand around joe08h_subtle_nudge — the picked direction.
# Two tracks:
#   1) Run the exact 08h prompt 4x — sampling variance at temp=0.9 (joe08h1..4)
#   2) Tight sibling prompts — same DNA, micro-tweaks (joe08ha..d)
# Goal: give Chris a broader pool inside the 08h zone to pick the final Joe.

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

# The exact 08h prompt — run 4x for sampling variance
BASE_08H = (
    "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
    "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
    "consonants, natural conversational rhythm — slightly faster pace than deliberate, with "
    "subtle melodic inflection. Feels like a smart, reliable colleague who's engaged and present."
)

# Tight sibling prompts — 08h flavor, minor wording shifts
SIBLINGS = [
    ("joe08ha_warm_nudge",
     "Modern American male assistant voice, early to mid thirties. Clean warm midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
     "consonants, natural conversational rhythm — slightly faster pace than deliberate, with "
     "subtle melodic inflection. Feels like a smart, reliable colleague who's engaged and present."),

    ("joe08hb_grounded_nudge",
     "Modern American male assistant voice, early to mid thirties. Clean midrange with a grounded "
     "low edge, natural contemporary American accent, friendly-but-professional tone. Concise "
     "delivery, clear consonants, natural conversational rhythm — slightly quicker than deliberate, "
     "with subtle melodic inflection. A smart, reliable colleague who's present and paying attention."),

    ("joe08hc_smooth_nudge",
     "Modern American male assistant voice, early to mid thirties. Smooth clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, crisp "
     "consonants, natural conversational rhythm — slightly faster pace, with subtle melodic "
     "inflection. Feels like a smart, reliable colleague."),

    ("joe08hd_attentive_nudge",
     "Modern American male assistant voice, early to mid thirties. Clean midrange, natural "
     "contemporary American accent, friendly-but-professional tone. Concise delivery, clear "
     "consonants, natural conversational rhythm — slightly faster pace than deliberate, with "
     "subtle natural inflection. Attentive and composed — feels like a smart, reliable colleague."),
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")
print()

results = []

def generate(name, instruct):
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
        return None
    audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
    out_path = OUT_DIR / f"{name}.wav"
    sf.write(str(out_path), audio, 24000)
    dur = len(audio) / 24000
    print(f"   ✓ {dur:.1f}s audio, gen {time.time()-t:.1f}s → {out_path.name}")
    return (name, instruct, out_path, dur)

# Four reruns of exact 08h prompt
for i in range(1, 5):
    r = generate(f"joe08h{i}_rerun", BASE_08H)
    if r:
        results.append(r)

# Tight siblings
for name, instruct in SIBLINGS:
    r = generate(name, instruct)
    if r:
        results.append(r)

index_path = OUT_DIR / "index.txt"
with open(index_path, "a") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"joe08h expansion — {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write("joe08h selected as direction. Reruns (h1-h4) + tight siblings (ha-hd).\n")
    f.write("=" * 80 + "\n")
    for name, instruct, path, dur in results:
        f.write(f"\n{name}  ({dur:.1f}s)\n  → {path.name}\n  instruct: {instruct}\n")

print(f"\nDone — {len(results)} joe08h expansions")
