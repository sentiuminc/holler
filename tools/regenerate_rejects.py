#!/usr/bin/env python3
"""Regenerate rejected/undecided clips with enhanced ref at temp 0.85."""
import json
import os
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

KATIE_DIR = Path(__file__).parent.parent / "voices" / "katie" / "training-data"
REF_AUDIO = os.path.expanduser("~/Downloads/ref-variants/4_C_louder_presence.wav")
REF_TEXT = "Oh my god, I just figured it out! The whole time we were looking at it wrong — it was right there in front of us! I can't believe how simple it actually is!"
SR = 24000
TEMP = 0.85

# Load curation — regenerate everything that isn't "keep"
with open(KATIE_DIR / "curation.json") as f:
    curation = json.load(f)
keep_clips = {k for k, v in curation.items() if v == "keep"}
all_clips = {f"clip_{i:04d}.wav" for i in range(1, 476)}
regen_clips = sorted(all_clips - keep_clips)

# Load texts from train.jsonl
texts = {}
with open(KATIE_DIR / "train.jsonl") as f:
    for line in f:
        e = json.loads(line)
        fname = os.path.basename(e["audio"])
        texts[fname] = e["text"]

print(f"Regenerating {len(regen_clips)} clips at temp {TEMP}")
print(f"Ref: {REF_AUDIO}")

# Back up
backup_dir = KATIE_DIR / "audio-regen-backup-2"
backup_dir.mkdir(exist_ok=True)
for fname in regen_clips:
    src = KATIE_DIR / "audio" / fname
    if src.exists():
        shutil.copy2(src, backup_dir / fname)
print(f"Backed up to {backup_dir}/")

# Load model
print("Loading Qwen3-TTS 1.7B Base (8-bit)...")
import time
t0 = time.time()
from mlx_audio.tts import load
model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
print(f"Model loaded in {time.time()-t0:.1f}s")

print("Warmup...")
for _ in model.generate(text="Hello!", ref_audio=REF_AUDIO, ref_text=REF_TEXT, language="en", temperature=TEMP):
    pass
print("Done.\n")

WIN = int(SR * 0.01)
PAD = int(SR * 0.1)
FADE = int(SR * 0.02)
THRESHOLD = 0.001
CUTOFF_THRESHOLD = 0.03
CUTOFF_WINDOW = int(SR * 0.05)

failed = []
regenerated = 0

for i, fname in enumerate(regen_clips):
    text = texts.get(fname, "")
    if not text:
        print(f"[{i+1}/{len(regen_clips)}] {fname}: NO TEXT, skipping")
        failed.append(fname)
        continue

    print(f"[{i+1}/{len(regen_clips)}] {fname}: {text[:60]}...", end=" ", flush=True)

    try:
        chunks = []
        for result in model.generate(
            text=text, ref_audio=REF_AUDIO, ref_text=REF_TEXT,
            language="en", temperature=TEMP,
        ):
            audio = np.array(result.audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            chunks.append(audio)

        full_audio = np.concatenate(chunks)
        full_audio = np.concatenate([full_audio, np.zeros(SR, dtype=np.float32)])

        # Check abrupt cutoff
        silence_start = len(full_audio) - SR
        check_region = full_audio[silence_start - CUTOFF_WINDOW:silence_start]
        if np.max(np.abs(check_region)) > CUTOFF_THRESHOLD:
            print("RETRY...", end=" ", flush=True)
            chunks = []
            for result in model.generate(
                text=text, ref_audio=REF_AUDIO, ref_text=REF_TEXT,
                language="en", temperature=TEMP,
            ):
                audio = np.array(result.audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()
                chunks.append(audio)
            full_audio = np.concatenate(chunks)
            full_audio = np.concatenate([full_audio, np.zeros(SR, dtype=np.float32)])

        # Trim
        speech_start = 0
        for j in range(0, len(full_audio) - WIN, WIN):
            if np.sqrt(np.mean(full_audio[j:j+WIN]**2)) > THRESHOLD:
                speech_start = j
                break

        speech_end = len(full_audio)
        for j in range(len(full_audio) - WIN, -1, -WIN):
            if np.sqrt(np.mean(full_audio[j:j+WIN]**2)) > THRESHOLD:
                speech_end = j + WIN
                break

        trim_start = max(0, speech_start - PAD)
        trim_end = min(len(full_audio), speech_end + PAD)
        trimmed = full_audio[trim_start:trim_end].copy()

        if len(trimmed) > FADE:
            trimmed[-FADE:] *= np.linspace(1.0, 0.0, FADE, dtype=np.float32)

        sf.write(str(KATIE_DIR / "audio" / fname), trimmed, SR)
        regenerated += 1
        print(f"OK ({len(trimmed)/SR:.2f}s)")

    except Exception as e:
        print(f"FAILED: {e}")
        failed.append(fname)

# Reset curation for regenerated clips
for fname in regen_clips:
    if fname not in failed:
        curation.pop(fname, None)
with open(KATIE_DIR / "curation.json", "w") as f:
    json.dump(curation, f, indent=2)

print(f"\n{'='*60}")
print(f"Regenerated: {regenerated} | Failed: {len(failed)}")
print(f"Curation reset — will appear as undecided in Tinder")
if failed:
    print(f"Failed: {failed}")
