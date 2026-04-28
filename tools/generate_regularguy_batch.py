#!/usr/bin/env python3
"""Generate 150 'regular guy' male voices — mid-range, smooth, natural.

Based on analysis of passing vs failing males: velvety/smooth/warm pass,
gravelly/rough/crisp fail. Mid-range registers, ages 30s-40s.
"""

import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

random.seed(9999)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Okay, so I looked into it and here's what I found. The file you were working on "
    "got saved to your Downloads folder, not your Desktop. Want me to move it over, "
    "or would you rather keep it where it is?"
)

OUT_DIR = Path("/Users/nagy/Downloads/voice-batches/batch_regularguy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTERS = [
    "mid-range baritone", "medium register", "warm baritone", "light baritone",
    "smooth mid-range", "warm mid-range", "clear baritone", "natural baritone",
    "even mid-range", "comfortable baritone", "relaxed baritone", "balanced baritone",
]

TEXTURES = [
    "smooth", "warm", "clean", "natural", "velvety", "clear", "rich",
    "silky", "rounded", "soft", "mellow", "gentle", "polished", "easy",
    "buttery", "full", "honeyed", "bright",
]

PACING = [
    "relaxed delivery", "natural conversational pace", "easy pace",
    "calm delivery", "measured pace", "steady cadence", "unhurried delivery",
    "patient delivery", "comfortable pace",
]

PERSONALITY = [
    "helpful and genuine", "warm and trustworthy", "friendly and natural",
    "calm and capable", "approachable and real", "sincere and clear",
    "natural and engaging", "confident and warm", "easygoing and helpful",
    "grounded and friendly",
]

AGES = [
    "late twenties", "early thirties", "mid-thirties", "late thirties",
    "early forties",
]

DISTINGUISHING = [
    "Sounds like a regular guy who happens to be really helpful.",
    "A voice you would hear from a friend.",
    "Sounds like someone you already know.",
    "An assistant who sounds completely natural.",
    "Like talking to a smart, calm friend.",
    "Sounds like a normal person, not a character.",
    "The kind of voice that blends into conversation.",
    "An assistant who sounds genuinely real.",
    "Nothing flashy, just naturally pleasant.",
    "Sounds like the guy next door who knows everything.",
]

prompts = []
for i in range(150):
    prompt = (
        f"Male voice assistant, {random.choice(AGES)}. "
        f"{random.choice(REGISTERS).capitalize()}, {random.choice(TEXTURES)} timbre. "
        f"{random.choice(PACING).capitalize()}. "
        f"{random.choice(PERSONALITY).capitalize()}. "
        f"{random.choice(DISTINGUISHING)}"
    )
    prompts.append((f"rg_{i+1:03d}", prompt))

with open(OUT_DIR / "prompts.txt", "w") as f:
    for name, p in prompts:
        f.write(f"{name}\tmale\t{p}\n")

# Skip already-generated clips
existing = {f.stem for f in OUT_DIR.glob("*.wav")}
prompts = [(n, p) for n, p in prompts if n not in existing]
print(f"Skipping {150 - len(prompts)} already generated, {len(prompts)} remaining")

print(f"Generating {len(prompts)} regular guy voices at temp 0.8...")
print(f"Output: {OUT_DIR}", flush=True)

model = load(MODEL_ID)
print("Model loaded", flush=True)

gen_start = time.time()
generated = 0
fails = 0
for i, (name, instruct) in enumerate(prompts):
    if (i + 1) % 10 == 0:
        elapsed = time.time() - gen_start
        rate = (i + 1) / elapsed
        eta = (len(prompts) - i - 1) / rate / 60
        print(f"[{i+1}/{len(prompts)}] {generated} ok, {fails} fail, ~{eta:.0f}min left", flush=True)
    try:
        chunks = []
        for result in model.generate_voice_design(
            text=TARGET_TEXT, instruct=instruct, language="english",
            temperature=0.8, stream=False,
        ):
            chunks.append(result.audio)
        if not chunks:
            fails += 1
            continue
        audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
        sf.write(str(OUT_DIR / f"{name}.wav"), audio, 24000)
        generated += 1
    except Exception as e:
        print(f"  !! {name}: {e}", flush=True)
        fails += 1

print(f"\nDone: {generated} clips, {fails} fails, {(time.time()-gen_start)/60:.1f} min")
