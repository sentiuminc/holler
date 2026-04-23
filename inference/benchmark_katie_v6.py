"""Benchmark Katie v6 (bf16) — 10 random texts, streaming TTFA + resource usage."""
import time
import os
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import psutil

from mlx_audio.tts import load
import mlx.core as mx

CHECKPOINT = str(Path.home() / "Desktop/Files/AI/holler/checkpoints/katie-v6")
OUTPUT_DIR = Path.home() / "Desktop/Files/AI/holler/samples/benchmark-katie-v6-bf16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "Oh I've never actually tried milk tea before, but maybe when I'm in Britain I will.",
    "Wait, so you're telling me the whole thing was just a misunderstanding? That's hilarious.",
    "I think the best part about working from home is just being able to cook lunch whenever.",
    "Hey do you know if there's a good coffee place near the park? I'm heading there now.",
    "Honestly I wasn't sure about it at first but now I kind of love it.",
    "So my friend just got back from Tokyo and she said the food was absolutely incredible.",
    "I'm not gonna lie, I spent way too long picking out a paint color for the bathroom.",
    "The weather's been so weird lately. Like yesterday it was sunny and today it's freezing.",
    "Can you remind me to call the dentist tomorrow? I keep forgetting.",
    "Yeah that makes sense. Let me think about it and I'll get back to you tonight.",
]

proc = psutil.Process(os.getpid())

print(f"Model: {CHECKPOINT}")
print(f"Format: bf16 (not quantized)")
print(f"Pipeline: mlx-audio streaming, temp=0.6, streaming_interval=0.1")
print(f"Voice: katie (slot 3000)")
print()

# Measure model load
mem_before = proc.memory_info().rss / 1024 / 1024
t_load = time.time()
model = load(CHECKPOINT)
load_ms = (time.time() - t_load) * 1000
mem_after = proc.memory_info().rss / 1024 / 1024
print(f"Model loaded in {load_ms:.0f}ms | RAM: {mem_before:.0f}MB -> {mem_after:.0f}MB (+{mem_after - mem_before:.0f}MB)")
print()

# Warm-up run (first inference is always slower due to JIT)
print("Warm-up run...", flush=True)
for result in model.generate(
    text="Hello there.",
    voice="katie",
    language="english",
    temperature=0.6,
    stream=True,
    streaming_interval=0.1,
):
    pass
mem_warm = proc.memory_info().rss / 1024 / 1024
print(f"Warm-up done | RAM: {mem_warm:.0f}MB (+{mem_warm - mem_after:.0f}MB from JIT)")
print()

# Benchmark
print(f"{'#':<4} {'TTFA':>6} {'Total':>7} {'Audio':>7} {'Peak':>6} {'Tok/s':>6}  Text")
print("-" * 90)

ttfas = []
totals = []

for i, text in enumerate(TEXTS):
    t0 = time.time()
    chunks = []
    ttfa = None

    for result in model.generate(
        text=text,
        voice="katie",
        language="english",
        temperature=0.6,
        stream=True,
        streaming_interval=0.1,
    ):
        if hasattr(result, 'audio'):
            if ttfa is None:
                ttfa = (time.time() - t0) * 1000
            chunks.append(result.audio)

    total_ms = (time.time() - t0) * 1000
    audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
    audio_len_s = len(audio) / 24000
    audio_len_ms = audio_len_s * 1000
    peak = np.abs(audio).max()
    tok_per_sec = (len(audio) / 24000 * 12) / (total_ms / 1000) if total_ms > 0 else 0

    ttfas.append(ttfa)
    totals.append(total_ms)

    # Save audio
    sf.write(str(OUTPUT_DIR / f"test_{i+1:02d}.wav"), audio, 24000)

    print(f"{i+1:<4} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_len_ms:>6.0f}ms {peak:>5.3f} {tok_per_sec:>5.1f}  {text[:55]}")

mem_final = proc.memory_info().rss / 1024 / 1024
print()
print(f"TTFA  — median: {np.median(ttfas):.0f}ms | mean: {np.mean(ttfas):.0f}ms | min: {min(ttfas):.0f}ms | max: {max(ttfas):.0f}ms")
print(f"Total — median: {np.median(totals):.0f}ms | mean: {np.mean(totals):.0f}ms")
print(f"RAM   — final: {mem_final:.0f}MB (process RSS)")
print(f"Saved — {OUTPUT_DIR}/")
