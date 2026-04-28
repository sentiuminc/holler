#!/usr/bin/env python3
"""Generate a batch of diverse voice candidates via VoiceDesign.

Usage:
  python generate_voice_batch.py --batch 1 --count 200
  python generate_voice_batch.py --batch 2 --count 200

Each batch gets a different random seed for variety. Outputs to
~/Downloads/voice-batches/batch_N/ with raw .wav files + prompts.txt
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Okay, so I looked into it and here's what I found. The file you were working on "
    "got saved to your Downloads folder, not your Desktop. Want me to move it over, "
    "or would you rather keep it where it is?"
)

# ============================================================================
# Attribute pools
# ============================================================================

MALE_REGISTERS = [
    "deep bass", "low baritone", "rich baritone", "mid-range baritone",
    "warm baritone", "medium register", "higher baritone", "light baritone",
    "deep mid-range", "resonant bass-baritone", "smooth mid-range",
    "tenor-range", "low mid-range", "gravelly baritone", "velvety bass",
]

FEMALE_REGISTERS = [
    "rich contralto", "deep alto", "low alto", "warm alto",
    "mid-range alto", "medium register", "lower midrange", "smooth mezzo",
    "warm mezzo-soprano", "grounded mid-range", "dark alto",
    "velvety low register", "steady mid-range", "clear mid-range",
    "bright mid-range", "clear higher register",
]

NEUTRAL_REGISTERS = [
    "clean mid-range", "balanced mid-range", "smooth medium register",
    "even mid-range", "clear neutral register", "warm mid-range",
]

TEXTURES = [
    "smooth", "warm", "clear", "crisp", "velvety", "rich", "clean",
    "rounded", "bright", "dark", "silky", "honeyed", "steady",
    "gentle", "grounded", "solid", "polished", "natural", "earthy",
    "mellow", "soft", "full-bodied", "buttery", "crystalline",
]

ROUGH_TEXTURES = [
    "slightly rough", "gravelly", "textured", "raspy edge",
    "weathered", "husky", "scratchy warmth", "gritty",
    "rough-edged", "sandpaper warmth", "lived-in",
]

PACING = [
    "unhurried delivery", "measured pace", "steady cadence",
    "relaxed delivery", "easy pace", "deliberate pacing",
    "quick but clear delivery", "efficient delivery",
    "natural conversational pace", "calm delivery",
    "brisk but warm delivery", "patient delivery",
]

PERSONALITY = [
    "helpful and genuine", "warm and trustworthy", "calm and capable",
    "friendly and direct", "reassuring and clear", "approachable and smart",
    "dependable and kind", "steady and reliable", "natural and engaging",
    "grounded and real", "confident and warm", "patient and thoughtful",
    "easygoing and helpful", "competent and friendly", "sincere and clear",
]

AGES_MALE = [
    "early thirties", "mid-thirties", "late thirties", "early forties",
    "mid-forties", "late forties", "early fifties", "mid-fifties",
]

AGES_FEMALE = [
    "late twenties", "early thirties", "mid-thirties", "late thirties",
    "early forties", "mid-forties",
]

AGES_NEUTRAL = [
    "late twenties", "early thirties", "mid-thirties",
]

DISTINGUISHING = [
    "Sounds like someone you'd trust immediately.",
    "Makes information feel like good conversation.",
    "An assistant who sounds like a real person.",
    "Sounds both intelligent and approachable.",
    "A voice you'd want on a long drive.",
    "Every word sounds considered and genuine.",
    "Sounds like the calmest person in any room.",
    "An assistant who makes problems feel smaller.",
    "Sounds like someone who actually listens.",
    "The kind of voice that puts you at ease.",
    "Sounds like someone who enjoys being helpful.",
    "A voice that makes you feel heard.",
    "An assistant who sounds fully present.",
    "Makes even boring information sound interesting.",
    "Sounds like quiet confidence.",
    "A voice you'd recognize anywhere.",
    "An assistant who sounds genuinely interested.",
]

DISTINGUISHING_ROUGH = [
    "Rough but oddly comforting.",
    "Sounds like a man with good stories.",
    "Character in every syllable.",
    "An assistant who sounds like he's lived.",
    "Texture that keeps you listening.",
    "An assistant with real character.",
]


def build_prompt(gender, use_rough=False):
    if gender == "male":
        register = random.choice(MALE_REGISTERS)
        age = random.choice(AGES_MALE)
        pronoun = "Male"
    elif gender == "female":
        register = random.choice(FEMALE_REGISTERS)
        age = random.choice(AGES_FEMALE)
        pronoun = "Female"
    else:
        register = random.choice(NEUTRAL_REGISTERS)
        age = random.choice(AGES_NEUTRAL)
        pronoun = "Gender-neutral"

    if use_rough and gender == "male":
        texture = random.choice(ROUGH_TEXTURES)
        dist = random.choice(DISTINGUISHING_ROUGH)
    else:
        texture = random.choice(TEXTURES)
        dist = random.choice(DISTINGUISHING)

    pacing = random.choice(PACING)
    personality = random.choice(PERSONALITY)

    return (
        f"{pronoun} voice assistant, {age}. {register.capitalize()}, "
        f"{texture} timbre. {pacing.capitalize()}. "
        f"{personality.capitalize()}. {dist}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, help="Batch number (affects seed)")
    parser.add_argument("--count", type=int, default=200, help="Clips per batch")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    args = parser.parse_args()

    random.seed(args.batch * 1000)

    out_dir = Path(f"/Users/nagy/Downloads/voice-batches/batch_{args.batch}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribution: 45% male, 40% female, 15% neutral
    n_male = int(args.count * 0.45)
    n_female = int(args.count * 0.40)
    n_neutral = args.count - n_male - n_female

    prompts = []
    for i in range(n_male):
        use_rough = i < int(n_male * 0.18)
        prompts.append((f"b{args.batch}_m{i+1:03d}", "male", build_prompt("male", use_rough)))
    for i in range(n_female):
        prompts.append((f"b{args.batch}_f{i+1:03d}", "female", build_prompt("female")))
    for i in range(n_neutral):
        prompts.append((f"b{args.batch}_n{i+1:03d}", "neutral", build_prompt("neutral")))

    random.shuffle(prompts)

    # Save prompts
    with open(out_dir / "prompts.txt", "w") as f:
        for name, gender, instruct in prompts:
            f.write(f"{name}\t{gender}\t{instruct}\n")

    print(f"Batch {args.batch}: {len(prompts)} clips ({n_male}m/{n_female}f/{n_neutral}n)")
    print(f"Output: {out_dir}\n")

    print(f"Loading {MODEL_ID}...", flush=True)
    t0 = time.time()
    model = load(MODEL_ID)
    print(f"Loaded in {time.time()-t0:.1f}s\n", flush=True)

    generated = 0
    fails = 0
    gen_start = time.time()

    for i, (name, gender, instruct) in enumerate(prompts):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - gen_start
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate / 60
            print(f"[{i+1}/{len(prompts)}] {generated} ok, {fails} fail, ~{eta:.0f}min left", flush=True)

        try:
            chunks = []
            for result in model.generate_voice_design(
                text=TARGET_TEXT,
                instruct=instruct,
                language="english",
                temperature=args.temperature,
                stream=False,
            ):
                chunks.append(result.audio)

            if not chunks:
                fails += 1
                continue

            audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
            sf.write(str(out_dir / f"{name}.wav"), audio, 24000)
            generated += 1
        except Exception as e:
            print(f"  !! {name}: {e}", flush=True)
            fails += 1

    total = time.time() - gen_start
    print(f"\nBatch {args.batch} done: {generated} clips, {fails} fails, {total/60:.1f} minutes")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
