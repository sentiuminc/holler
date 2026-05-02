#!/usr/bin/env python3
"""Generate 1000 diverse voice candidates, enhance, analyze, keep top 200.

Distribution:
- 450 male (varied registers, textures)
- 400 female (skewed low/mid, minimal high-pitched)
- 150 androgynous/neutral

Prompts are built from attribute pools for maximum diversity.
"""

import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

random.seed(42)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TARGET_TEXT = (
    "Okay, so I looked into it and here's what I found. The file you were working on "
    "got saved to your Downloads folder, not your Desktop. Want me to move it over, "
    "or would you rather keep it where it is?"
)

OUT_DIR = Path("/Users/nagy/Downloads/voice-1000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Attribute pools — combined randomly to create diverse prompts
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
    # Only a few higher ones
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
    "Sounds like he could fix anything and explain it clearly.",
    "Every word sounds considered and genuine.",
    "Sounds like the calmest person in any room.",
    "An assistant who makes problems feel smaller.",
    "Sounds like someone who actually listens.",
    "The kind of voice that puts you at ease.",
    "Sounds like someone who enjoys being helpful.",
    "A voice that makes you feel heard.",
    "Sounds like a friend who happens to know everything.",
    "An assistant who sounds fully present.",
    "Makes even boring information sound interesting.",
    "Sounds like quiet confidence.",
    "A voice you'd recognize anywhere.",
    "Sounds like someone with nothing to prove.",
    "An assistant who sounds genuinely interested.",
]

DISTINGUISHING_ROUGH = [
    "Rough but oddly comforting.",
    "Sounds like a man with good stories.",
    "Character in every syllable.",
    "An assistant who sounds like he's lived.",
    "Rough edges make it interesting.",
    "Sounds like late nights and early mornings.",
    "Texture that keeps you listening.",
    "An assistant with real character.",
]


def build_prompt(gender, use_rough=False):
    """Build a random voice assistant instruct prompt."""
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

    prompt = (
        f"{pronoun} voice assistant, {age}. {register.capitalize()}, "
        f"{texture} timbre. {pacing.capitalize()}. "
        f"{personality.capitalize()}. {dist}"
    )
    return prompt


def generate_prompt_list():
    """Generate 1000 diverse prompts."""
    prompts = []

    # 450 male
    for i in range(450):
        use_rough = i < 80  # ~80 rough/textured males
        prompts.append((f"m{i+1:04d}", "male", build_prompt("male", use_rough)))

    # 400 female (skewed low/mid — FEMALE_REGISTERS already skewed)
    for i in range(400):
        prompts.append((f"f{i+1:04d}", "female", build_prompt("female")))

    # 150 neutral/androgynous
    for i in range(150):
        prompts.append((f"n{i+1:04d}", "neutral", build_prompt("neutral")))

    random.shuffle(prompts)
    return prompts


# ============================================================================
# Main
# ============================================================================

from mlx_audio.tts import load

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s\n")

prompts = generate_prompt_list()
print(f"Generated {len(prompts)} prompts")
print(f"  Male: {sum(1 for _,g,_ in prompts if g=='male')}")
print(f"  Female: {sum(1 for _,g,_ in prompts if g=='female')}")
print(f"  Neutral: {sum(1 for _,g,_ in prompts if g=='neutral')}")
print(f"\nGenerating to {OUT_DIR}...\n", flush=True)

# Save prompt index
with open(OUT_DIR / "prompts.txt", "w") as f:
    for name, gender, instruct in prompts:
        f.write(f"{name}\t{gender}\t{instruct}\n")

results = []
fails = 0
for i, (name, gender, instruct) in enumerate(prompts):
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / (elapsed - 4)  # subtract load time
        eta = (len(prompts) - i - 1) / rate / 60
        print(f"[{i+1}/{len(prompts)}] ~{eta:.0f}min remaining, {fails} fails", flush=True)

    try:
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
            fails += 1
            continue

        audio = np.concatenate([np.asarray(c).squeeze() for c in chunks])
        sf.write(str(OUT_DIR / f"{name}.wav"), audio, 24000)
        results.append((name, gender, instruct, len(audio) / 24000))
    except Exception as e:
        print(f"  !! {name} error: {e}", flush=True)
        fails += 1

total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"GENERATION DONE")
print(f"  {len(results)} clips generated, {fails} failures")
print(f"  Total time: {total_time/60:.1f} minutes")
print(f"  Output: {OUT_DIR}")
print(f"{'='*60}\n", flush=True)

# ============================================================================
# Phase 2: Enhance all clips
# ============================================================================

print("Loading enhancement pipeline...", flush=True)
sys.path.insert(0, str(Path(__file__).parent))
from enhance_clips import enhance_full

FEMALE_SET = {"female"}
MALE_SET = {"male"}

gender_map = {name: gender for name, gender, _, _ in results}

wavs = sorted(OUT_DIR.glob("*.wav"))
enhanced_dir = OUT_DIR / "enhanced"
enhanced_dir.mkdir(exist_ok=True)

print(f"Enhancing {len(wavs)} clips...", flush=True)
for i, wav in enumerate(wavs):
    if (i + 1) % 50 == 0:
        print(f"  enhanced [{i+1}/{len(wavs)}]", flush=True)
    name = wav.stem
    gender = gender_map.get(name, "female")
    enh_gender = "female" if gender in ("female", "neutral") else "male"

    data, sr = sf.read(str(wav))
    enhanced = enhance_full(data, sr, gender=enh_gender)
    sf.write(str(enhanced_dir / wav.name), enhanced, sr)

print(f"  enhanced [{len(wavs)}/{len(wavs)}] done.\n", flush=True)

# ============================================================================
# Phase 3: Analyze enhanced clips
# ============================================================================

print("Analyzing enhanced clips...", flush=True)
from analyze_voice_quality import analyze_clip
import json

enhanced_wavs = sorted(enhanced_dir.glob("*.wav"))
analyses = []
for i, wav in enumerate(enhanced_wavs):
    if (i + 1) % 50 == 0:
        print(f"  analyzed [{i+1}/{len(enhanced_wavs)}]", flush=True)
    clip = analyze_clip(wav)
    clip["file"] = wav.name
    clip["gender"] = gender_map.get(wav.stem, "unknown")
    analyses.append(clip)

print(f"  analyzed [{len(enhanced_wavs)}/{len(enhanced_wavs)}] done.\n", flush=True)

# Save full analysis
with open(OUT_DIR / "analysis.json", "w") as f:
    json.dump(analyses, f, indent=2)

# ============================================================================
# Phase 4: Score and rank — keep top 200
# ============================================================================

def quality_score(clip):
    """Higher = better. Combines key metrics into one ranking score."""
    score = 0.0
    # DNSMOS is the strongest perceptual quality signal (0-5 scale)
    score += clip.get("dnsmos_ovrl", 0) * 20  # 0-100 points
    # HNR: higher = cleaner voice (typically 8-25 dB)
    score += min(clip.get("hnr_db", 0), 25) * 2  # 0-50 points
    # Harshness: lower = better (0-0.15 typical range)
    score -= clip.get("harsh_2_4k", 0) * 500  # -75 to 0 points
    # Sibilance: lower = better
    score -= clip.get("sib_4_10k", 0) * 300  # -9 to 0 points
    # Peak: prefer -8 to -4, penalize extremes
    peak = clip.get("peak_db", -10)
    if peak > -3:
        score -= 20  # too hot
    elif peak < -15:
        score -= 10  # too quiet
    return score

for clip in analyses:
    clip["quality_score"] = quality_score(clip)

# Sort by quality score, descending
analyses.sort(key=lambda c: c["quality_score"], reverse=True)

# Keep top 200
top200 = analyses[:200]
bottom = analyses[200:]

# Move top 200 to final folder
final_dir = OUT_DIR / "top200"
final_dir.mkdir(exist_ok=True)
rejected_dir = OUT_DIR / "rejected"
rejected_dir.mkdir(exist_ok=True)

import shutil
for clip in top200:
    src = enhanced_dir / clip["file"]
    if src.exists():
        shutil.copy2(str(src), str(final_dir / clip["file"]))

for clip in bottom:
    src = enhanced_dir / clip["file"]
    if src.exists():
        shutil.move(str(src), str(rejected_dir / clip["file"]))

# Summary
print(f"{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Generated: {len(results)}")
print(f"Enhanced:  {len(enhanced_wavs)}")
print(f"Top 200:   {final_dir}")
print(f"Rejected:  {rejected_dir} ({len(bottom)} clips)")
print()

# Gender breakdown of top 200
genders = {}
for clip in top200:
    g = clip.get("gender", "unknown")
    genders[g] = genders.get(g, 0) + 1
print("Top 200 gender breakdown:")
for g, n in sorted(genders.items()):
    print(f"  {g}: {n}")

# Quality stats
scores = [c["quality_score"] for c in top200]
print(f"\nTop 200 quality scores: min={min(scores):.1f}, max={max(scores):.1f}, mean={np.mean(scores):.1f}")

# Metric summary for top 200
print(f"\nTop 200 metric averages:")
for metric in ["peak_db", "harsh_2_4k", "sib_4_10k", "hnr_db", "dnsmos_ovrl"]:
    vals = [c[metric] for c in top200 if metric in c]
    if vals:
        label = metric
        if "harsh" in metric or "sib" in metric:
            print(f"  {label}: {np.mean(vals)*100:.1f}%")
        else:
            print(f"  {label}: {np.mean(vals):.1f}")

print(f"\nDone! Listen to {final_dir}")
