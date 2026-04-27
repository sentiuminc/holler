#!/usr/bin/env python3
"""Trim training clips to 100ms lead/trail silence + 20ms fade-out.

Usage:
  python trim_and_merge.py --dry-run          # preview what would happen
  python trim_and_merge.py                    # do it

Steps:
  1. Process new emotional clips from ~/Downloads/katie-emotional-t1/:
     - Reject clips with abrupt cutoffs (energy in last 50ms of audio)
     - Trim to 100ms lead + 100ms trail + 20ms fade-out
     - Copy into Katie's training-data/audio/ folder
     - Append to train.jsonl
  2. Trim existing Katie clips in-place:
     - Backup originals to audio-original/ (replaces existing backups)
     - Trim to 100ms lead + 100ms trail + 20ms fade-out
"""
import argparse
import json
import os
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 24000
WIN = int(SR * 0.01)       # 10ms window
PAD = int(SR * 0.1)        # 100ms padding
FADE = int(SR * 0.02)      # 20ms fade-out
THRESHOLD = 0.001           # RMS threshold for silence detection
CUTOFF_THRESHOLD = 0.03     # peak amplitude threshold for abrupt cutoff detection
CUTOFF_WINDOW = int(SR * 0.05)  # 50ms window at end of audio

KATIE_DIR = Path(__file__).parent.parent / "voices" / "katie" / "training-data"
EMOTIONAL_DIR = Path(os.path.expanduser("~/Downloads/katie-emotional-t1"))


def find_speech_bounds(audio):
    """Return (start, end) sample indices of speech region."""
    speech_start = 0
    for i in range(0, len(audio) - WIN, WIN):
        rms = np.sqrt(np.mean(audio[i:i+WIN]**2))
        if rms > THRESHOLD:
            speech_start = i
            break

    speech_end = len(audio)
    for i in range(len(audio) - WIN, -1, -WIN):
        rms = np.sqrt(np.mean(audio[i:i+WIN]**2))
        if rms > THRESHOLD:
            speech_end = i + WIN
            break

    return speech_start, speech_end


def trim_clip(audio):
    """Trim to 100ms pad on both sides + 20ms fade-out. Returns trimmed audio."""
    start, end = find_speech_bounds(audio)
    trim_start = max(0, start - PAD)
    trim_end = min(len(audio), end + PAD)
    trimmed = audio[trim_start:trim_end].copy()

    if len(trimmed) > FADE:
        trimmed[-FADE:] *= np.linspace(1.0, 0.0, FADE, dtype=np.float32)

    return trimmed


def has_abrupt_cutoff(audio):
    """Check if audio ends abruptly (high energy in last 50ms).

    If the model cut off mid-speech, there will be significant energy
    at the very end of the audio with no natural fade.
    """
    if len(audio) < CUTOFF_WINDOW:
        return False, 0.0

    check_region = audio[-CUTOFF_WINDOW:]
    peak = np.max(np.abs(check_region))
    return peak > CUTOFF_THRESHOLD, peak


def process_emotional_clips(dry_run=True):
    """Process new emotional clips: filter, trim, merge into Katie training data."""
    audio_dir = EMOTIONAL_DIR / "audio"
    if not audio_dir.exists():
        print("No emotional clips found at", EMOTIONAL_DIR)
        return [], []

    jsonl_path = EMOTIONAL_DIR / "train.jsonl"
    entries = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                e = json.loads(line)
                fname = os.path.basename(e["audio"])
                entries[fname] = e["text"]

    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

    accepted = []
    rejected = []

    print(f"\n=== Emotional clips: {len(files)} files ===\n")

    for f in files:
        audio, _ = sf.read(audio_dir / f)
        is_abrupt, peak = has_abrupt_cutoff(audio)

        start, end = find_speech_bounds(audio)
        lead_ms = start / SR * 1000
        trail_ms = (len(audio) - end) / SR * 1000
        speech_dur = (end - start) / SR

        if is_abrupt:
            rejected.append((f, peak))
            print(f"  REJECT {f}: abrupt cutoff peak={peak:.4f}")
        else:
            trimmed = trim_clip(audio)
            new_dur = len(trimmed) / SR
            accepted.append((f, entries.get(f, ""), trimmed))
            print(f"  OK     {f}: {len(audio)/SR:.2f}s -> {new_dur:.2f}s  (lead {lead_ms:.0f}ms, trail {trail_ms:.0f}ms)")

    print(f"\n  Accepted: {len(accepted)}, Rejected: {len(rejected)}")

    if not dry_run and accepted:
        # Find next clip number in Katie's training data
        existing_clips = [f for f in os.listdir(KATIE_DIR / "audio") if f.endswith(".wav")]
        max_num = 0
        for c in existing_clips:
            try:
                num = int(c.replace("clip_", "").replace(".wav", ""))
                max_num = max(max_num, num)
            except ValueError:
                pass

        katie_jsonl = KATIE_DIR / "train.jsonl"

        with open(katie_jsonl, "a") as jf:
            for i, (f, text, trimmed_audio) in enumerate(accepted):
                new_name = f"clip_{max_num + i + 1:04d}.wav"
                out_path = KATIE_DIR / "audio" / new_name
                sf.write(str(out_path), trimmed_audio, SR)
                jf.write(json.dumps({
                    "audio": f"./audio/{new_name}",
                    "text": text,
                    "ref_audio": "./ref.wav",
                }) + "\n")

        print(f"\n  Wrote {len(accepted)} clips as clip_{max_num+1:04d}–clip_{max_num+len(accepted):04d}")

    return accepted, rejected


def process_existing_clips(dry_run=True):
    """Trim existing Katie clips in-place, backup to audio-original/."""
    audio_dir = KATIE_DIR / "audio"
    backup_dir = KATIE_DIR / "audio-original"

    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

    print(f"\n=== Existing Katie clips: {len(files)} files ===\n")

    total_saved_ms = 0
    for f in files:
        audio, _ = sf.read(audio_dir / f)
        start, end = find_speech_bounds(audio)
        lead_ms = start / SR * 1000
        trail_ms = (len(audio) - end) / SR * 1000
        orig_dur = len(audio) / SR

        trimmed = trim_clip(audio)
        new_dur = len(trimmed) / SR
        saved_ms = (orig_dur - new_dur) * 1000
        total_saved_ms += saved_ms

        if saved_ms > 50:  # only report if meaningful trimming
            print(f"  {f}: {orig_dur:.2f}s -> {new_dur:.2f}s  (lead {lead_ms:.0f}ms, trail {trail_ms:.0f}ms)")

        if not dry_run:
            # Backup original
            backup_dir.mkdir(exist_ok=True)
            shutil.copy2(audio_dir / f, backup_dir / f)
            # Overwrite with trimmed
            sf.write(str(audio_dir / f), trimmed, SR)

    print(f"\n  Total time saved: {total_saved_ms/1000:.1f}s across {len(files)} clips")
    if dry_run:
        print(f"  (dry run — no files modified)")
    else:
        print(f"  Originals backed up to {backup_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify files")
    args = parser.parse_args()

    if args.dry_run:
        print("*** DRY RUN — no files will be modified ***")

    process_emotional_clips(dry_run=args.dry_run)
    process_existing_clips(dry_run=args.dry_run)

    if args.dry_run:
        print("\n*** Re-run without --dry-run to apply ***")


if __name__ == "__main__":
    main()
