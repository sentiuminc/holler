#!/usr/bin/env python3
"""Build combined multi-voice train.jsonl for GPU training.

Auto-discovers voices from voices/ that have train_curated.jsonl.
Rewrites audio/ref_audio paths to absolute on-instance paths and tags each
sample with `voice_name`. Output is ready for Qwen3-TTS tokenize step.

Run locally before uploading to the GPU instance.

Usage:
  python training/build_combined_jsonl.py
  python training/build_combined_jsonl.py --voices kit dakota
  python training/build_combined_jsonl.py --voices kit dakota nora joe --base /workspace/training-data
"""
import argparse
import json
import os
import sys
from pathlib import Path

VOICES_DIR = Path(__file__).parent.parent / "voices"
DEFAULT_BASE = "/workspace/training-data"


def find_voices_with_data():
    """Return voice names that have a train_curated.jsonl."""
    voices = []
    for d in sorted(VOICES_DIR.iterdir()):
        if d.is_dir() and (d / "training-data" / "train_curated.jsonl").exists():
            voices.append(d.name)
    return voices


def main():
    ap = argparse.ArgumentParser(description="Build combined multi-voice JSONL for training")
    ap.add_argument("--voices", nargs="*", help="Voice names (default: auto-discover from voices/)")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"Base path on remote instance (default: {DEFAULT_BASE})")
    ap.add_argument("--output", default=None, help="Output path (default: training/train_multivoice.jsonl)")
    args = ap.parse_args()

    voices = args.voices or find_voices_with_data()
    if not voices:
        print("ERROR: no voices found with train_curated.jsonl", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or str(Path(__file__).parent / "train_multivoice.jsonl")
    count = 0
    per_voice = {}

    with open(out_path, "w") as out:
        for voice_name in voices:
            jsonl_path = VOICES_DIR / voice_name / "training-data" / "train_curated.jsonl"
            if not jsonl_path.exists():
                print(f"ERROR: missing {jsonl_path}", file=sys.stderr)
                sys.exit(1)

            remote_dir = f"{args.base}/{voice_name}"
            n = 0
            for line in open(jsonl_path):
                entry = json.loads(line)
                audio_rel = entry["audio"].lstrip("./")
                ref_rel = entry["ref_audio"].lstrip("./")
                rewritten = {
                    "audio": f"{remote_dir}/{audio_rel}",
                    "text": entry["text"],
                    "ref_audio": f"{remote_dir}/{ref_rel}",
                    "voice_name": voice_name,
                }
                out.write(json.dumps(rewritten) + "\n")
                count += 1
                n += 1
            per_voice[voice_name] = n
            print(f"  {voice_name}: {n} samples")

    print(f"\nWrote {count} combined samples → {out_path}")
    print(f"Per voice: {per_voice}")


if __name__ == "__main__":
    main()
