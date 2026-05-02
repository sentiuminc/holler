#!/usr/bin/env python3
"""Generate test clips from a checkpoint to verify voice quality.

Usage:
  python inference/test_checkpoint.py checkpoints/holler-kit-dakota-6bit kit dakota
  python inference/test_checkpoint.py path/to/checkpoint voice1 voice2 --output ~/Downloads/samples
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load

TEST_TEXTS = [
    "What time is it in Tokyo right now?",
    "The quick brown fox jumps over the lazy dog.",
    "Before we begin, let me make sure I have this right.",
    "I think we should grab dinner somewhere downtown.",
    "There's a package waiting for you at the front desk.",
    "Would you prefer coffee or tea with your breakfast?",
    "The train arrives at the station in eight minutes.",
    "Don't worry, I'll handle everything from here.",
    "Sorry to interrupt, but I have an urgent question.",
    "It's been a long day. Let's call it a night.",
]

def main():
    ap = argparse.ArgumentParser(description="Test a checkpoint by generating clips per voice")
    ap.add_argument("checkpoint", help="Path to checkpoint directory")
    ap.add_argument("voices", nargs="+", help="Voice names to test")
    ap.add_argument("--output", default=None, help="Output directory (default: <checkpoint>/samples)")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else Path(args.checkpoint) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    t0 = time.time()
    model = load(args.checkpoint)
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    for voice in args.voices:
        voice_dir = out_dir / voice
        voice_dir.mkdir(exist_ok=True)
        print(f"=== {voice} ===")
        for i, text in enumerate(TEST_TEXTS, 1):
            t = time.time()
            chunks = []
            for result in model.generate(
                text=text, voice=voice, language="english", temperature=0.6, stream=False
            ):
                chunks.append(np.asarray(result.audio).squeeze())
            audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            dur = len(audio) / 24000
            peak = np.abs(audio).max()
            clip_flag = " CLIP" if peak >= 0.999 else ""
            sf.write(str(voice_dir / f"test_{i:02d}.wav"), audio, 24000)
            print(f"  {i:02d}: {dur:.1f}s  peak={peak:.3f}  gen={time.time()-t:.1f}s{clip_flag}")
        print()

    print(f"Done. Samples in {out_dir}")

if __name__ == "__main__":
    main()
