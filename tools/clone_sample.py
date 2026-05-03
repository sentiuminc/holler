"""Clone a voice from a reference audio clip.

Takes one audio file as reference, generates N outputs saying specified text.
For quick testing of reference clips before committing to full training data generation.

Usage:
  python clone_sample.py --ref audio.wav --ref-text "transcript of audio.wav" --text "what to say" --count 5
  python clone_sample.py --ref audio.wav --ref-text "transcript" --text "what to say" --output ~/Downloads/clones
  python clone_sample.py --ref audio.wav --ref-text "transcript" --text "what to say" --count 10 --temperature 0.9
"""
import argparse
import time
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_TEXT = "Okay, so I looked into it and here's what I found. The file you were working on got saved to your Downloads folder, not your Desktop. Want me to move it over, or would you rather keep it where it is?"


def main():
    parser = argparse.ArgumentParser(description="Clone a voice from a reference clip")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--ref-text", required=True, help="Exact transcript of the reference audio")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text for the clone to say (default: assistant sample)")
    parser.add_argument("--count", type=int, default=5, help="Number of clones to generate (default: 5)")
    parser.add_argument("--output", default=None, help="Output directory (default: ~/Downloads/clone-samples/)")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature (default: 0.85)")
    args = parser.parse_args()

    ref_path = Path(args.ref).resolve()
    if not ref_path.exists():
        sys.exit(f"ERROR: Reference file not found: {ref_path}")

    out_dir = Path(args.output) if args.output else Path.home() / "Downloads" / "clone-samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_name = ref_path.stem

    print(f"Reference: {ref_path}")
    print(f"Ref text:  {args.ref_text[:80]}...")
    print(f"Output:    {args.text[:80]}...")
    print(f"Count:     {args.count}")
    print(f"Temp:      {args.temperature}")
    print(f"Output to: {out_dir}")
    print()

    print("Loading Qwen3-TTS 1.7B Base bf16...")
    t0 = time.time()
    from mlx_audio.tts import load
    model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    for i in range(args.count):
        print(f"[{i+1}/{args.count}] Generating...")
        t0 = time.time()
        chunks = []

        try:
            for result in model.generate(
                text=args.text,
                ref_audio=str(ref_path),
                ref_text=args.ref_text,
                language="english",
                temperature=args.temperature,
            ):
                audio = np.array(result.audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()
                chunks.append(audio)

            full_audio = np.concatenate(chunks)
            duration = len(full_audio) / 24000
            elapsed_ms = (time.time() - t0) * 1000

            out_path = out_dir / f"{ref_name}_clone{i+1:02d}.wav"
            sf.write(str(out_path), full_audio, 24000)
            print(f"  ✓ {out_path.name} ({duration:.1f}s, {elapsed_ms:.0f}ms)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDone! {args.count} clones in {out_dir}")


if __name__ == "__main__":
    main()
