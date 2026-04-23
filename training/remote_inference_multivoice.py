"""Runs on Vast instance. Generates N test clips per voice from a multi-voice checkpoint.
Outputs to /workspace/inference_samples/<voice>/test_NN.wav and writes a manifest
/workspace/inference_samples/manifest.jsonl with (voice, text, audio_path).

Key diagnostic: any clip that hits max_new_tokens means EOS is broken → retrain.
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", default="/workspace/inference_samples")
    ap.add_argument("--voices", nargs="+", default=["katie", "joe"])
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Supported speakers: {model.get_supported_speakers()}")

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    n_clips_total = 0
    n_eos_broken = 0
    with open(manifest_path, "w") as manifest:
        for voice in args.voices:
            vdir = os.path.join(args.output_dir, voice)
            os.makedirs(vdir, exist_ok=True)
            for i, text in enumerate(TEST_TEXTS, 1):
                label = f"test_{i:02d}"
                print(f"  [{voice}] {label}: {text[:60]}")
                try:
                    wavs, sr = model.generate_custom_voice(
                        text=text,
                        language="English",
                        speaker=voice,
                        max_new_tokens=args.max_new_tokens,
                    )
                    audio = wavs[0]
                    out_path = os.path.join(vdir, f"{label}.wav")
                    sf.write(out_path, audio, sr)
                    duration = len(audio) / sr
                    # Heuristic: if duration is suspiciously long (>15s for a short sentence), EOS may be broken
                    words = len(text.split())
                    expected = words * 0.3  # ~180wpm nominal
                    eos_broken = duration > (expected * 3 + 5)  # 3x expected + 5s buffer
                    if eos_broken:
                        n_eos_broken += 1
                        print(f"    ⚠ suspect EOS broken: {duration:.1f}s for {words} words")
                    manifest.write(json.dumps({
                        "voice": voice, "label": label, "text": text,
                        "path": out_path, "duration_s": duration,
                        "suspect_eos_broken": eos_broken,
                    }) + "\n")
                    n_clips_total += 1
                except Exception as e:
                    print(f"    ❌ failed: {e}")
                    manifest.write(json.dumps({
                        "voice": voice, "label": label, "text": text,
                        "error": str(e),
                    }) + "\n")

    print(f"\nDone. {n_clips_total} clips generated, {n_eos_broken} suspect EOS broken.")
    print(f"Manifest: {manifest_path}")
    # Non-zero exit if >50% of clips broken
    if n_clips_total > 0 and n_eos_broken / n_clips_total > 0.5:
        print("ERROR: majority of clips look broken")
        sys.exit(2)


if __name__ == "__main__":
    main()
