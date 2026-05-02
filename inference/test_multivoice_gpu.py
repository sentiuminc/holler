"""
Multi-voice GPU inference test. Generates samples for both voices from a checkpoint.

Usage:
  python3 test_multivoice_gpu.py --checkpoint /workspace/output/checkpoint-epoch-0
  python3 test_multivoice_gpu.py --checkpoint /workspace/output/checkpoint-epoch-1
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

TEST_TEXTS = [
    "Hey, what's going on?",
    "I checked the settings and everything looks fine to me.",
    "That's a really interesting way to think about it, actually.",
    "Okay, give me just a second.",
    "The weather today is partly cloudy with a high of seventy-two degrees.",
    "I'm not sure I agree with that, but let me think about it.",
    "Done! Anything else you need?",
    "So basically, what happened was the server went down around three in the morning and nobody noticed until the alerts fired at six.",
    "Would you like me to read that back to you?",
    "Happy birthday! I hope you have an amazing day.",
]

VOICES = ["nora", "joe"]


def load_config(checkpoint_path):
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        return json.load(f)


def run_standard_inference(checkpoint_path, output_dir):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    print("\n=== Standard inference (baseline) ===")
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    config = load_config(checkpoint_path)
    voice_slots = config.get("talker_config", {}).get("spk_id", {})
    print(f"Voice slots: {voice_slots}")

    for voice in VOICES:
        if voice not in voice_slots:
            print(f"WARNING: {voice} not in checkpoint, skipping")
            continue

        voice_dir = os.path.join(output_dir, f"standard-{voice}")
        os.makedirs(voice_dir, exist_ok=True)

        for i, text in enumerate(TEST_TEXTS):
            t0 = time.time()
            wav = model.generate_custom_voice(
                text=text,
                speaker=voice,
                temperature=0.6,
            )
            elapsed = time.time() - t0
            audio = wav.cpu().numpy().squeeze()
            duration = len(audio) / 24000
            rtf = duration / elapsed if elapsed > 0 else 0
            peak = np.abs(audio).max()
            out_path = os.path.join(voice_dir, f"test_{i:02d}.wav")
            sf.write(out_path, audio, 24000)
            print(f"  [{voice}] {i:02d}: {elapsed:.1f}s gen, {duration:.1f}s audio, RTF {rtf:.2f}, peak {peak:.3f} | {text[:50]}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/workspace/samples")
    args = parser.parse_args()

    epoch = args.checkpoint.rstrip("/").split("-")[-1]
    output_dir = os.path.join(args.output, f"epoch-{epoch}")
    os.makedirs(output_dir, exist_ok=True)

    run_standard_inference(args.checkpoint, output_dir)

    print(f"\nAll samples saved to {output_dir}")


if __name__ == "__main__":
    main()
