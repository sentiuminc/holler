#!/usr/bin/env python3
"""Quick GPU inference test for multi-voice checkpoints.
Uses the high-level qwen_tts API with speaker parameter.

Usage:
  python3 test_multivoice_gpu.py --checkpoint /workspace/output/checkpoint-epoch-1 --output /workspace/samples/
"""
import argparse, os, json
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

TEST_TEXTS = [
    "Hey, I just finished looking through everything, and honestly, it's not as bad as I thought. There are a few things we should probably fix, but nothing that can't wait until tomorrow.",
    "So here's the deal. The report came back, and the numbers look solid. I'd say we're in a pretty good spot overall, wouldn't you agree?",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./samples")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        config = json.load(f)

    spk_ids = config.get("talker_config", {}).get("spk_id", {})
    voices = {name: slot for name, slot in spk_ids.items() if 3000 <= slot <= 3071}
    print(f"Custom voices: {voices}")

    if not voices:
        print("No custom voices found in config!")
        return

    model = Qwen3TTSModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.model.to("cuda")

    for voice_name, slot in sorted(voices.items()):
        for i, text in enumerate(TEST_TEXTS):
            print(f"  {voice_name} (slot {slot}) text {i}...", end=" ", flush=True)
            try:
                inputs = model.processor(
                    text=text, speaker=voice_name, return_tensors="pt"
                )
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

                with torch.no_grad():
                    result = model.model.generate(
                        **inputs,
                        temperature=args.temperature,
                        max_new_tokens=2048,
                    )

                if isinstance(result, tuple):
                    audio = result[0]
                elif isinstance(result, list):
                    audio = result[0]
                else:
                    audio = result

                audio_np = audio.cpu().float().numpy().squeeze()
                fname = f"{voice_name}_text{i}.wav"
                sf.write(os.path.join(args.output, fname), audio_np, 24000)
                peak = abs(audio_np).max()
                dur = len(audio_np) / 24000
                print(f"{dur:.1f}s, peak={peak:.3f}")

                if dur > 30:
                    print(f"    ⚠ EOS failure — {dur:.1f}s is way too long")
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"\nSamples saved to {args.output}/")
    print(f"Files: {sorted(os.listdir(args.output))}")

if __name__ == "__main__":
    main()
