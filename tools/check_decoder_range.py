#!/usr/bin/env python3
"""Check if the codec decoder is producing values outside [-1, 1] range.

If it is, then the hard clip at the decoder output IS the source of the pops.
We need to use the Python mlx-audio to generate with a modified decoder
that returns unclipped values.

This script generates audio using the Python mlx-audio library (not Swift)
and measures the raw decoder output range before clipping.
"""
import sys
sys.path.insert(0, '/Users/nagy/Desktop/Files/AI/ivi/holler/.venv/lib/python3.13/site-packages')

import mlx.core as mx
import numpy as np

# Monkey-patch the speech tokenizer's decode to capture pre-clip values
PRE_CLIP_MAX = [0.0]
PRE_CLIP_VALUES = []

def patch_decoder(model):
    """Patch the speech tokenizer decoder to capture pre-clip output."""
    decoder = model.speech_tokenizer.decoder
    original_call = decoder.__class__.__call__

    def patched_call(self, codes):
        # Run the normal decode pipeline but capture before clip
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden)
        hidden = hidden.swapaxes(1, 2)
        hidden = self.pre_transformer(hidden)
        hidden = hidden.swapaxes(1, 2)

        for layer in self.upsample:
            hidden = layer(hidden)

        wav = hidden
        for layer in self.decoder:
            wav = layer(wav)

        # Capture raw values BEFORE clip
        raw = np.array(wav.astype(mx.float32))
        raw_abs = np.abs(raw)
        max_val = float(np.max(raw_abs))
        PRE_CLIP_MAX[0] = max(PRE_CLIP_MAX[0], max_val)

        above_1 = np.sum(raw_abs > 1.0)
        above_09 = np.sum(raw_abs > 0.9)
        total = raw.size
        PRE_CLIP_VALUES.append({
            'max': max_val,
            'above_1_pct': above_1 / total * 100,
            'above_09_pct': above_09 / total * 100,
            'samples': total,
        })

        print(f"  [decoder] max={max_val:.3f}, above_1.0={above_1}/{total} ({above_1/total*100:.2f}%), "
              f"above_0.9={above_09}/{total} ({above_09/total*100:.2f}%)")

        # Apply the normal clip
        return mx.clip(wav, -1, 1)

    decoder.__class__.__call__ = patched_call
    return decoder


def main():
    from mlx_audio.tts.utils import load as tts_load

    checkpoint = "/Users/nagy/Desktop/Files/AI/ivi/holler/checkpoints/holler-6voice-v12-6bit"
    print(f"Loading {checkpoint}...")
    model = tts_load(checkpoint)

    print("Patching decoder...")
    patch_decoder(model)

    test_texts = [
        "Oh wow, that actually worked! Can you believe it?",
        "The researchers found that implementing these changes resulted in a fifteen percent improvement across all metrics, which was honestly quite surprising given the initial skepticism from the team.",
        "They also mentioned that the new approach could potentially revolutionize how we think about scaling these systems in production environments.",
    ]

    for voice in ["nora", "kit", "dakota"]:
        print(f"\n{'='*60}")
        print(f"Voice: {voice}")
        print(f"{'='*60}")
        PRE_CLIP_MAX[0] = 0.0
        PRE_CLIP_VALUES.clear()

        for i, text in enumerate(test_texts):
            print(f"\nText {i+1}: {text[:60]}...")
            result = model.generate(
                text=text,
                voice=voice,
                language="english",
            )

        if PRE_CLIP_VALUES:
            max_all = max(v['max'] for v in PRE_CLIP_VALUES)
            avg_above_1 = np.mean([v['above_1_pct'] for v in PRE_CLIP_VALUES])
            avg_above_09 = np.mean([v['above_09_pct'] for v in PRE_CLIP_VALUES])
            print(f"\nSummary for {voice}:")
            print(f"  Global max: {max_all:.3f}")
            print(f"  Avg % above 1.0: {avg_above_1:.2f}%")
            print(f"  Avg % above 0.9: {avg_above_09:.2f}%")


if __name__ == '__main__':
    main()
