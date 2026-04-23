"""Proper benchmark — MLX metal memory, all three quants, same process."""
import time
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

from mlx_audio.tts import load
import mlx.core as mx

BASE = Path.home() / "Desktop/Files/AI/holler"

MODELS = {
    "bf16": str(BASE / "checkpoints/katie-v6"),
    "8bit": str(BASE / "checkpoints/katie-v6-8bit"),
    "4bit": str(BASE / "checkpoints/katie-v6-4bit"),
}

TEXTS = [
    "Oh I've never actually tried milk tea before, but maybe when I'm in Britain I will.",
    "Wait, so you're telling me the whole thing was just a misunderstanding? That's hilarious.",
    "I think the best part about working from home is just being able to cook lunch whenever.",
    "Hey do you know if there's a good coffee place near the park? I'm heading there now.",
    "Honestly I wasn't sure about it at first but now I kind of love it.",
    "So my friend just got back from Tokyo and she said the food was absolutely incredible.",
    "I'm not gonna lie, I spent way too long picking out a paint color for the bathroom.",
    "The weather's been so weird lately. Like yesterday it was sunny and today it's freezing.",
    "Can you remind me to call the dentist tomorrow? I keep forgetting.",
    "Yeah that makes sense. Let me think about it and I'll get back to you tonight.",
]

quant_filter = sys.argv[1] if len(sys.argv) > 1 else None

def mb(bytes):
    return bytes / 1024 / 1024

for label, path in MODELS.items():
    if quant_filter and label != quant_filter:
        continue

    # Reset MLX memory tracking
    mx.metal.reset_peak_memory()
    mx.clear_cache()
    mem_before = mx.metal.get_active_memory()

    print(f"\n{'='*70}")
    print(f"  {label.upper()} — {path.split('/')[-1]}")
    print(f"{'='*70}")

    t_load = time.time()
    model = load(path)
    mx.eval(model.parameters())  # force all weights into Metal memory
    load_ms = (time.time() - t_load) * 1000
    mem_loaded = mx.metal.get_active_memory()
    disk_mb = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1024 / 1024

    # Warm up
    for r in model.generate(text="Hello.", voice="katie", language="english",
                            temperature=0.6, stream=True, streaming_interval=0.1):
        pass
    mem_warm = mx.metal.get_active_memory()
    peak_warm = mx.metal.get_peak_memory()

    print(f"  Disk: {disk_mb:.0f}MB | Metal after load: {mb(mem_loaded - mem_before):.0f}MB | After warmup: {mb(mem_warm):.0f}MB | Peak: {mb(peak_warm):.0f}MB")
    print(f"  Load time: {load_ms:.0f}ms")
    print()
    print(f"  {'#':<4} {'TTFA':>6} {'Total':>7} {'Audio':>7} {'Peak':>6}  Text")
    print(f"  {'-'*80}")

    out_dir = BASE / f"samples/benchmark-katie-v6-{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mx.metal.reset_peak_memory()
    ttfas = []
    totals = []
    peaks = []

    for i, text in enumerate(TEXTS):
        t0 = time.time()
        chunks = []
        ttfa = None

        for result in model.generate(
            text=text, voice="katie", language="english",
            temperature=0.6, stream=True, streaming_interval=0.1,
        ):
            if hasattr(result, 'audio'):
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                chunks.append(result.audio)

        total_ms = (time.time() - t0) * 1000
        audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
        audio_len_ms = len(audio) / 24000 * 1000
        peak = np.abs(audio).max()

        ttfas.append(ttfa)
        totals.append(total_ms)
        peaks.append(peak)

        sf.write(str(out_dir / f"test_{i+1:02d}.wav"), audio, 24000)
        print(f"  {i+1:<4} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_len_ms:>6.0f}ms {peak:>5.3f}  {text[:50]}")

    inference_peak = mx.metal.get_peak_memory()
    print()
    print(f"  TTFA   — median: {np.median(ttfas):.0f}ms | mean: {np.mean(ttfas):.0f}ms | range: {min(ttfas):.0f}-{max(ttfas):.0f}ms")
    print(f"  Total  — median: {np.median(totals):.0f}ms | mean: {np.mean(totals):.0f}ms")
    print(f"  Peaks  — max: {max(peaks):.3f} | mean: {np.mean(peaks):.3f}")
    print(f"  Metal  — inference peak: {mb(inference_peak):.0f}MB")
    print(f"  Saved  — {out_dir}/")

    del model
    mx.clear_cache()
