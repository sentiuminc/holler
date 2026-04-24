"""Quantize Katie v6 bf16 into 7 variants and benchmark each.

Phase 1: Quantize all variants from bf16 source
Phase 2: Benchmark each — disk size, Metal RAM, TTFA, RTF, save audio samples

Usage:
  python3 inference/experiment_quant_methods.py quantize   # Phase 1 only
  python3 inference/experiment_quant_methods.py benchmark  # Phase 2 only
  python3 inference/experiment_quant_methods.py            # Both phases
"""
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx

BASE = Path.home() / "Desktop/Files/AI/holler"
BF16_PATH = BASE / "checkpoints/katie-v6"
QUANT_DIR = BASE / "checkpoints/quant-experiment"
SAMPLES_DIR = BASE / "samples/quant-experiment"

VARIANTS = [
    # (name, q_bits, q_group_size, q_mode)
    ("affine-4bit-g64",  4,  64,  "affine"),   # current baseline
    ("affine-4bit-g32",  4,  32,  "affine"),   # finer groups
    ("affine-4bit-g128", 4,  128, "affine"),   # coarser groups
    ("affine-3bit-g64",  3,  64,  "affine"),   # push smaller
    ("affine-6bit-g64",  6,  64,  "affine"),   # quality ceiling
    ("mxfp4",            4,  32,  "mxfp4"),    # microscaling float
    ("nvfp4",            4,  16,  "nvfp4"),    # nvidia float, finest groups
    ("mxfp8",            8,  32,  "mxfp8"),    # higher precision float
]

TEXTS = [
    "Oh I've never actually tried milk tea before, but maybe when I'm in Britain I will.",
    "Wait, so you're telling me the whole thing was just a misunderstanding? That's hilarious.",
    "I think the best part about working from home is just being able to cook lunch whenever.",
    "Hey do you know if there's a good coffee place near the park? I'm heading there now.",
    "Honestly I wasn't sure about it at first but now I kind of love it.",
    "Okay.",
]

def mb(b):
    return b / 1024 / 1024

def disk_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 / 1024

def safetensors_mb(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1024 / 1024


def phase_quantize(only=None):
    from mlx_audio.convert import convert

    QUANT_DIR.mkdir(parents=True, exist_ok=True)

    for name, q_bits, q_group_size, q_mode in VARIANTS:
        if only and name != only:
            continue

        out_path = QUANT_DIR / name
        if out_path.exists():
            print(f"[skip] {name} already exists at {out_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Quantizing: {name} (bits={q_bits}, group={q_group_size}, mode={q_mode})")
        print(f"{'='*60}")

        t0 = time.time()
        convert(
            hf_path=str(BF16_PATH),
            mlx_path=str(out_path),
            quantize=True,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
        )
        elapsed = time.time() - t0
        sz = safetensors_mb(out_path)
        print(f"  Done in {elapsed:.1f}s — {sz:.0f}MB on disk")

    print(f"\nAll variants in {QUANT_DIR}/")


def phase_benchmark(only=None):
    from mlx_audio.tts import load

    results = []

    for name, q_bits, q_group_size, q_mode in VARIANTS:
        if only and name != only:
            continue

        model_path = QUANT_DIR / name
        if not model_path.exists():
            print(f"[skip] {name} — not found at {model_path}")
            continue

        mx.metal.reset_peak_memory()
        mx.clear_cache()
        mem_before = mx.metal.get_active_memory()

        print(f"\n{'='*70}")
        print(f"  {name} (bits={q_bits}, group={q_group_size}, mode={q_mode})")
        print(f"{'='*70}")

        t_load = time.time()
        model = load(str(model_path))
        mx.eval(model.parameters())
        load_ms = (time.time() - t_load) * 1000
        mem_loaded = mx.metal.get_active_memory() - mem_before
        disk = safetensors_mb(model_path)

        # Warmup
        for r in model.generate(text="Hello.", voice="katie", language="english",
                                temperature=0.6, stream=True, streaming_interval=0.1):
            pass
        mx.metal.reset_peak_memory()

        print(f"  Disk: {disk:.0f}MB | Metal RAM: {mb(mem_loaded):.0f}MB | Load: {load_ms:.0f}ms")
        print()
        print(f"  {'#':<4} {'TTFA':>6} {'Total':>7} {'Audio':>7} {'RTF':>6} {'Peak':>6}  Text")
        print(f"  {'-'*85}")

        out_dir = SAMPLES_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        ttfas = []
        totals = []
        rtfs = []
        peaks_audio = []

        for i, text in enumerate(TEXTS):
            t0 = time.time()
            chunks = []
            ttfa = None

            for result in model.generate(
                text=text, voice="katie", language="english",
                temperature=0.6, stream=True, streaming_interval=0.1,
            ):
                if hasattr(result, "audio"):
                    if ttfa is None:
                        ttfa = (time.time() - t0) * 1000
                    chunks.append(result.audio)

            total_ms = (time.time() - t0) * 1000

            if not chunks:
                print(f"  {i+1:<4} {'FAIL':>6} — no audio generated")
                continue

            audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
            audio_ms = len(audio) / 24000 * 1000
            peak = float(np.abs(audio).max())
            rtf = (total_ms / audio_ms) if audio_ms > 0 else 999

            ttfas.append(ttfa or 0)
            totals.append(total_ms)
            rtfs.append(rtf)
            peaks_audio.append(peak)

            sf.write(str(out_dir / f"test_{i+1:02d}.wav"), audio, 24000)
            print(f"  {i+1:<4} {(ttfa or 0):>5.0f}ms {total_ms:>6.0f}ms {audio_ms:>6.0f}ms {rtf:>5.2f}x {peak:>5.3f}  {text[:45]}")

        inference_peak = mx.metal.get_peak_memory()

        if ttfas:
            print()
            print(f"  TTFA  — median: {np.median(ttfas):.0f}ms | range: {min(ttfas):.0f}-{max(ttfas):.0f}ms")
            print(f"  Total — median: {np.median(totals):.0f}ms")
            print(f"  RTF   — mean: {np.mean(rtfs):.2f}x | range: {min(rtfs):.2f}-{max(rtfs):.2f}x")
            print(f"  Metal — peak: {mb(inference_peak):.0f}MB")
            print(f"  Audio — {out_dir}/")

        results.append({
            "name": name,
            "bits": q_bits,
            "group": q_group_size,
            "mode": q_mode,
            "disk_mb": disk,
            "ram_mb": mb(mem_loaded),
            "load_ms": load_ms,
            "ttfa_median": np.median(ttfas) if ttfas else None,
            "rtf_mean": np.mean(rtfs) if rtfs else None,
            "peak_mb": mb(inference_peak),
        })

        del model
        mx.clear_cache()

    # Summary table
    if results:
        print(f"\n\n{'='*90}")
        print(f"  SUMMARY")
        print(f"{'='*90}")
        print(f"  {'Name':<20} {'Disk':>6} {'RAM':>6} {'TTFA':>7} {'RTF':>6} {'Peak':>6}")
        print(f"  {'-'*60}")
        for r in results:
            ttfa_str = f"{r['ttfa_median']:.0f}ms" if r['ttfa_median'] else "N/A"
            rtf_str = f"{r['rtf_mean']:.2f}x" if r['rtf_mean'] else "N/A"
            print(f"  {r['name']:<20} {r['disk_mb']:>5.0f}M {r['ram_mb']:>5.0f}M {ttfa_str:>7} {rtf_str:>6} {r['peak_mb']:>5.0f}M")
        print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    only = sys.argv[2] if len(sys.argv) > 2 else None

    if mode in ("quantize", "both"):
        phase_quantize(only)
    if mode in ("benchmark", "both"):
        phase_benchmark(only)
