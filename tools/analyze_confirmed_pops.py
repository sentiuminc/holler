#!/usr/bin/env python3
"""Analyze user-confirmed pop locations vs false positives.

Goal: find what ACTUALLY distinguishes a pop from normal loud speech.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 24000
SRC = Path.home() / "Downloads/nora-g32-10para"

# User-confirmed REAL pops/crackles (approximate timestamps)
REAL_POPS = [
    ("para_00.wav", 29.0),
    ("para_01.wav", 0.5),
    ("para_01.wav", 14.0),
    ("para_01.wav", 18.0),
    ("para_01.wav", 24.5),
    ("para_01.wav", 29.0),
    ("para_02.wav", 3.5),
    ("para_02.wav", 6.5),
    ("para_02.wav", 9.0),
    ("para_02.wav", 16.0),
    ("para_02.wav", 22.0),
    ("para_02.wav", 27.0),
    ("para_02.wav", 30.0),
    ("para_02.wav", 33.5),
    ("para_03.wav", 0.3),
    ("para_03.wav", 5.0),
    ("para_03.wav", 12.0),
    ("para_03.wav", 15.0),
    ("para_03.wav", 18.0),
    ("para_03.wav", 24.5),
]

# User-confirmed FALSE POSITIVES (I flagged these, user said no pop)
FALSE_POS = [
    ("para_08.wav", 25.9),  # "regular speech, just says succeeds"
    ("para_00.wav", 22.9),  # "fifteen percent faster topping, not crackle"
    ("para_04.wav", 27.1),  # "not crackle, just high notes on eSSes"
    ("para_09.wav", 8.2),   # "monthS, not pop"
    ("para_04.wav", 23.3),  # "no crackle, just riSk word"
    ("para_00.wav", 38.7),  # "no crackle, just goes from silence to speaking"
]


def load(fname):
    audio, sr = sf.read(str(SRC / fname))
    assert sr == SR
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def analyze_region(audio, time_s, window_ms=50):
    """Analyze a region around a timestamp. Returns dict of features."""
    center = int(time_s * SR)
    half = int(SR * window_ms / 1000)
    start = max(0, center - half)
    end = min(len(audio), center + half)
    region = audio[start:end]

    if len(region) < 10:
        return None

    # Basic stats
    peak = float(np.max(np.abs(region)))
    rms = float(np.sqrt(np.mean(region ** 2)))

    # Delta stats
    deltas = np.abs(np.diff(region.astype(np.float64)))
    max_delta = float(np.max(deltas))
    mean_delta = float(np.mean(deltas))
    p99_delta = float(np.percentile(deltas, 99))
    p95_delta = float(np.percentile(deltas, 95))

    # How many samples are near the peak (saturation measure)
    # If many samples are at 0.65+ and 0.70 peak, the waveform is "squared off"
    near_peak_ratio = float(np.mean(np.abs(region) > 0.6))

    # Clipping indicator: how many times does the waveform hit exactly 0.700 (our soft clip ceiling)?
    at_ceiling = float(np.mean(np.abs(region) > 0.695))

    # Flatness: if the waveform is clipped, consecutive samples will be nearly equal near the peak
    # Measure: how many consecutive samples have delta < 0.01 while abs > 0.5
    flat_at_peak = 0
    for i in range(len(region) - 1):
        if abs(region[i]) > 0.5 and abs(region[i+1] - region[i]) < 0.01:
            flat_at_peak += 1
    flat_at_peak_ratio = flat_at_peak / max(1, len(region))

    # Zero crossing rate (high = noise/distortion, normal speech = moderate)
    zc = 0
    for i in range(len(region) - 1):
        if region[i] * region[i+1] < 0:
            zc += 1
    zcr = zc / (len(region) / SR)  # crossings per second

    # Spectral centroid (higher = brighter/harsher)
    fft = np.fft.rfft(region * np.hanning(len(region)))
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(region), 1/SR)
    centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0

    # THD proxy: ratio of energy above 4kHz to total energy
    # High-frequency distortion produces excess energy above natural speech band
    freq_mask_4k = freqs > 4000
    hf_energy = float(np.sum(magnitude[freq_mask_4k] ** 2))
    total_energy = float(np.sum(magnitude ** 2))
    hf_ratio = hf_energy / total_energy if total_energy > 0 else 0

    # Crest factor (peak/rms) — lower = more compressed/clipped
    crest = peak / rms if rms > 0 else 0

    # Count rapid sign changes in deltas (oscillation indicator)
    delta_signs = np.sign(np.diff(region))
    sign_changes = np.sum(np.abs(np.diff(delta_signs)) > 0)
    sign_change_rate = sign_changes / len(region) * SR

    return {
        'peak': peak,
        'rms': rms,
        'crest': crest,
        'max_delta': max_delta,
        'mean_delta': mean_delta,
        'p95_delta': p95_delta,
        'p99_delta': p99_delta,
        'near_peak_ratio': near_peak_ratio,
        'at_ceiling': at_ceiling,
        'flat_at_peak': flat_at_peak_ratio,
        'zcr': zcr,
        'centroid': centroid,
        'hf_ratio': hf_ratio,
        'sign_change_rate': sign_change_rate,
    }


def main():
    # Cache loaded audio
    cache = {}
    def get_audio(fname):
        if fname not in cache:
            cache[fname] = load(fname)
        return cache[fname]

    print("="*100)
    print("REAL POPS (user-confirmed)")
    print("="*100)
    print(f"{'File':<12} {'Time':>5} {'Peak':>5} {'RMS':>5} {'Crest':>5} {'MaxΔ':>6} {'p99Δ':>6} "
          f"{'NearPk':>7} {'AtCeil':>7} {'Flat':>5} {'HF%':>5} {'Centrd':>6}")
    print("-"*100)

    real_features = []
    for fname, t in REAL_POPS:
        audio = get_audio(fname)
        f = analyze_region(audio, t)
        if f is None:
            continue
        real_features.append(f)
        print(f"{fname:<12} {t:>5.1f} {f['peak']:>5.3f} {f['rms']:>5.3f} {f['crest']:>5.1f} "
              f"{f['max_delta']:>6.3f} {f['p99_delta']:>6.3f} {f['near_peak_ratio']:>7.3f} "
              f"{f['at_ceiling']:>7.3f} {f['flat_at_peak']:>5.3f} {f['hf_ratio']:>5.3f} {f['centroid']:>6.0f}")

    print(f"\n{'='*100}")
    print("FALSE POSITIVES (I flagged, user said no pop)")
    print("="*100)
    print(f"{'File':<12} {'Time':>5} {'Peak':>5} {'RMS':>5} {'Crest':>5} {'MaxΔ':>6} {'p99Δ':>6} "
          f"{'NearPk':>7} {'AtCeil':>7} {'Flat':>5} {'HF%':>5} {'Centrd':>6}")
    print("-"*100)

    false_features = []
    for fname, t in FALSE_POS:
        audio = get_audio(fname)
        f = analyze_region(audio, t)
        if f is None:
            continue
        false_features.append(f)
        print(f"{fname:<12} {t:>5.1f} {f['peak']:>5.3f} {f['rms']:>5.3f} {f['crest']:>5.1f} "
              f"{f['max_delta']:>6.3f} {f['p99_delta']:>6.3f} {f['near_peak_ratio']:>7.3f} "
              f"{f['at_ceiling']:>7.3f} {f['flat_at_peak']:>5.3f} {f['hf_ratio']:>5.3f} {f['centroid']:>6.0f}")

    # Compare averages
    print(f"\n{'='*100}")
    print("COMPARISON (average across all samples)")
    print("="*100)

    metrics = ['peak', 'rms', 'crest', 'max_delta', 'p99_delta', 'near_peak_ratio',
               'at_ceiling', 'flat_at_peak', 'hf_ratio', 'centroid', 'zcr', 'sign_change_rate']

    print(f"\n{'Metric':<20} {'Real Pops':>12} {'False Pos':>12} {'Ratio':>8} {'Discriminates?':>15}")
    print("-"*70)
    for m in metrics:
        real_avg = np.mean([f[m] for f in real_features])
        false_avg = np.mean([f[m] for f in false_features])
        ratio = real_avg / false_avg if false_avg > 0 else float('inf')
        # Flag metrics with >1.5x or <0.67x ratio as discriminating
        disc = ""
        if ratio > 1.5:
            disc = f"REAL {ratio:.1f}x HIGHER"
        elif ratio < 0.67:
            disc = f"REAL {1/ratio:.1f}x LOWER"
        print(f"{m:<20} {real_avg:>12.4f} {false_avg:>12.4f} {ratio:>8.2f} {disc:>15}")

    # Also look at a CLEAN region for baseline (middle of para_08, user said it was clean)
    print(f"\n{'='*100}")
    print("CLEAN BASELINE (para_08 at 15s, known clean region)")
    print("="*100)
    audio = get_audio("para_08.wav")
    f = analyze_region(audio, 15.0)
    print(f"Peak={f['peak']:.3f} RMS={f['rms']:.3f} Crest={f['crest']:.1f} MaxΔ={f['max_delta']:.3f} "
          f"p99Δ={f['p99_delta']:.3f} NearPk={f['near_peak_ratio']:.3f} AtCeil={f['at_ceiling']:.3f} "
          f"HF%={f['hf_ratio']:.3f} Centroid={f['centroid']:.0f}")


if __name__ == '__main__':
    main()
