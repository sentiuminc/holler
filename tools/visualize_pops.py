#!/usr/bin/env python3
"""Visualize the actual waveform at confirmed pop locations.

Prints the raw sample values so we can see what the distortion looks like.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 24000
SRC = Path.home() / "Downloads/nora-g32-10para"

# A selection of confirmed pops with varying severity
POPS = [
    ("para_02.wav", 9.0, "crackle"),
    ("para_02.wav", 16.0, "crackle - 'they also mention'"),
    ("para_02.wav", 22.0, "crackle"),
    ("para_03.wav", 12.0, "pop"),
    ("para_03.wav", 15.0, "pop"),
    ("para_01.wav", 14.0, "slight pop"),
    ("para_01.wav", 18.0, "slight pop"),
    ("para_03.wav", 0.3, "giant clip at start"),
]

# Clean regions for comparison
CLEAN = [
    ("para_08.wav", 15.0, "clean speech"),
    ("para_08.wav", 25.9, "clean 'succeeds' - was false positive"),
]


def load(fname):
    audio, sr = sf.read(str(SRC / fname))
    assert sr == SR
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def analyze_distortion(audio, center_sample, window_ms=20):
    """Look at the waveform shape for signs of codec distortion."""
    half = int(SR * window_ms / 1000)
    start = max(0, center_sample - half)
    end = min(len(audio), center_sample + half)
    region = audio[start:end]

    # Look for "flat tops" — consecutive samples at similar high amplitude
    # This is the signature of waveform clipping/saturation
    flat_segments = []
    run_start = None
    for i in range(len(region) - 1):
        if abs(region[i]) > 0.3 and abs(region[i+1] - region[i]) < 0.005:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= 3:
                flat_segments.append((run_start, i, float(np.mean(region[run_start:i]))))
            run_start = None

    # Look for waveform symmetry — distorted signals have asymmetric half-cycles
    # Split by zero crossings, measure positive vs negative half-cycle amplitudes
    pos_peaks = []
    neg_peaks = []
    current_peak = region[0]
    for i in range(1, len(region)):
        if region[i] * region[i-1] < 0:  # zero crossing
            if current_peak > 0:
                pos_peaks.append(current_peak)
            else:
                neg_peaks.append(current_peak)
            current_peak = region[i]
        elif abs(region[i]) > abs(current_peak):
            current_peak = region[i]

    pos_mean = np.mean(pos_peaks) if pos_peaks else 0
    neg_mean = np.mean(neg_peaks) if neg_peaks else 0
    asymmetry = abs(pos_mean + neg_mean) / max(abs(pos_mean), abs(neg_mean), 0.001)

    # THD estimate: compare fundamental energy to harmonic energy
    fft = np.fft.rfft(region * np.hanning(len(region)))
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(region), 1/SR)

    # Find the fundamental (strongest frequency in 80-400Hz range)
    voice_mask = (freqs > 80) & (freqs < 400)
    if np.any(voice_mask):
        fund_idx = np.argmax(magnitude[voice_mask]) + np.where(voice_mask)[0][0]
        fund_freq = freqs[fund_idx]
        fund_mag = magnitude[fund_idx]

        # Sum harmonics 2-8
        harmonic_mag = 0
        for h in range(2, 9):
            h_freq = fund_freq * h
            h_idx = np.argmin(np.abs(freqs - h_freq))
            harmonic_mag += magnitude[h_idx] ** 2
        harmonic_mag = np.sqrt(harmonic_mag)

        thd = harmonic_mag / fund_mag if fund_mag > 0 else 0
    else:
        fund_freq = 0
        thd = 0

    return {
        'flat_segments': flat_segments,
        'asymmetry': asymmetry,
        'thd': thd,
        'fund_freq': fund_freq,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
    }


def print_waveform_snippet(audio, time_s, window_ms=5):
    """Print actual sample values around a timestamp."""
    center = int(time_s * SR)
    half = int(SR * window_ms / 1000)
    start = max(0, center - half)
    end = min(len(audio), center + half)
    region = audio[start:end]

    # Find the peak in this region
    peak_idx = np.argmax(np.abs(region))
    peak_val = region[peak_idx]

    # Print samples around the peak, showing waveform shape
    # Group by ~0.5ms (12 samples) for readability
    chunk_size = 12
    n_chunks = len(region) // chunk_size
    for c in range(min(n_chunks, 20)):
        chunk = region[c*chunk_size:(c+1)*chunk_size]
        time_offset = (start + c * chunk_size) / SR - time_s
        bar = ""
        for s in chunk:
            # Visual representation: each char = 0.1 amplitude
            pos = int(s * 10)
            pos = max(-7, min(7, pos))
            bar += "█" if pos > 0 else "░" if pos < 0 else "·"
        vals = " ".join(f"{s:+.2f}" for s in chunk[::3])  # every 3rd sample
        print(f"  {time_offset*1000:+5.1f}ms: [{bar}] {vals}")


def main():
    cache = {}
    def get_audio(fname):
        if fname not in cache:
            cache[fname] = load(fname)
        return cache[fname]

    for label, items in [("CONFIRMED POPS", POPS), ("CLEAN REGIONS", CLEAN)]:
        print(f"\n{'='*80}")
        print(label)
        print(f"{'='*80}")

        for fname, t, desc in items:
            audio = get_audio(fname)
            center = int(t * SR)

            print(f"\n--- {fname} at {t}s: {desc} ---")

            # Basic stats in 50ms window
            half = int(SR * 0.025)
            region = audio[max(0, center-half):min(len(audio), center+half)]
            peak = np.max(np.abs(region))
            rms = np.sqrt(np.mean(region ** 2))
            print(f"Peak={peak:.3f} RMS={rms:.3f} Crest={peak/rms:.1f}")

            # Distortion analysis
            d = analyze_distortion(audio, center)
            print(f"THD={d['thd']:.2f} Asymmetry={d['asymmetry']:.3f} F0≈{d['fund_freq']:.0f}Hz")
            if d['flat_segments']:
                print(f"Flat-top segments: {len(d['flat_segments'])} (clipping signature)")
            print(f"Pos/Neg half-cycle: +{d['pos_mean']:.3f} / {d['neg_mean']:.3f}")

            # Print waveform
            print(f"Waveform (5ms window, ±{t}s):")
            print_waveform_snippet(audio, t, window_ms=5)


if __name__ == '__main__':
    main()
