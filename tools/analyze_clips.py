#!/usr/bin/env python3
"""Spectral analysis of training clips. Samples N clips evenly across the set.

Analyzes what matters for TTS training data quality:
- Noise floor (dB) — background noise level, lower is better
- SNR (dB) — signal-to-noise ratio
- Spectral centroid (Hz) — brightness/darkness of voice
- Sibilance energy (4-10kHz relative to total) — de-essing concern
- Harshness zone (2-4kHz relative to total) — ear fatigue zone
- Peak level (dBFS) — headroom, clipping detection
- RMS level (dBFS) — overall loudness
- Silence ratio — fraction of clip that's silence (< -40dB)
- Duration
- DC offset — can cause clicks between clips
- Spectral rolloff (Hz) — where 85% of energy lives

Usage:
  python analyze_clips.py --voice joe
  python analyze_clips.py --voice joe --n 50
  python analyze_clips.py --voice katie --source audio-original  # compare pre-enhancement
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import json

VOICES_DIR = Path(__file__).parent.parent / "voices"
SR = 24000


def analyze_clip(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    duration = len(audio) / sr
    n = len(audio)

    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    dc_offset = np.abs(np.mean(audio))

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = [audio[i:i+frame_len] for i in range(0, n - frame_len, hop)]
    frame_rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    frame_db = 20 * np.log10(frame_rms + 1e-10)

    silence_threshold = -40
    silence_ratio = np.mean(frame_db < silence_threshold)

    noise_frames = frame_db[frame_db < silence_threshold]
    noise_floor_db = float(np.median(noise_frames)) if len(noise_frames) > 5 else -80.0

    signal_frames = frame_db[frame_db >= silence_threshold]
    if len(signal_frames) > 0:
        snr = float(np.median(signal_frames)) - noise_floor_db
    else:
        snr = 0.0

    freqs, psd = signal.welch(audio, sr, nperseg=2048)
    total_power = np.sum(psd)

    centroid_idx = np.sum(freqs * psd) / (total_power + 1e-10)

    harsh_mask = (freqs >= 2000) & (freqs <= 4000)
    harsh_ratio = np.sum(psd[harsh_mask]) / (total_power + 1e-10)

    sibilance_mask = (freqs >= 4000) & (freqs <= 10000)
    sibilance_ratio = np.sum(psd[sibilance_mask]) / (total_power + 1e-10)

    cumulative = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * total_power)
    rolloff_hz = freqs[min(rolloff_idx, len(freqs) - 1)]

    return {
        "file": path.name,
        "duration_s": round(duration, 2),
        "peak_dbfs": round(peak_db, 1),
        "rms_dbfs": round(rms_db, 1),
        "dc_offset": round(dc_offset, 5),
        "noise_floor_db": round(noise_floor_db, 1),
        "snr_db": round(snr, 1),
        "silence_ratio": round(silence_ratio, 3),
        "spectral_centroid_hz": round(float(centroid_idx), 0),
        "harsh_2_4k_ratio": round(float(harsh_ratio), 4),
        "sibilance_4_10k_ratio": round(float(sibilance_ratio), 4),
        "spectral_rolloff_hz": round(float(rolloff_hz), 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--source", default="audio", help="Subdirectory under training-data/")
    args = parser.parse_args()

    audio_dir = VOICES_DIR / args.voice / "training-data" / args.source
    if not audio_dir.exists():
        print(f"Error: {audio_dir} not found")
        return

    wavs = sorted(f for f in audio_dir.iterdir() if f.suffix == '.wav')
    print(f"Found {len(wavs)} clips in {audio_dir}")

    if args.n < len(wavs):
        indices = np.linspace(0, len(wavs) - 1, args.n, dtype=int)
        sampled = [wavs[i] for i in indices]
    else:
        sampled = wavs

    print(f"Analyzing {len(sampled)} clips...\n")

    results = []
    for i, path in enumerate(sampled):
        r = analyze_clip(path)
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sampled)}] done")

    keys = ["duration_s", "peak_dbfs", "rms_dbfs", "dc_offset", "noise_floor_db",
            "snr_db", "silence_ratio", "spectral_centroid_hz", "harsh_2_4k_ratio",
            "sibilance_4_10k_ratio", "spectral_rolloff_hz"]

    print(f"\n{'='*70}")
    print(f"SPECTRAL ANALYSIS — {args.voice} ({args.source})")
    print(f"{'='*70}")
    print(f"Clips analyzed: {len(sampled)} / {len(wavs)}")
    print()

    print(f"{'Metric':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
    print("-" * 75)
    for k in keys:
        vals = [r[k] for r in results]
        print(f"{k:<25} {np.mean(vals):>10.2f} {np.median(vals):>10.2f} {np.min(vals):>10.2f} {np.max(vals):>10.2f} {np.std(vals):>10.2f}")

    print()

    issues = []
    hot_clips = [r for r in results if r["peak_dbfs"] > -0.5]
    if hot_clips:
        issues.append(f"  {len(hot_clips)} clips with peak > -0.5 dBFS (near clipping)")

    dc_clips = [r for r in results if r["dc_offset"] > 0.005]
    if dc_clips:
        issues.append(f"  {len(dc_clips)} clips with DC offset > 0.005")

    noisy = [r for r in results if r["snr_db"] < 20]
    if noisy:
        issues.append(f"  {len(noisy)} clips with SNR < 20 dB (noisy)")

    harsh = [r for r in results if r["harsh_2_4k_ratio"] > 0.15]
    if harsh:
        issues.append(f"  {len(harsh)} clips with high 2-4kHz harshness (>{15}%)")

    sibilant = [r for r in results if r["sibilance_4_10k_ratio"] > 0.10]
    if sibilant:
        issues.append(f"  {len(sibilant)} clips with high sibilance (>{10}%)")

    short = [r for r in results if r["duration_s"] < 0.5]
    if short:
        issues.append(f"  {len(short)} clips under 0.5s (may be too short)")

    silent = [r for r in results if r["silence_ratio"] > 0.5]
    if silent:
        issues.append(f"  {len(silent)} clips with >50% silence")

    if issues:
        print("POTENTIAL ISSUES:")
        for issue in issues:
            print(issue)
    else:
        print("No major issues detected.")

    print()

    worst_snr = sorted(results, key=lambda r: r["snr_db"])[:5]
    print("LOWEST SNR (noisiest):")
    for r in worst_snr:
        print(f"  {r['file']}: SNR {r['snr_db']} dB, noise floor {r['noise_floor_db']} dB")

    most_harsh = sorted(results, key=lambda r: r["harsh_2_4k_ratio"], reverse=True)[:5]
    print("\nHIGHEST HARSHNESS (2-4kHz):")
    for r in most_harsh:
        print(f"  {r['file']}: {r['harsh_2_4k_ratio']*100:.1f}% of energy in 2-4kHz")

    most_sibilant = sorted(results, key=lambda r: r["sibilance_4_10k_ratio"], reverse=True)[:5]
    print("\nHIGHEST SIBILANCE (4-10kHz):")
    for r in most_sibilant:
        print(f"  {r['file']}: {r['sibilance_4_10k_ratio']*100:.1f}% of energy in 4-10kHz")


if __name__ == "__main__":
    main()
