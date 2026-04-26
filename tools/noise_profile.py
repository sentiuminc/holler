#!/usr/bin/env python3
"""Noise profiler: shows WHERE noise lives in frequency, before and after enhancement.

Compares noise spectral profiles between two directories of WAV files.
Extracts noise from silent frames AND estimates noise during speech using
harmonic/noise decomposition.

Usage:
  python noise_profile.py --before voices/joe/training-data/audio-original \
                          --after voices/joe/training-data/audio \
                          --n 30
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path


def extract_noise_profile(audio, sr, silence_threshold_db=-40):
    """Extract noise spectral profile from silent frames."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 2048

    # Find silent frames
    frames = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i+frame_len]
        rms_db = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-10)
        if rms_db < silence_threshold_db:
            frames.append(frame)

    if len(frames) < 3:
        return None, None

    # STFT of silent frames → noise spectrum
    spectra = []
    for frame in frames:
        padded = np.zeros(n_fft)
        padded[:len(frame)] = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(padded))
        spectra.append(spectrum)

    mean_spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    return freqs, 20 * np.log10(mean_spectrum + 1e-10)


def extract_speech_noise(audio, sr):
    """Estimate noise during speech using spectral subtraction approach.

    Computes the "smoothness" of the spectrum in voiced frames.
    Noise manifests as elevated floor between harmonics.
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 2048

    noise_floors = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i+frame_len]
        rms_db = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-10)
        if rms_db < -40:  # skip silence
            continue

        padded = np.zeros(n_fft)
        padded[:len(frame)] = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(padded))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)

        # Noise floor estimate: 10th percentile of spectrum in voiced region
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        voiced_mask = (freqs >= 100) & (freqs <= 4000)
        if np.sum(voiced_mask) > 0:
            noise_floors.append(np.percentile(spectrum_db[voiced_mask], 10))

    return np.mean(noise_floors) if noise_floors else -80.0


def analyze_dir(source, n=None):
    wavs = sorted(f for f in source.iterdir() if f.suffix == '.wav')
    if n and n < len(wavs):
        indices = np.linspace(0, len(wavs) - 1, n, dtype=int)
        wavs = [wavs[i] for i in indices]

    all_silence_profiles = []
    speech_noise_floors = []
    band_noise = {'80-300': [], '300-1k': [], '1k-3k': [], '3k-6k': [], '6k-10k': [], '10k+': []}

    for path in wavs:
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Silence noise profile
        freqs, profile = extract_noise_profile(audio, sr)
        if profile is not None:
            all_silence_profiles.append(profile)

            # Band-specific noise levels
            for band, (lo, hi) in [('80-300', (80, 300)), ('300-1k', (300, 1000)),
                                     ('1k-3k', (1000, 3000)), ('3k-6k', (3000, 6000)),
                                     ('6k-10k', (6000, 10000)), ('10k+', (10000, 12000))]:
                mask = (freqs >= lo) & (freqs <= hi)
                if np.sum(mask) > 0:
                    band_noise[band].append(np.mean(profile[mask]))

        # Speech noise floor
        snf = extract_speech_noise(audio, sr)
        speech_noise_floors.append(snf)

    return {
        'n_clips': len(wavs),
        'silence_noise_mean': np.mean(all_silence_profiles, axis=0) if all_silence_profiles else None,
        'silence_noise_freqs': freqs if all_silence_profiles else None,
        'speech_noise_floor_db': np.mean(speech_noise_floors),
        'speech_noise_std': np.std(speech_noise_floors),
        'band_noise': {k: (np.mean(v), np.std(v)) for k, v in band_noise.items() if v},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Pre-enhancement audio dir")
    parser.add_argument("--after", required=True, help="Post-enhancement audio dir")
    parser.add_argument("--n", type=int, default=30, help="Number of clips to sample")
    args = parser.parse_args()

    before_dir = Path(args.before)
    after_dir = Path(args.after)

    print(f"Analyzing {args.n} clips from each source...\n")

    print("Processing BEFORE (original)...")
    before = analyze_dir(before_dir, args.n)
    print("Processing AFTER (enhanced)...")
    after = analyze_dir(after_dir, args.n)

    print(f"\n{'='*70}")
    print(f"NOISE PROFILE COMPARISON")
    print(f"{'='*70}")

    print(f"\n  Speech noise floor (during voiced frames):")
    print(f"    BEFORE: {before['speech_noise_floor_db']:.1f} dB  (±{before['speech_noise_std']:.1f})")
    print(f"    AFTER:  {after['speech_noise_floor_db']:.1f} dB  (±{after['speech_noise_std']:.1f})")
    diff = after['speech_noise_floor_db'] - before['speech_noise_floor_db']
    print(f"    CHANGE: {diff:+.1f} dB  ({'quieter' if diff < 0 else 'LOUDER' if diff > 0 else 'same'})")

    print(f"\n  Silence noise by frequency band:")
    print(f"  {'Band':<12} {'BEFORE':>12} {'AFTER':>12} {'CHANGE':>12}")
    print(f"  {'-'*48}")
    for band in ['80-300', '300-1k', '1k-3k', '3k-6k', '6k-10k', '10k+']:
        b_mean, b_std = before['band_noise'].get(band, (0, 0))
        a_mean, a_std = after['band_noise'].get(band, (0, 0))
        diff = a_mean - b_mean
        print(f"  {band:<12} {b_mean:>10.1f}dB {a_mean:>10.1f}dB {diff:>+10.1f}dB")

    # Overall silence noise
    if before['silence_noise_mean'] is not None and after['silence_noise_mean'] is not None:
        b_overall = np.mean(before['silence_noise_mean'])
        a_overall = np.mean(after['silence_noise_mean'])
        print(f"\n  Overall silence noise floor:")
        print(f"    BEFORE: {b_overall:.1f} dB")
        print(f"    AFTER:  {a_overall:.1f} dB")
        print(f"    CHANGE: {a_overall - b_overall:+.1f} dB")


if __name__ == "__main__":
    main()
