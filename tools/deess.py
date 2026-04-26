#!/usr/bin/env python3
"""Spectral de-esser: STFT-based per-bin adaptive sibilance reduction.

Targets exact frequency bins that spike above their local median in the sibilance
region. More transparent than sidechain compressor approach — only reduces what
actually sticks out, leaves everything else untouched.

Usage:
  python deess.py --input clip.wav --output clip_deessed.wav
  python deess.py --dir /tmp/joe-clips --out /tmp/joe-deessed
  python deess.py --dir /tmp/joe-clips --out /tmp/joe-deessed --threshold -6 --max-reduction 10
"""
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from pathlib import Path


def spectral_deess(audio, sr=24000, low_hz=4000, high_hz=10000,
                   threshold_db=-8.0, max_reduction_db=10.0,
                   ratio=4.0, smoothing_frames=3, lookahead_ms=8.0):
    """STFT-based spectral de-esser with adaptive per-bin threshold.

    Each frequency bin in [low_hz, high_hz] uses its own median energy as
    the baseline. Bins that spike above median + threshold_db get compressed.
    Bins outside the sibilance range pass through unchanged.

    Args:
        low_hz/high_hz: sibilance detection/reduction band
        threshold_db: how far above median a bin must spike to trigger (lower = more aggressive)
        max_reduction_db: maximum gain cut per bin (positive number, applied as negative)
        ratio: compression ratio (4.0 = 4:1)
        smoothing_frames: temporal smoothing to avoid per-frame jitter
        lookahead_ms: delay audio relative to gain to catch sibilant onsets
    """
    nperseg = 1024
    noverlap = nperseg * 3 // 4
    hop = nperseg - noverlap

    # Lookahead: delay the audio signal
    lookahead_samples = int(lookahead_ms * 0.001 * sr)
    if lookahead_samples > 0:
        audio_delayed = np.pad(audio, (0, lookahead_samples))[lookahead_samples:]
        audio_for_analysis = audio
    else:
        audio_delayed = audio
        audio_for_analysis = audio

    f, t, Zxx = stft(audio_delayed, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_detect = stft(audio_for_analysis, fs=sr, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    detect_mag_db = 20 * np.log10(np.abs(Zxx_detect) + 1e-10)

    bin_lo = max(1, int(low_hz * nperseg / sr))
    bin_hi = min(detect_mag_db.shape[0] - 1, int(high_hz * nperseg / sr))

    gain_db = np.zeros_like(detect_mag_db)
    total_reduction = np.zeros(detect_mag_db.shape[1])

    for k in range(bin_lo, bin_hi + 1):
        median_level = np.median(detect_mag_db[k, :])
        excess = detect_mag_db[k, :] - median_level - threshold_db
        excess = np.maximum(excess, 0)

        reduction = excess * (1.0 - 1.0 / ratio)
        reduction = np.minimum(reduction, max_reduction_db)

        gain_db[k, :] = -reduction
        total_reduction += reduction

    # Temporal smoothing
    if smoothing_frames > 1:
        kernel = np.ones(smoothing_frames) / smoothing_frames
        for k in range(bin_lo, bin_hi + 1):
            gain_db[k, :] = np.convolve(gain_db[k, :], kernel, mode='same')

    gain_linear = 10 ** (gain_db / 20)
    Zxx_out = magnitude * gain_linear * np.exp(1j * phase)

    _, audio_out = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap)
    audio_out = audio_out[:len(audio)].astype(np.float32)

    # Stats
    frames_active = np.sum(total_reduction > 0.5)
    max_red = np.max(total_reduction / max(1, bin_hi - bin_lo))
    mean_red = np.mean(total_reduction[total_reduction > 0.5] / max(1, bin_hi - bin_lo)) if frames_active > 0 else 0

    return audio_out, {
        'max_reduction_db': round(float(max_red), 1),
        'mean_reduction_db': round(float(mean_red), 1),
        'active_pct': round(float(frames_active / detect_mag_db.shape[1] * 100), 0),
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single WAV file")
    group.add_argument("--dir", help="Directory of WAV files")
    parser.add_argument("--output", help="Output file (for --input)")
    parser.add_argument("--out", help="Output directory (for --dir)")
    parser.add_argument("--threshold", type=float, default=-8.0)
    parser.add_argument("--max-reduction", type=float, default=10.0)
    parser.add_argument("--ratio", type=float, default=4.0)
    parser.add_argument("--low", type=float, default=4000)
    parser.add_argument("--high", type=float, default=10000)
    parser.add_argument("--lookahead", type=float, default=8.0, help="Lookahead in ms")
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    kwargs = dict(low_hz=args.low, high_hz=args.high, threshold_db=args.threshold,
                  max_reduction_db=args.max_reduction, ratio=args.ratio,
                  smoothing_frames=args.smoothing, lookahead_ms=args.lookahead)

    if args.input:
        audio, sr = sf.read(args.input)
        out_path = args.output or args.input.replace('.wav', '_deessed.wav')
        deessed, stats = spectral_deess(audio, sr, **kwargs)
        sf.write(out_path, deessed, sr)
        print(f"{Path(args.input).name} → {Path(out_path).name}  "
              f"(max: {stats['max_reduction_db']}dB, mean: {stats['mean_reduction_db']}dB, active: {stats['active_pct']}%)")
    else:
        src = Path(args.dir)
        out = Path(args.out or str(src) + "-deessed")
        out.mkdir(exist_ok=True)

        wavs = sorted(f for f in src.iterdir() if f.suffix == '.wav')
        print(f"Spectral de-essing {len(wavs)} clips")
        print(f"  Band: {args.low}-{args.high} Hz, threshold: {args.threshold} dB, ratio: {args.ratio}:1")
        print(f"  Max reduction: {args.max_reduction} dB, lookahead: {args.lookahead} ms\n")

        for i, path in enumerate(wavs):
            audio, sr = sf.read(path)
            deessed, stats = spectral_deess(audio, sr, **kwargs)
            sf.write(str(out / path.name), deessed, sr)
            print(f"  [{i+1}/{len(wavs)}] {path.name}: max {stats['max_reduction_db']}dB, "
                  f"mean {stats['mean_reduction_db']}dB, active {stats['active_pct']}%")

        print(f"\nDone! Output: {out}")


if __name__ == "__main__":
    main()
