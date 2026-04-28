#!/usr/bin/env python3
"""Enhancement pipeline for VoiceDesign output.

Separate from enhance_clips.py (which is for training data from the 1.7B cloner).
VoiceDesign output is already synthetic and clean — it doesn't need DeepFilter
denoising or STFT-based spectral processing, both of which introduce chirping
artifacts on already-clean TTS audio.

Pipeline: Trim → LUFS normalize → IIR notch at 5.5kHz

The IIR notch replaces the STFT de-esser. It cuts the center of the sibilance
band (where S/T/Ch sounds live) without any spectral processing, so zero
chirping/musical noise artifacts.

Usage:
  python enhance_voicedesign.py --input ~/Downloads/voice-batches/batch_1 --output ~/Downloads/voice-batches/batch_1/enhanced
  python enhance_voicedesign.py --input some/folder --output some/folder/enhanced --notch-freq 5500 --notch-q 3.0 --lufs -22
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import iirnotch, lfilter


def trim_silence(audio, sr, pad_ms=100, fade_ms=20, threshold=0.001):
    """Trim leading/trailing silence, keep padding, fade out."""
    win = int(sr * 0.01)
    rms = np.array([np.sqrt(np.mean(audio[i:i+win]**2))
                     for i in range(0, len(audio) - win, win)])
    speech = np.where(rms > threshold)[0]
    if len(speech) == 0:
        return audio

    start_sample = max(0, speech[0] * win - int(pad_ms * sr / 1000))
    end_sample = min(len(audio), (speech[-1] + 1) * win + int(pad_ms * sr / 1000))
    trimmed = audio[start_sample:end_sample].copy()

    fade_samples = int(fade_ms * sr / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        trimmed[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return trimmed


def lufs_normalize(audio, sr, target_lufs=-22.0, peak_ceiling=-3.0):
    """LUFS-approximate normalization with peak ceiling."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-10:
        return audio
    current_db = 20 * np.log10(rms)
    target_db = target_lufs + 3
    gain_db = target_db - current_db

    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)
    max_gain_db = peak_ceiling - peak_db
    gain_db = min(gain_db, max_gain_db)

    return audio * (10 ** (gain_db / 20))


def notch_deess(audio, sr, freq=5500, q=3.0):
    """Simple IIR notch filter to tame sibilance. No STFT, no artifacts."""
    b, a = iirnotch(freq, q, fs=sr)
    return lfilter(b, a, audio).astype(np.float32)


def enhance_voicedesign(audio, sr=24000, target_lufs=-22.0, peak_ceiling=-3.0,
                        notch_freq=5500, notch_q=3.0, skip_notch=False):
    """Full VoiceDesign enhancement: trim → LUFS → notch."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, target_lufs, peak_ceiling)
    if not skip_notch:
        audio = notch_deess(audio, sr, notch_freq, notch_q)
    return audio


def main():
    parser = argparse.ArgumentParser(description="Enhance VoiceDesign output (no STFT, no DeepFilter)")
    parser.add_argument("--input", required=True, help="Input directory with .wav files")
    parser.add_argument("--output", required=True, help="Output directory for enhanced files")
    parser.add_argument("--notch-freq", type=float, default=5500, help="Notch center frequency (default: 5500 Hz)")
    parser.add_argument("--notch-q", type=float, default=3.0, help="Notch Q factor (default: 3.0)")
    parser.add_argument("--lufs", type=float, default=-22.0, help="Target LUFS (default: -22)")
    parser.add_argument("--peak-ceiling", type=float, default=-3.0, help="Peak ceiling dBFS (default: -3)")
    parser.add_argument("--skip-notch", action="store_true", help="Skip notch filter")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(f for f in in_dir.iterdir() if f.suffix == ".wav")
    print(f"Enhancing {len(wavs)} clips: trim → LUFS({args.lufs}) → notch({args.notch_freq}Hz, Q={args.notch_q})")

    for i, wav in enumerate(wavs):
        data, sr = sf.read(str(wav))
        enhanced = enhance_voicedesign(
            data, sr,
            target_lufs=args.lufs,
            peak_ceiling=args.peak_ceiling,
            notch_freq=args.notch_freq,
            notch_q=args.notch_q,
            skip_notch=args.skip_notch,
        )
        sf.write(str(out_dir / wav.name), enhanced, sr)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(wavs)}]", flush=True)

    print(f"Done — {len(wavs)} clips enhanced to {out_dir}")


if __name__ == "__main__":
    main()
