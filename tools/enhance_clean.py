#!/usr/bin/env python3
"""[CURRENT — use this for all Holler training data and VoiceDesign output]

Enhancement pipeline for clean synthetic TTS audio: 1.7B cloner output and
VoiceDesign candidates. Replaces both enhance_clips.py and enhance_voicedesign.py.
Validated 2026-05-02 against both on Kit, Dakota, Nora, and Joe.

Pipeline: Trim → K-weighted LUFS normalize → IIR notch de-ess

What each stage does and why:

1. TRIM — scans audio in 10ms RMS windows (threshold 0.001, ~-60 dBFS). Finds
   the first and last window above threshold, pads 100ms on each side, applies
   a 20ms linear fade-out at the tail. Removes leading/trailing silence without
   touching speech. 0.001 threshold is intentionally low — only catches true
   silence, never soft consonant tails or decaying vowels.

2. K-WEIGHTED LUFS NORMALIZE — measures loudness per ITU-R BS.1770: runs audio
   through two biquad filters (high-shelf pre-filter + high-pass) to mimic how
   human hearing weights frequencies, then measures RMS of the filtered signal
   and applies -0.691dB offset. Computes gain needed to hit -18 LUFS, then clips
   to a -3dBFS peak ceiling. Linear gain only — no compression, no dynamics.
   Why not RMS+3: the +3 shortcut lands 2-3dB hot depending on spectral content
   (measured: -19.4 LUFS actual vs -22.0 target on Nora using the old shortcut).
   Target changed from -22 to -18 LUFS (2026-05-02) — voice assistant output
   should match podcast/Siri loudness, not broadcast TV.

3. IIR NOTCH at 5500Hz (Q=3.0) — single biquad notch filter targeting the
   center of the sibilance band (S, T, Ch sounds). No STFT, no spectral
   reconstruction, zero chirping or musical noise artifacts. Narrower and more
   effective than the STFT de-esser in enhance_clips.py (measured: 0.0036 vs
   0.0052 sibilance 4-10kHz on Nora, 31% more reduction). Q=3.0 keeps the notch
   tight — only the sibilance center is cut, not the surrounding presence.

What was removed vs enhance_clips.py and why:
- DeepFilterNet3: neural denoiser that takes seconds per clip and adds artifacts
  on audio that has no noise to remove. Zero measured benefit on cloner output.
- STFT de-esser: less effective than IIR notch and introduces spectral chirping.
- Dynamic presence: frequency-band booster that was adding harshness back in
  after the de-esser removed it. Counterproductive on balanced synthetic audio.

Runs in seconds for 500 clips (no model load, pure scipy signal processing).

Usage:
  python enhance_clean.py --input voices/nora/training-data/audio-original --output voices/nora/training-data/audio
  python enhance_clean.py --input <dir> --output <dir> --notch-freq 5500 --notch-q 3.0 --lufs -18
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import iirnotch, lfilter


def trim_silence(audio, sr, pad_ms=100, fade_ms=20, threshold=0.001):
    win = int(sr * 0.01)
    rms = np.array([np.sqrt(np.mean(audio[i:i+win]**2))
                    for i in range(0, len(audio) - win, win)])
    speech = np.where(rms > threshold)[0]
    if len(speech) == 0:
        return audio
    start = max(0, speech[0] * win - int(pad_ms * sr / 1000))
    end = min(len(audio), (speech[-1] + 1) * win + int(pad_ms * sr / 1000))
    trimmed = audio[start:end].copy()
    fade = int(fade_ms * sr / 1000)
    if fade > 0 and len(trimmed) > fade:
        trimmed[-fade:] *= np.linspace(1, 0, fade)
    return trimmed


def compute_lufs(audio, sr):
    """K-weighted integrated loudness per ITU-R BS.1770."""
    from scipy import signal
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    y = signal.lfilter(b1, a1, audio)
    y = signal.lfilter(b2, a2, y)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        return 20 * np.log10(rms) - 0.691
    return -70.0


def lufs_normalize(audio, sr, target_lufs=-18.0, peak_ceiling=-1.0):
    """Normalize to target LUFS with peak ceiling. K-weighted, linear gain only."""
    current = compute_lufs(audio, sr)
    gain_db = target_lufs - current
    audio = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(audio))
    ceiling = 10 ** (peak_ceiling / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio


def notch_deess(audio, sr, freq=5500, q=3.0):
    """IIR notch to tame sibilance. No STFT, no artifacts."""
    b, a = iirnotch(freq, q, fs=sr)
    return lfilter(b, a, audio).astype(np.float32)


def enhance(audio, sr=24000, target_lufs=-18.0, peak_ceiling=-1.0,
            notch_freq=5500, notch_q=3.0, skip_notch=False):
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, target_lufs, peak_ceiling)
    if not skip_notch:
        audio = notch_deess(audio, sr, notch_freq, notch_q)
    return audio


def main():
    parser = argparse.ArgumentParser(description="Enhance clean synthetic TTS audio")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--notch-freq", type=float, default=5500)
    parser.add_argument("--notch-q", type=float, default=3.0)
    parser.add_argument("--lufs", type=float, default=-18.0)
    parser.add_argument("--peak-ceiling", type=float, default=-1.0)
    parser.add_argument("--skip-notch", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(f for f in in_dir.iterdir() if f.suffix == ".wav")
    print(f"Enhancing {len(wavs)} clips: trim → LUFS({args.lufs}) K-weighted → notch({args.notch_freq}Hz Q={args.notch_q})")

    for i, wav in enumerate(wavs):
        data, sr = sf.read(str(wav))
        out = enhance(data, sr, target_lufs=args.lufs, peak_ceiling=args.peak_ceiling,
                      notch_freq=args.notch_freq, notch_q=args.notch_q,
                      skip_notch=args.skip_notch)
        sf.write(str(out_dir / wav.name), out, sr)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(wavs)}]", flush=True)

    print(f"Done — {len(wavs)} clips → {out_dir}")


if __name__ == "__main__":
    main()
