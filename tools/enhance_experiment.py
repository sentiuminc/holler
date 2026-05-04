#!/usr/bin/env python3
"""Per-voice enhancement experiments. Generates named variants for A/B listening.

Usage:
  .venv-enhance-audio/bin/python tools/enhance_experiment.py \
    voices/tessa/training-data/audio-original/clip_0470.wav \
    voices/tessa/training-data/audio-original/clip_0476.wav \
    --output ~/Downloads/tessa-enhance-test
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import iirnotch, lfilter, iirpeak
from pathlib import Path


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
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    y = signal.lfilter(b1, a1, audio)
    y = signal.lfilter(b2, a2, y)
    rms_val = np.sqrt(np.mean(y ** 2))
    if rms_val > 0:
        return 20 * np.log10(rms_val) - 0.691
    return -70.0


def lufs_normalize(audio, sr, target_lufs=-18.0, peak_ceiling=-1.0):
    current = compute_lufs(audio, sr)
    gain_db = target_lufs - current
    audio = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(audio))
    ceiling = 10 ** (peak_ceiling / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio


def parametric_eq(audio, sr, freq, gain_db, q):
    """Peaking EQ (boost or cut) at freq with given Q."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.lfilter(b, a, audio).astype(np.float32)


def low_shelf(audio, sr, gain_db, freq):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cos_w0 = np.cos(w0)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1_c = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2_c = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1_c/a0, a2_c/a0])
    return signal.lfilter(b, a, audio).astype(np.float32)


def high_shelf(audio, sr, gain_db, freq):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cos_w0 = np.cos(w0)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1_c = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2_c = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1_c/a0, a2_c/a0])
    return signal.lfilter(b, a, audio).astype(np.float32)


def notch(audio, sr, freq, q):
    b, a = iirnotch(freq, q, fs=sr)
    return lfilter(b, a, audio).astype(np.float32)


# ── Tessa-tuned enhancement variants ──

def v_current_pipeline(audio, sr):
    """Current enhance_clean.py: trim + LUFS + notch@5500."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = notch(audio, sr, 5500, 3.0)
    return audio


def v_lufs_only(audio, sr):
    """Just trim + LUFS, no EQ at all."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    return audio


def v_gentle_harshcut(audio, sr):
    """Trim + LUFS + gentle 3kHz cut (-1.5dB, Q=2)."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 3000, -1.5, 2.0)
    return audio


def v_moderate_harshcut(audio, sr):
    """Trim + LUFS + moderate 3kHz cut (-2.5dB, Q=2)."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 3000, -2.5, 2.0)
    return audio


def v_harshcut_plus_warmth(audio, sr):
    """Trim + LUFS + 3kHz cut (-2dB) + low shelf +2dB@200Hz."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 3000, -2.0, 2.0)
    audio = low_shelf(audio, sr, 2.0, 200)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    return audio


def v_harshcut_warmth_air(audio, sr):
    """Trim + LUFS + 3kHz cut (-2dB) + low shelf +2dB + high shelf +1dB@9kHz."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 3000, -2.0, 2.0)
    audio = low_shelf(audio, sr, 2.0, 200)
    audio = high_shelf(audio, sr, 1.0, 9000)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    return audio


def v_wide_presence_scoop(audio, sr):
    """Trim + LUFS + wide scoop 2-4kHz (-2dB, Q=1 = very wide)."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 2800, -2.0, 1.0)
    return audio


def v_surgical_2_8k(audio, sr):
    """Trim + LUFS + narrow cut at 2.8kHz (-3dB, Q=4 = tight surgical)."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 2800, -3.0, 4.0)
    return audio


def v_notch_4500(audio, sr):
    """Trim + LUFS + notch@4500 Q=3."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = notch(audio, sr, 4500, 3.0)
    return audio


def v_notch_5000(audio, sr):
    """Trim + LUFS + notch@5000 Q=3."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = notch(audio, sr, 5000, 3.0)
    return audio


def v_notch_6500(audio, sr):
    """Trim + LUFS + notch@6500 Q=3."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = notch(audio, sr, 6500, 3.0)
    return audio


def v_notch_7500(audio, sr):
    """Trim + LUFS + notch@7500 Q=3."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = notch(audio, sr, 7500, 3.0)
    return audio


def v_harshcut_notch_7k(audio, sr):
    """Trim + LUFS + 3kHz cut (-2dB) + notch@7000 Q=3 (harshness + sibilance)."""
    audio = trim_silence(audio, sr)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    audio = parametric_eq(audio, sr, 3000, -2.0, 2.0)
    audio = notch(audio, sr, 7000, 3.0)
    return audio


def v_full_polish(audio, sr):
    """Trim + LUFS + low shelf +2dB + 3kHz cut (-2dB) + notch@7kHz + air +1dB@10kHz."""
    audio = trim_silence(audio, sr)
    audio = low_shelf(audio, sr, 2.0, 200)
    audio = parametric_eq(audio, sr, 3000, -2.0, 2.0)
    audio = notch(audio, sr, 7000, 3.0)
    audio = high_shelf(audio, sr, 1.0, 10000)
    audio = lufs_normalize(audio, sr, -18.0, -1.0)
    return audio


VARIANTS = {
    "01_current_notch5500": v_current_pipeline,
    "02_lufs_only": v_lufs_only,
    "03_notch_4500": v_notch_4500,
    "04_notch_5000": v_notch_5000,
    "05_notch_6500": v_notch_6500,
    "06_notch_7500": v_notch_7500,
    "07_gentle_harshcut_3k": v_gentle_harshcut,
    "08_moderate_harshcut_3k": v_moderate_harshcut,
    "09_harshcut_warmth": v_harshcut_plus_warmth,
    "10_harshcut_warmth_air": v_harshcut_warmth_air,
    "11_wide_presence_scoop": v_wide_presence_scoop,
    "12_surgical_2_8k": v_surgical_2_8k,
    "13_harshcut_notch7k": v_harshcut_notch_7k,
    "14_full_polish": v_full_polish,
}


def main():
    parser = argparse.ArgumentParser(description="Generate enhancement variants for A/B comparison")
    parser.add_argument("files", nargs="+", help="Input WAV file(s)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in args.files:
        wav_path = Path(wav_path)
        data, sr = sf.read(str(wav_path))
        if len(data.shape) > 1:
            data = data[:, 0]
        stem = wav_path.stem

        print(f"\n{'='*60}")
        print(f"  {stem}  ({len(data)/sr:.1f}s, {sr}Hz)")
        print(f"{'='*60}")

        for name, fn in VARIANTS.items():
            out = fn(data.copy(), sr)
            out_path = out_dir / f"{name}_{stem}.wav"
            sf.write(str(out_path), out, sr)

            peak_db = 20 * np.log10(np.max(np.abs(out)) + 1e-10)
            rms_db = 20 * np.log10(np.sqrt(np.mean(out**2)) + 1e-10)
            lufs = compute_lufs(out, sr)
            print(f"  {name:<30} peak={peak_db:>5.1f}  rms={rms_db:>5.1f}  lufs={lufs:>5.1f}")

    print(f"\nAll variants in {out_dir}/")


if __name__ == "__main__":
    main()
