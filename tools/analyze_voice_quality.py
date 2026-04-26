#!/usr/bin/env python3
"""Comprehensive voice quality analysis for TTS training data and outputs.

Measures everything that influences voice quality:
- Levels: peak dBFS, RMS dBFS, crest factor, LUFS (integrated)
- Spectrum: centroid, harshness (2-4k), sibilance (4-10k), presence (1-5k),
  spectral tilt (dB/oct), spectral flatness, rolloff
- Voice quality: F0 (pitch mean/std/range), jitter %, shimmer %, HNR dB
- Dynamics: silence ratio, dynamic range
- Noise: DNSMOS (SIG/BAK/OVRL — no-reference perceptual quality)

Usage:
  python analyze_voice_quality.py --source ~/Downloads/qwen3-builtin-voices --label "CustomVoice"
  python analyze_voice_quality.py --source voices/katie/training-data/audio --label "Katie train" --n 80
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import parselmouth
from parselmouth.praat import call
import json
import sys
import warnings
warnings.filterwarnings('ignore')

SR = 24000

# Try to load DNSMOS
try:
    import torch
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
    _dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
    HAS_DNSMOS = True
    print("DNSMOS loaded.")
except Exception as e:
    HAS_DNSMOS = False
    print(f"DNSMOS not available: {e}")


def compute_lufs(audio, sr):
    """Simplified integrated LUFS (K-weighted RMS)."""
    # K-weighting: pre-filter (high shelf +4dB at 1681Hz) + RLB filter (high-pass 38Hz)
    # Simplified: apply the two biquad stages
    # Stage 1: high shelf
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    # Stage 2: high-pass (RLB weighting)
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])

    y = signal.lfilter(b1, a1, audio)
    y = signal.lfilter(b2, a2, y)

    # Gated loudness (simplified — no gating for short clips)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        return 20 * np.log10(rms) - 0.691
    return -70.0


def compute_dnsmos(audio, sr):
    """DNSMOS P.835 scores (SIG, BAK, OVRL). Requires 16kHz."""
    if not HAS_DNSMOS:
        return None, None, None
    try:
        if sr != 16000:
            audio_16k = signal.resample_poly(audio, 16000, sr).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)
        tensor = torch.from_numpy(audio_16k).unsqueeze(0)
        scores = _dnsmos(tensor)
        # Returns [p808_mos, sig, bak, ovrl]
        return round(float(scores[1]), 2), round(float(scores[2]), 2), round(float(scores[3]), 2)
    except Exception:
        return None, None, None


def analyze_clip(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    n = len(audio)
    duration = n / sr

    # === LEVELS ===
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    crest_factor = peak_db - rms_db
    lufs = compute_lufs(audio, sr)
    dc_offset = float(np.abs(np.mean(audio)))

    # === SPECTRAL (Welch PSD) ===
    freqs, psd = signal.welch(audio, sr, nperseg=2048)
    total_power = np.sum(psd)

    centroid = np.sum(freqs * psd) / (total_power + 1e-10)

    def band_ratio(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        return np.sum(psd[mask]) / (total_power + 1e-10)

    harsh = band_ratio(2000, 4000)
    sibilance = band_ratio(4000, 10000)
    presence = band_ratio(1000, 5000)
    low_end = band_ratio(80, 300)
    high_air = band_ratio(10000, 12000)

    rolloff_idx = np.searchsorted(np.cumsum(psd), 0.85 * total_power)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Spectral tilt: linear regression of log-PSD vs log-freq
    voiced_mask = (freqs >= 100) & (freqs <= 8000)
    if np.sum(voiced_mask) > 10:
        log_f = np.log2(freqs[voiced_mask] + 1e-10)
        log_psd = 10 * np.log10(psd[voiced_mask] + 1e-10)
        coeffs = np.polyfit(log_f, log_psd, 1)
        spectral_tilt = coeffs[0]
    else:
        spectral_tilt = 0.0

    # Spectral flatness (Wiener entropy)
    psd_pos = psd[psd > 0]
    if len(psd_pos) > 0:
        log_mean = np.exp(np.mean(np.log(psd_pos + 1e-20)))
        arith_mean = np.mean(psd_pos)
        spectral_flatness = log_mean / (arith_mean + 1e-10)
    else:
        spectral_flatness = 0.0

    # === DYNAMICS ===
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames_audio = [audio[i:i+frame_len] for i in range(0, n - frame_len, hop)]
    frame_rms = np.array([np.sqrt(np.mean(f**2)) for f in frames_audio])
    frame_db = 20 * np.log10(frame_rms + 1e-10)
    silence_ratio = np.mean(frame_db < -40)

    # Dynamic range: difference between 95th percentile and 5th percentile of frame levels
    if len(frame_db) > 10:
        dynamic_range = float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5))
    else:
        dynamic_range = 0.0

    # === VOICE QUALITY (Praat) ===
    snd = parselmouth.Sound(audio, sampling_frequency=sr)

    # F0 (pitch)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_values = pitch.selected_array['frequency']
    f0_voiced = f0_values[f0_values > 0]

    if len(f0_voiced) > 5:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        f0_range = float(np.max(f0_voiced) - np.min(f0_voiced))
        f0_voiced_ratio = len(f0_voiced) / len(f0_values) if len(f0_values) > 0 else 0
    else:
        f0_mean = f0_std = f0_range = 0.0
        f0_voiced_ratio = 0.0

    # Jitter and shimmer
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    try:
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = 0.0
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = 0.0

    # HNR (Harmonics-to-Noise Ratio)
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        if hnr == -200:
            hnr = 0.0
    except Exception:
        hnr = 0.0

    # Formants (F1, F2, F3 means)
    try:
        formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1_vals = [call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
                   for t in np.linspace(0, duration, 20)]
        f2_vals = [call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
                   for t in np.linspace(0, duration, 20)]
        f3_vals = [call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
                   for t in np.linspace(0, duration, 20)]
        f1_mean = float(np.nanmean([v for v in f1_vals if v == v and v > 0]))
        f2_mean = float(np.nanmean([v for v in f2_vals if v == v and v > 0]))
        f3_mean = float(np.nanmean([v for v in f3_vals if v == v and v > 0]))
    except Exception:
        f1_mean = f2_mean = f3_mean = 0.0

    # === DNSMOS ===
    sig, bak, ovrl = compute_dnsmos(audio, sr)

    return {
        'file': path.name,
        'dur': round(duration, 2),
        # Levels
        'peak_db': round(peak_db, 1),
        'rms_db': round(rms_db, 1),
        'crest_db': round(crest_factor, 1),
        'lufs': round(lufs, 1),
        'dc_offset': round(dc_offset, 5),
        # Spectrum
        'centroid_hz': round(float(centroid)),
        'harsh_2_4k': round(float(harsh), 4),
        'sib_4_10k': round(float(sibilance), 4),
        'presence_1_5k': round(float(presence), 4),
        'low_80_300': round(float(low_end), 4),
        'air_10k': round(float(high_air), 4),
        'rolloff_hz': round(float(rolloff)),
        'tilt_db_oct': round(float(spectral_tilt), 2),
        'flatness': round(float(spectral_flatness), 4),
        # Dynamics
        'silence': round(silence_ratio, 3),
        'dyn_range_db': round(dynamic_range, 1),
        # Voice quality
        'f0_mean': round(f0_mean, 1),
        'f0_std': round(f0_std, 1),
        'f0_range': round(f0_range, 1),
        'f0_voiced': round(f0_voiced_ratio, 3),
        'jitter_pct': round(jitter * 100, 3) if jitter else 0.0,
        'shimmer_pct': round(shimmer * 100, 2) if shimmer else 0.0,
        'hnr_db': round(hnr, 1),
        'f1_hz': round(f1_mean),
        'f2_hz': round(f2_mean),
        'f3_hz': round(f3_mean),
        # DNSMOS
        'dnsmos_sig': sig,
        'dnsmos_bak': bak,
        'dnsmos_ovrl': ovrl,
    }


METRIC_GROUPS = [
    ("LEVELS", [
        ('peak_db', 'Peak dBFS', '.1f'),
        ('rms_db', 'RMS dBFS', '.1f'),
        ('crest_db', 'Crest Factor dB', '.1f'),
        ('lufs', 'LUFS (integ)', '.1f'),
    ]),
    ("SPECTRUM", [
        ('centroid_hz', 'Centroid Hz', '.0f'),
        ('harsh_2_4k', 'Harsh 2-4kHz', '.4f'),
        ('sib_4_10k', 'Sibilance 4-10k', '.4f'),
        ('presence_1_5k', 'Presence 1-5k', '.4f'),
        ('low_80_300', 'Low 80-300Hz', '.4f'),
        ('air_10k', 'Air 10k+', '.4f'),
        ('tilt_db_oct', 'Tilt dB/oct', '.2f'),
        ('flatness', 'Flatness', '.4f'),
        ('rolloff_hz', 'Rolloff Hz', '.0f'),
    ]),
    ("DYNAMICS", [
        ('silence', 'Silence ratio', '.3f'),
        ('dyn_range_db', 'Dyn range dB', '.1f'),
    ]),
    ("VOICE QUALITY", [
        ('f0_mean', 'F0 mean Hz', '.1f'),
        ('f0_std', 'F0 stddev Hz', '.1f'),
        ('f0_range', 'F0 range Hz', '.1f'),
        ('f0_voiced', 'Voiced ratio', '.3f'),
        ('jitter_pct', 'Jitter %', '.3f'),
        ('shimmer_pct', 'Shimmer %', '.2f'),
        ('hnr_db', 'HNR dB', '.1f'),
        ('f1_hz', 'F1 Hz', '.0f'),
        ('f2_hz', 'F2 Hz', '.0f'),
        ('f3_hz', 'F3 Hz', '.0f'),
    ]),
    ("DNSMOS (perceptual)", [
        ('dnsmos_sig', 'SIG (speech)', '.2f'),
        ('dnsmos_bak', 'BAK (noise)', '.2f'),
        ('dnsmos_ovrl', 'OVRL (overall)', '.2f'),
    ]),
]


def print_summary(results, label):
    print(f"\n{'='*85}")
    print(f"  {label}  ({len(results)} clips)")
    print(f"{'='*85}")

    for group_name, metrics in METRIC_GROUPS:
        print(f"\n  --- {group_name} ---")
        print(f"  {'Metric':<18} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
        print(f"  {'-'*68}")
        for key, name, fmt in metrics:
            vals = [r[key] for r in results if r[key] is not None and (r[key] != 0.0 or key in ('peak_db', 'rms_db', 'tilt_db_oct', 'lufs'))]
            if not vals:
                print(f"  {name:<18} {'n/a':>10}")
                continue
            print(f"  {name:<18} {np.mean(vals):>10{fmt}} {np.median(vals):>10{fmt}} {np.min(vals):>10{fmt}} {np.max(vals):>10{fmt}} {np.std(vals):>10{fmt}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory of WAV files")
    parser.add_argument("--label", default=None, help="Label for this source")
    parser.add_argument("--n", type=int, default=None, help="Sample N clips evenly")
    parser.add_argument("--json", default=None, help="Save raw results to JSON")
    args = parser.parse_args()

    source = Path(args.source)
    label = args.label or source.name

    if not source.exists():
        print(f"Error: {source} not found")
        return

    wavs = sorted(f for f in source.iterdir() if f.suffix == '.wav')
    if not wavs:
        print(f"No WAV files in {source}")
        return

    if args.n and args.n < len(wavs):
        indices = np.linspace(0, len(wavs) - 1, args.n, dtype=int)
        wavs = [wavs[i] for i in indices]

    print(f"\nAnalyzing {len(wavs)} clips from {source}...")

    results = []
    for i, path in enumerate(wavs):
        try:
            r = analyze_clip(path)
            results.append(r)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(wavs)}]")

    print_summary(results, label)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRaw results saved to {args.json}")


if __name__ == "__main__":
    main()
