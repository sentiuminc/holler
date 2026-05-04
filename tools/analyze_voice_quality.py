#!/usr/bin/env python3
"""Voice quality analysis for TTS training data. Single script for all analysis needs.

Accepts a directory, single file, or multiple files:
  python analyze_voice_quality.py --source voices/tessa/training-data/audio --label "Tessa" --n 80
  python analyze_voice_quality.py --source clip_0470.wav clip_0476.wav --label "two clips"
  python analyze_voice_quality.py --source clip_0470.wav  # single file

Compare mode (--compare): A/B analysis between original and processed clips:
  python analyze_voice_quality.py --compare --original voices/kit/training-data/audio-original --processed voices/kit/training-data/audio --label "Kit" --n 50

Metrics measured:
- Levels: peak dBFS, RMS dBFS, crest factor, LUFS (integrated)
- Noise: noise floor dB, SNR dB
- Spectrum: centroid, harshness (2-4k), sibilance (4-10k), presence (1-5k),
  spectral tilt (dB/oct), spectral flatness, rolloff
- Voice quality: F0 (pitch mean/std/range), jitter %, shimmer %, HNR dB
- Dynamics: silence ratio, dynamic range
- Perceptual: DNSMOS P.835 (SIG/BAK/OVRL)

Prints summary stats + worst outliers per category. Use --json for per-clip data.
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
    silence_threshold = -40
    silence_ratio = np.mean(frame_db < silence_threshold)

    noise_frames = frame_db[frame_db < silence_threshold]
    noise_floor_db = float(np.median(noise_frames)) if len(noise_frames) > 5 else -80.0

    signal_frames = frame_db[frame_db >= silence_threshold]
    snr_db = float(np.median(signal_frames)) - noise_floor_db if len(signal_frames) > 0 else 0.0

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
        # Noise
        'noise_floor_db': round(noise_floor_db, 1),
        'snr_db': round(snr_db, 1),
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


# ============================================================
# Compare mode: A/B artifact analysis
# ============================================================

def _detect_transients(audio, sr, threshold_db=6.0, min_gap_ms=30):
    """Find transient onset positions (plosives, clicks)."""
    win = int(0.003 * sr)  # 3ms for fast transients
    hop = win // 2
    n_frames = (len(audio) - win) // hop + 1
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        frame = audio[s:s+win]
        rms = np.sqrt(np.mean(frame**2))
        energy[i] = 20 * np.log10(rms + 1e-10)
    energy_diff = np.diff(energy)
    min_gap_frames = int(min_gap_ms * sr / 1000 / hop)
    onsets = []
    last = -min_gap_frames
    for i, d in enumerate(energy_diff):
        if d > threshold_db and (i - last) >= min_gap_frames:
            onsets.append(i * hop)
            last = i
    return onsets


def _plosive_harshness(audio, sr, onsets, band_lo=3000, band_hi=6000, window_ms=10):
    """Measure energy burst in plosive band at each transient, relative to clip average."""
    if not onsets:
        return 0.0, 0
    sos = signal.butter(4, [band_lo, band_hi], btype='band', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    avg_rms = np.sqrt(np.mean(filtered**2))
    avg_db = 20 * np.log10(avg_rms + 1e-10)
    win = int(window_ms * sr / 1000)
    ratios = []
    for onset in onsets:
        end = min(onset + win, len(filtered))
        burst = filtered[onset:end]
        if len(burst) < win // 2:
            continue
        burst_rms = np.sqrt(np.mean(burst**2))
        burst_db = 20 * np.log10(burst_rms + 1e-10)
        ratios.append(burst_db - avg_db)
    if not ratios:
        return 0.0, 0
    return float(np.mean(ratios)), len(ratios)


def compare_clips(orig_path, proc_path, min_duration=1.5):
    """A/B comparison between original and processed clip."""
    orig, sr_o = sf.read(orig_path)
    proc, sr_p = sf.read(proc_path)
    sr = sr_o

    # Align lengths (processing may trim)
    min_len = min(len(orig), len(proc))
    if min_len == 0:
        return None

    # Short clips produce unreliable spectral metrics
    if min_len / sr < min_duration:
        return {'file': Path(orig_path).name, '_skipped': True,
                'dur_orig': round(len(orig)/sr, 2), 'dur_proc': round(len(proc)/sr, 2),
                'skip_reason': f'too short ({min_len/sr:.1f}s < {min_duration}s)'}

    # For difference analysis, use the shorter length
    orig_short = orig[:min_len]
    proc_short = proc[:min_len]

    # --- Difference signal analysis ---
    diff = proc_short - orig_short
    freqs_d = np.fft.rfftfreq(min_len, 1/sr)
    spec_diff = np.abs(np.fft.rfft(diff))
    spec_orig = np.abs(np.fft.rfft(orig_short))
    total_orig = np.sum(spec_orig**2) + 1e-10

    def diff_band_ratio(lo, hi):
        mask = (freqs_d >= lo) & (freqs_d <= hi)
        return float(np.sum(spec_diff[mask]**2) / total_orig)

    artifact_low = diff_band_ratio(80, 2000)    # should be near 0
    artifact_mid = diff_band_ratio(2000, 4000)   # harshness region
    artifact_high = diff_band_ratio(4000, 10000) # sibilance region
    artifact_total = diff_band_ratio(80, 12000)

    # --- De-essing effectiveness ---
    freqs_w, psd_orig = signal.welch(orig_short, sr, nperseg=2048)
    _, psd_proc = signal.welch(proc_short, sr, nperseg=2048)
    sib_mask = (freqs_w >= 4000) & (freqs_w <= 10000)
    clarity_mask = (freqs_w >= 1000) & (freqs_w <= 4000)
    total_orig_w = np.sum(psd_orig) + 1e-10
    total_proc_w = np.sum(psd_proc) + 1e-10

    sib_orig = np.sum(psd_orig[sib_mask]) / total_orig_w
    sib_proc = np.sum(psd_proc[sib_mask]) / total_proc_w
    sib_reduction = (sib_orig - sib_proc) / (sib_orig + 1e-10)

    clarity_orig = np.sum(psd_orig[clarity_mask]) / total_orig_w
    clarity_proc = np.sum(psd_proc[clarity_mask]) / total_proc_w
    clarity_change = (clarity_proc - clarity_orig) / (clarity_orig + 1e-10)

    # --- Plosive harshness (on processed) ---
    onsets = _detect_transients(proc_short, sr)
    plosive_harsh, n_transients = _plosive_harshness(proc_short, sr, onsets)
    # Also measure on original for comparison
    onsets_orig = _detect_transients(orig_short, sr)
    plosive_harsh_orig, _ = _plosive_harshness(orig_short, sr, onsets_orig)
    plosive_delta = plosive_harsh - plosive_harsh_orig

    # --- Noise floor comparison ---
    # Measure RMS in quiet portions (bottom 10% energy frames)
    win = int(0.025 * sr)
    hop_n = int(0.010 * sr)
    frames_orig = [orig_short[i:i+win] for i in range(0, min_len - win, hop_n)]
    frames_proc = [proc_short[i:i+win] for i in range(0, min_len - win, hop_n)]
    rms_orig = np.array([np.sqrt(np.mean(f**2)) for f in frames_orig])
    rms_proc = np.array([np.sqrt(np.mean(f**2)) for f in frames_proc])
    quiet_thresh = np.percentile(rms_orig, 10)
    quiet_mask = rms_orig <= quiet_thresh
    if np.sum(quiet_mask) > 2:
        noise_orig = 20 * np.log10(np.mean(rms_orig[quiet_mask]) + 1e-10)
        noise_proc = 20 * np.log10(np.mean(rms_proc[quiet_mask]) + 1e-10)
        noise_reduction = noise_orig - noise_proc
    else:
        noise_orig = noise_proc = noise_reduction = 0.0

    # --- Full quality metrics on both ---
    orig_metrics = analyze_clip(orig_path)
    proc_metrics = analyze_clip(proc_path)

    result = {
        'file': Path(orig_path).name,
        'dur_orig': round(len(orig) / sr, 2),
        'dur_proc': round(len(proc) / sr, 2),
        # Difference signal energy by band
        'artifact_low': round(artifact_low, 6),
        'artifact_mid': round(artifact_mid, 6),
        'artifact_high': round(artifact_high, 6),
        'artifact_total': round(artifact_total, 6),
        # De-essing
        'sib_reduction_%': round(sib_reduction * 100, 2),
        'clarity_change_%': round(clarity_change * 100, 2),
        # Plosives
        'plosive_harsh_db': round(plosive_harsh, 1),
        'plosive_delta_db': round(plosive_delta, 1),
        'n_transients': n_transients,
        # Noise
        'noise_orig_db': round(noise_orig, 1),
        'noise_proc_db': round(noise_proc, 1),
        'noise_reduction_db': round(noise_reduction, 1),
    }

    # Add full metrics as orig_* and proc_*, plus delta_*
    skip_keys = {'file', 'dur'}
    for key in orig_metrics:
        if key in skip_keys:
            continue
        oval = orig_metrics[key]
        pval = proc_metrics.get(key)
        result[f'orig_{key}'] = oval
        result[f'proc_{key}'] = pval
        if oval is not None and pval is not None and isinstance(oval, (int, float)) and isinstance(pval, (int, float)):
            result[f'delta_{key}'] = round(pval - oval, 4)

    return result


COMPARE_METRICS = [
    ("ARTIFACTS", [
        ('artifact_total', 'Artifact total', '.6f', 'Total difference signal energy / original energy'),
        ('artifact_mid', 'Artifact 2-4k', '.6f', 'Difference energy in harshness band'),
        ('artifact_high', 'Artifact 4-10k', '.6f', 'Difference energy in sibilance band'),
    ]),
    ("DE-ESSING", [
        ('sib_reduction_%', 'Sibilance reduction %', '.2f', '>0 = less sibilance. Negative = added sibilance'),
        ('clarity_change_%', 'Clarity change %', '.2f', 'Change in 1-4kHz. Should stay near 0'),
    ]),
    ("PLOSIVES", [
        ('plosive_harsh_db', 'Plosive burst dB', '.1f', 'Transient energy in 3-6kHz vs clip average'),
        ('plosive_delta_db', 'Plosive delta dB', '.1f', 'Change from original. >0 = harsher plosives'),
        ('n_transients', 'Transients found', '.0f', 'Number of detected transient onsets'),
    ]),
    ("NOISE", [
        ('noise_orig_db', 'Noise floor orig', '.1f', 'RMS in quiet frames (original)'),
        ('noise_proc_db', 'Noise floor proc', '.1f', 'RMS in quiet frames (processed)'),
        ('noise_reduction_db', 'Noise reduction dB', '.1f', '>0 = quieter noise floor'),
    ]),
]


def print_compare_summary(results, label):
    print(f"\n{'='*95}")
    print(f"  COMPARE: {label}  ({len(results)} clip pairs)")
    print(f"{'='*95}")

    for group_name, metrics in COMPARE_METRICS:
        print(f"\n  --- {group_name} ---")
        print(f"  {'Metric':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
        print(f"  {'-'*75}")
        for key, name, fmt, _ in metrics:
            vals = [r[key] for r in results if r.get(key) is not None]
            if not vals:
                print(f"  {name:<25} {'n/a':>10}")
                continue
            print(f"  {name:<25} {np.mean(vals):>10{fmt}} {np.median(vals):>10{fmt}} "
                  f"{np.min(vals):>10{fmt}} {np.max(vals):>10{fmt}} {np.std(vals):>10{fmt}}")

    # Full quality metric deltas (orig → proc)
    DELTA_METRICS = [
        ("LEVELS (delta)", [
            ('delta_peak_db', 'Δ Peak dBFS', '.1f'),
            ('delta_rms_db', 'Δ RMS dBFS', '.1f'),
            ('delta_lufs', 'Δ LUFS', '.1f'),
        ]),
        ("SPECTRUM (delta)", [
            ('delta_centroid_hz', 'Δ Centroid Hz', '.0f'),
            ('delta_harsh_2_4k', 'Δ Harsh 2-4kHz', '.4f'),
            ('delta_sib_4_10k', 'Δ Sibilance 4-10k', '.4f'),
            ('delta_presence_1_5k', 'Δ Presence 1-5k', '.4f'),
            ('delta_tilt_db_oct', 'Δ Tilt dB/oct', '.2f'),
            ('delta_air_10k', 'Δ Air 10k+', '.4f'),
        ]),
        ("VOICE QUALITY (delta)", [
            ('delta_hnr_db', 'Δ HNR dB', '.1f'),
            ('delta_jitter_pct', 'Δ Jitter %', '.3f'),
            ('delta_shimmer_pct', 'Δ Shimmer %', '.2f'),
        ]),
        ("PERCEPTUAL (delta)", [
            ('delta_dnsmos_sig', 'Δ DNSMOS SIG', '.2f'),
            ('delta_dnsmos_bak', 'Δ DNSMOS BAK', '.2f'),
            ('delta_dnsmos_ovrl', 'Δ DNSMOS OVRL', '.2f'),
        ]),
    ]

    for group_name, metrics in DELTA_METRICS:
        print(f"\n  --- {group_name} ---")
        print(f"  {'Metric':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
        print(f"  {'-'*75}")
        for key, name, fmt in metrics:
            vals = [r[key] for r in results if r.get(key) is not None]
            if not vals:
                print(f"  {name:<25} {'n/a':>10}")
                continue
            print(f"  {name:<25} {np.mean(vals):>10{fmt}} {np.median(vals):>10{fmt}} "
                  f"{np.min(vals):>10{fmt}} {np.max(vals):>10{fmt}} {np.std(vals):>10{fmt}}")

    # Flag outliers
    print(f"\n  --- FLAGGED CLIPS ---")
    flagged = []
    for r in results:
        issues = []
        if r.get('plosive_delta_db', 0) > 8.0:
            issues.append(f"plosive+{r['plosive_delta_db']:.1f}dB")
        if r.get('clarity_change_%', 0) < -15.0:
            issues.append(f"clarity{r['clarity_change_%']:+.1f}%")
        if r.get('artifact_total', 0) > 5.0:
            issues.append(f"artifact={r['artifact_total']:.4f}")
        if issues:
            flagged.append((r['file'], issues))
    if flagged:
        for fname, issues in sorted(flagged):
            print(f"  ⚠ {fname}: {', '.join(issues)}")
    else:
        print(f"  None — all clips within thresholds")


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
    ("NOISE", [
        ('noise_floor_db', 'Noise floor dB', '.1f'),
        ('snr_db', 'SNR dB', '.1f'),
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


def print_outliers(results):
    if len(results) < 5:
        return

    print(f"\n  --- WORST OUTLIERS ---")

    worst_snr = sorted(results, key=lambda r: r['snr_db'])[:5]
    print(f"\n  Lowest SNR (noisiest):")
    for r in worst_snr:
        print(f"    {r['file']:<20} SNR {r['snr_db']:.1f} dB, noise floor {r['noise_floor_db']:.1f} dB")

    most_harsh = sorted(results, key=lambda r: r['harsh_2_4k'], reverse=True)[:5]
    print(f"\n  Highest harshness (2-4kHz):")
    for r in most_harsh:
        print(f"    {r['file']:<20} {r['harsh_2_4k']*100:.1f}% of energy in 2-4kHz")

    most_sibilant = sorted(results, key=lambda r: r['sib_4_10k'], reverse=True)[:5]
    print(f"\n  Highest sibilance (4-10kHz):")
    for r in most_sibilant:
        print(f"    {r['file']:<20} {r['sib_4_10k']*100:.1f}% of energy in 4-10kHz")

    worst_ovrl = [r for r in results if r['dnsmos_ovrl'] is not None]
    if worst_ovrl:
        worst_ovrl = sorted(worst_ovrl, key=lambda r: r['dnsmos_ovrl'])[:5]
        print(f"\n  Lowest DNSMOS OVRL:")
        for r in worst_ovrl:
            print(f"    {r['file']:<20} OVRL {r['dnsmos_ovrl']:.2f}, SIG {r['dnsmos_sig']:.2f}, BAK {r['dnsmos_bak']:.2f}")

    worst_hnr = sorted(results, key=lambda r: r['hnr_db'])[:5]
    print(f"\n  Lowest HNR (roughest):")
    for r in worst_hnr:
        print(f"    {r['file']:<20} HNR {r['hnr_db']:.1f} dB")


def print_summary(results, label):
    print(f"\n{'='*85}")
    print(f"  {label}  ({len(results)} clips)")
    print(f"{'='*85}")

    for group_name, metrics in METRIC_GROUPS:
        print(f"\n  --- {group_name} ---")
        print(f"  {'Metric':<18} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
        print(f"  {'-'*68}")
        for key, name, fmt in metrics:
            vals = [r[key] for r in results if r[key] is not None and (r[key] != 0.0 or key in ('peak_db', 'rms_db', 'tilt_db_oct', 'lufs', 'noise_floor_db', 'snr_db'))]
            if not vals:
                print(f"  {name:<18} {'n/a':>10}")
                continue
            print(f"  {name:<18} {np.mean(vals):>10{fmt}} {np.median(vals):>10{fmt}} {np.min(vals):>10{fmt}} {np.max(vals):>10{fmt}} {np.std(vals):>10{fmt}}")

    print_outliers(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs='+', default=None, help="WAV file(s) or directory (single-source mode)")
    parser.add_argument("--label", default=None, help="Label for this analysis")
    parser.add_argument("--n", type=int, default=None, help="Sample N clips evenly")
    parser.add_argument("--json", default=None, help="Save raw results to JSON")
    # Compare mode
    parser.add_argument("--compare", action="store_true", help="A/B comparison mode")
    parser.add_argument("--original", default=None, help="Directory of original clips")
    parser.add_argument("--processed", default=None, help="Directory of processed clips (or multiple, comma-separated)")
    parser.add_argument("--match-suffix", default=None, help="Suffix to match in processed dir (e.g. '_3_new')")
    parser.add_argument("--original-suffix", default=None, help="Suffix to match in original dir (e.g. '_1_original')")
    args = parser.parse_args()

    if args.compare:
        if not args.original or not args.processed:
            print("Compare mode requires --original and --processed")
            return

        orig_dir = Path(args.original)
        proc_dirs = [Path(p.strip()) for p in args.processed.split(',')]

        for proc_dir in proc_dirs:
            label = args.label or f"{orig_dir.name} → {proc_dir.name}"

            orig_wavs = sorted(f for f in orig_dir.iterdir() if f.suffix == '.wav')
            proc_wavs = sorted(f for f in proc_dir.iterdir() if f.suffix == '.wav')

            # Build lookup by clip name
            def clip_key(path, suffix=None):
                name = path.stem
                if suffix and name.endswith(suffix):
                    name = name[:-len(suffix)]
                return name

            orig_map = {clip_key(w, args.original_suffix): w for w in orig_wavs}
            proc_map = {clip_key(w, args.match_suffix): w for w in proc_wavs}

            # Find matching pairs
            common = sorted(set(orig_map.keys()) & set(proc_map.keys()))
            if not common:
                # Fallback: match by position if names don't align
                common_by_pos = list(zip(sorted(orig_map.values()), sorted(proc_map.values())))
                if common_by_pos:
                    print(f"No name matches found, falling back to positional matching ({len(common_by_pos)} pairs)")
                    pairs = common_by_pos
                else:
                    print(f"No matching clips between {orig_dir} and {proc_dir}")
                    continue
            else:
                pairs = [(orig_map[k], proc_map[k]) for k in common]

            if args.n and args.n < len(pairs):
                indices = np.linspace(0, len(pairs) - 1, args.n, dtype=int)
                pairs = [pairs[i] for i in indices]

            print(f"\nComparing {len(pairs)} clip pairs: {orig_dir.name} → {proc_dir.name}")

            results = []
            skipped = []
            for i, (orig_path, proc_path) in enumerate(pairs):
                try:
                    r = compare_clips(orig_path, proc_path)
                    if r and r.get('_skipped'):
                        skipped.append(r)
                    elif r:
                        results.append(r)
                except Exception as e:
                    print(f"  SKIP {orig_path.name}: {e}", file=sys.stderr)
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(pairs)}]", flush=True)

            if skipped:
                print(f"\n  Skipped {len(skipped)} short clips (<1.5s): "
                      + ", ".join(r['file'] for r in skipped[:10])
                      + ("..." if len(skipped) > 10 else ""))

            if results:
                print_compare_summary(results, label)

            if args.json:
                json_path = args.json if len(proc_dirs) == 1 else f"{Path(args.json).stem}_{proc_dir.name}.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nRaw results saved to {json_path}")

    else:
        if not args.source:
            print("Single-source mode requires --source")
            return

        sources = [Path(s) for s in args.source]
        wavs = []
        for src in sources:
            if not src.exists():
                print(f"Error: {src} not found")
                return
            if src.is_file() and src.suffix == '.wav':
                wavs.append(src)
            elif src.is_dir():
                wavs.extend(sorted(f for f in src.iterdir() if f.suffix == '.wav'))
            else:
                print(f"Error: {src} is not a .wav file or directory")
                return

        if not wavs:
            print("No WAV files found")
            return

        label = args.label or (sources[0].name if len(sources) == 1 else f"{len(wavs)} files")

        if args.n and args.n < len(wavs):
            indices = np.linspace(0, len(wavs) - 1, args.n, dtype=int)
            wavs = [wavs[i] for i in indices]

        print(f"\nAnalyzing {len(wavs)} clip(s)...")

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
