#!/usr/bin/env python3
"""Full training data enhancement pipeline.

DeepFilterNet3 → LUFS normalize → Spectral de-ess → Dynamic presence.

Reads from audio-original/, writes to audio/. Idempotent — always processes from originals.

Usage:
  python enhance_clips.py --voice joe
  python enhance_clips.py --voice joe --gender male
  python enhance_clips.py --voice katie --gender female
  python enhance_clips.py --voice joe --skip-denoise        # skip DeepFilter
  python enhance_clips.py --voice joe --skip-deess          # skip de-esser
  python enhance_clips.py --voice joe --skip-presence       # skip dynamic presence
  python enhance_clips.py --voice joe --start 100 --end 200 # range
"""
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import stft, istft
from pathlib import Path
import torch

VOICES_DIR = Path(__file__).parent.parent / "voices"
SR = 24000

# --- Gender presets ---
PRESETS = {
    'male': {
        'deess_low': 4500, 'deess_high': 7000,
        'deess_threshold': -6.0, 'deess_max_reduction': 8.0,
        'presence_low': 3500, 'presence_high': 8000,
    },
    'female': {
        'deess_low': 6000, 'deess_high': 9000,
        'deess_threshold': -4.0, 'deess_max_reduction': 6.0,
        'presence_low': 3500, 'presence_high': 8000,
    },
}

# --- DeepFilterNet3 (loaded once) ---
print("Loading DeepFilterNet3...")
from df.enhance import enhance as df_enhance, init_df
df_model, df_state, _ = init_df()
print("Model loaded.\n")


# ============================================================
# Stage 1: DeepFilterNet3 single-pass denoise
# ============================================================
def denoise(audio, sr=24000):
    """DeepFilterNet3 single pass. Resamples 24k→48k→24k."""
    audio_48k = signal.resample_poly(audio, 2, 1).astype(np.float32)
    tensor = torch.from_numpy(audio_48k).unsqueeze(0)
    df_out = df_enhance(df_model, df_state, tensor).squeeze().numpy()
    return signal.resample_poly(df_out, 1, 2).astype(np.float32)


# ============================================================
# Stage 2: LUFS normalize
# ============================================================
def compute_lufs(audio, sr):
    """K-weighted integrated loudness (simplified LUFS)."""
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


def lufs_normalize(audio, sr, target_lufs=-22.0, peak_ceiling_db=-3.0):
    """LUFS normalize with peak ceiling. Linear gain only."""
    current_lufs = compute_lufs(audio, sr)
    gain_db = target_lufs - current_lufs
    audio = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(audio))
    ceiling = 10 ** (peak_ceiling_db / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio


# ============================================================
# Stage 3: Spectral de-esser
# ============================================================
def spectral_deess(audio, sr=24000, low_hz=4500, high_hz=7000,
                   threshold_db=-6.0, max_reduction_db=8.0,
                   ratio=4.0, smoothing_frames=3, lookahead_ms=8.0):
    """STFT-based per-bin adaptive sibilance reduction."""
    nperseg = 1024
    noverlap = nperseg * 3 // 4

    lookahead_samples = int(lookahead_ms * 0.001 * sr)
    if lookahead_samples > 0:
        audio_delayed = np.pad(audio, (0, lookahead_samples))[lookahead_samples:]
        audio_detect = audio
    else:
        audio_delayed = audio
        audio_detect = audio

    f, t, Zxx = stft(audio_delayed, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_d = stft(audio_detect, fs=sr, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    det_db = 20 * np.log10(np.abs(Zxx_d) + 1e-10)

    bin_lo = max(1, int(low_hz * nperseg / sr))
    bin_hi = min(det_db.shape[0] - 1, int(high_hz * nperseg / sr))

    gain_db = np.zeros_like(det_db)
    for k in range(bin_lo, bin_hi + 1):
        median = np.median(det_db[k, :])
        excess = np.maximum(det_db[k, :] - median - threshold_db, 0)
        reduction = np.minimum(excess * (1.0 - 1.0 / ratio), max_reduction_db)
        gain_db[k, :] = -reduction

    if smoothing_frames > 1:
        kernel = np.ones(smoothing_frames) / smoothing_frames
        for k in range(bin_lo, bin_hi + 1):
            gain_db[k, :] = np.convolve(gain_db[k, :], kernel, mode='same')

    Zxx_out = magnitude * (10 ** (gain_db / 20)) * np.exp(1j * phase)
    _, out = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return out[:len(audio)].astype(np.float32)


# ============================================================
# Stage 4: Dynamic presence
# ============================================================
def dynamic_presence(audio, sr=24000, low_hz=3500, high_hz=8000,
                     max_boost_db=2.5, sensitivity=1.0, smoothing_frames=5):
    """Proportional dynamic presence: boosts bins below their median
    smoothly and continuously. Never pushes already-bright moments."""
    nperseg = 1024
    noverlap = nperseg * 3 // 4

    f, t, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    mag_db = 20 * np.log10(magnitude + 1e-10)

    bin_lo = max(1, int(low_hz * nperseg / sr))
    bin_hi = min(mag_db.shape[0] - 1, int(high_hz * nperseg / sr))

    gain_db = np.zeros_like(mag_db)
    for k in range(bin_lo, bin_hi + 1):
        median = np.median(mag_db[k, :])
        deficit = median - mag_db[k, :]
        boost = np.where(deficit > 0, deficit * 0.5 * sensitivity, 0.0)
        gain_db[k, :] = max_boost_db * np.tanh(boost / max_boost_db)

    if smoothing_frames > 1:
        kernel = np.ones(smoothing_frames) / smoothing_frames
        for k in range(bin_lo, bin_hi + 1):
            gain_db[k, :] = np.convolve(gain_db[k, :], kernel, mode='same')

    Zxx_out = magnitude * (10 ** (gain_db / 20)) * np.exp(1j * phase)
    _, out = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap)
    out = out[:len(audio)].astype(np.float32)

    peak = np.max(np.abs(out))
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out


# ============================================================
# Full pipeline
# ============================================================
def enhance_full(audio, sr=24000, gender='male',
                 skip_denoise=False, skip_deess=False, skip_presence=False,
                 target_lufs=-22.0, peak_ceiling=-3.0):
    """Complete enhancement: denoise → LUFS → de-ess → presence."""
    preset = PRESETS[gender]

    if not skip_denoise:
        audio = denoise(audio, sr)

    audio = lufs_normalize(audio, sr, target_lufs, peak_ceiling)

    if not skip_deess:
        audio = spectral_deess(audio, sr,
                               low_hz=preset['deess_low'],
                               high_hz=preset['deess_high'],
                               threshold_db=preset['deess_threshold'],
                               max_reduction_db=preset['deess_max_reduction'])

    if not skip_presence:
        audio = dynamic_presence(audio, sr,
                                 low_hz=preset['presence_low'],
                                 high_hz=preset['presence_high'])

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True, help="Voice name (directory under voices/)")
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--skip-denoise", action="store_true")
    parser.add_argument("--skip-deess", action="store_true")
    parser.add_argument("--skip-presence", action="store_true")
    parser.add_argument("--target-lufs", type=float, default=-22.0)
    parser.add_argument("--peak-ceiling", type=float, default=-3.0)
    args = parser.parse_args()

    voice_dir = VOICES_DIR / args.voice / "training-data"
    src_dir = voice_dir / "audio-original"
    out_dir = voice_dir / "audio"

    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        return

    out_dir.mkdir(exist_ok=True)

    wavs = sorted(f for f in src_dir.iterdir() if f.suffix == '.wav')
    if args.start is not None or args.end is not None:
        start = args.start or 1
        end = args.end or len(wavs)
        wavs = [w for w in wavs if start <= int(w.stem.split('_')[-1]) <= end]

    stages = []
    if not args.skip_denoise: stages.append("DeepFilter")
    stages.append(f"LUFS {args.target_lufs}")
    if not args.skip_deess: stages.append(f"de-ess ({args.gender})")
    if not args.skip_presence: stages.append("presence")

    print(f"Voice: {args.voice} ({args.gender})")
    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Pipeline: {' → '.join(stages)}")
    print(f"Clips: {len(wavs)}\n")

    failed = []
    for i, src_path in enumerate(wavs):
        audio, sr = sf.read(src_path)
        print(f"[{i+1}/{len(wavs)}] {src_path.name} ({len(audio)/sr:.2f}s)...", end=" ", flush=True)
        try:
            enhanced = enhance_full(audio, sr, gender=args.gender,
                                    skip_denoise=args.skip_denoise,
                                    skip_deess=args.skip_deess,
                                    skip_presence=args.skip_presence,
                                    target_lufs=args.target_lufs,
                                    peak_ceiling=args.peak_ceiling)
            sf.write(str(out_dir / src_path.name), enhanced, sr)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(src_path.name)

    print(f"\nDone! {len(wavs) - len(failed)}/{len(wavs)} processed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
