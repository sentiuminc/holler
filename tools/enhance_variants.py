"""Generate multiple enhancement variants for A/B comparison."""
import numpy as np
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln
from pathlib import Path
import sys

# DeepFilterNet
from df.enhance import enhance, init_df

SOURCE = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load source
data, sr = sf.read(SOURCE)
if len(data.shape) > 1:
    data = data.mean(axis=1)  # mono

# Resample to 48kHz if needed (DeepFilter expects 48k)
TARGET_SR = 48000
if sr != TARGET_SR:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, TARGET_SR)
    data = resample_poly(data, TARGET_SR // g, sr // g)
    sr = TARGET_SR

print(f"Source: {SOURCE.name} | {len(data)/sr:.1f}s | {sr}Hz")

# --- Helper functions ---

def normalize_lufs(audio, sr, target=-18.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if loudness == float('-inf'):
        return audio
    return pyln.normalize.loudness(audio, loudness, target)

def low_shelf(audio, sr, gain_db=3.0, freq=200.0):
    """Low shelf filter boost."""
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    
    cos_w0 = np.cos(w0)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.lfilter(b, a, audio)

def cut_band(audio, sr, low=2000, high=4000, gain_db=-3.0):
    """Parametric EQ cut in a frequency band."""
    center = np.sqrt(low * high)
    Q = center / (high - low)
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * center / sr
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.lfilter(b, a, audio)

def high_shelf(audio, sr, gain_db=2.0, freq=8000.0):
    """High shelf for air/brightness."""
    A = 10**(gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    
    cos_w0 = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.lfilter(b, a, audio)

def peak_limit(audio, ceiling_db=-1.0):
    """Simple peak limiter."""
    ceiling = 10**(ceiling_db / 20.0)
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio

# --- Run DeepFilter once, reuse result ---
print("Running DeepFilterNet3...")
import torch
import torchaudio

# DeepFilter needs specific format
df_model, df_state, _ = init_df()
audio_tensor = torch.from_numpy(data).float().unsqueeze(0)
if sr != df_state.sr():
    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, df_state.sr())
enhanced_tensor = enhance(df_model, df_state, audio_tensor)
if sr != df_state.sr():
    enhanced_tensor = torchaudio.functional.resample(enhanced_tensor, df_state.sr(), sr)
df_clean = enhanced_tensor.squeeze().numpy()
print("DeepFilter done.")

# --- Generate variants ---
stem = SOURCE.stem

variants = {}

# 0. Original (just mono + peak limited)
variants["0_original"] = peak_limit(data)

# 1. DeepFilter only
variants["1_deepfilter_only"] = peak_limit(df_clean)

# 2. DeepFilter + LUFS normalize to -18
v2 = normalize_lufs(df_clean, sr, -18.0)
variants["2_deepfilter_lufs18"] = peak_limit(v2)

# 3. DeepFilter + low shelf +3dB at 200Hz (warmth)
v3 = low_shelf(df_clean, sr, gain_db=3.0, freq=200)
v3 = normalize_lufs(v3, sr, -18.0)
variants["3_deepfilter_lowshelf3db"] = peak_limit(v3)

# 4. DeepFilter + low shelf +5dB at 150Hz (more bass)
v4 = low_shelf(df_clean, sr, gain_db=5.0, freq=150)
v4 = normalize_lufs(v4, sr, -18.0)
variants["4_deepfilter_lowshelf5db"] = peak_limit(v4)

# 5. DeepFilter + harshness cut (-3dB at 2-4kHz)
v5 = cut_band(df_clean, sr, low=2000, high=4000, gain_db=-3.0)
v5 = normalize_lufs(v5, sr, -18.0)
variants["5_deepfilter_harshcut3db"] = peak_limit(v5)

# 6. DeepFilter + low shelf +3dB + harshness cut -3dB (balanced)
v6 = low_shelf(df_clean, sr, gain_db=3.0, freq=200)
v6 = cut_band(v6, sr, low=2000, high=4000, gain_db=-3.0)
v6 = normalize_lufs(v6, sr, -18.0)
variants["6_deepfilter_warm_smooth"] = peak_limit(v6)

# 7. DeepFilter + low shelf +4dB + harshness cut -4dB + air +2dB (radio voice)
v7 = low_shelf(df_clean, sr, gain_db=4.0, freq=200)
v7 = cut_band(v7, sr, low=2000, high=4000, gain_db=-4.0)
v7 = high_shelf(v7, sr, gain_db=2.0, freq=8000)
v7 = normalize_lufs(v7, sr, -18.0)
variants["7_deepfilter_radio"] = peak_limit(v7)

# 8. No DeepFilter — just EQ + LUFS (for comparison)
v8 = low_shelf(data, sr, gain_db=3.0, freq=200)
v8 = cut_band(v8, sr, low=2000, high=4000, gain_db=-3.0)
v8 = normalize_lufs(v8, sr, -18.0)
variants["8_eq_only_no_denoise"] = peak_limit(v8)

# --- Write all variants ---
print(f"\nWriting {len(variants)} variants to {OUT_DIR}/")
for name, audio in variants.items():
    out_path = OUT_DIR / f"{name}_{stem}.wav"
    sf.write(str(out_path), audio, sr)
    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
    print(f"  {name}: peak={peak_db:.1f} rms={rms_db:.1f}")

print("\nDone! Listen and compare.")
