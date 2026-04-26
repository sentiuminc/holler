#!/usr/bin/env python3
"""Enhance training clips: ClearVoice → DeepFilterNet3 → Recipe E.

Exact pipeline from 2026-04-24 session. Code from memory file.

Usage:
  python enhance_clips.py --start 386 --end 475
"""
import argparse
import os
import shutil
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent / "voices" / "katie" / "training-data" / "audio"
BACKUP_DIR = Path(__file__).parent.parent / "voices" / "katie" / "training-data" / "audio-enhanced-backup"
SR = 24000

# Init models once
print("Loading ClearVoice...")
from clearvoice import ClearVoice
cv = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

print("Loading DeepFilterNet3...")
from df.enhance import enhance as df_enhance, init_df
import torch
df_model, df_state, _ = init_df()

import noisereduce as nr
import tempfile

print("Models loaded.\n")


def enhance_one(audio, sr=24000):
    """Peak normalize → ClearVoice → DeepFilterNet3 → Recipe E."""

    # Step 0: Peak normalize to -1dB only if clipping
    peak = np.max(np.abs(audio))
    if peak >= 0.999:
        audio = audio * (10 ** (-1 / 20) / peak)

    # Step 1: ClearVoice MossFormer2 (spectral masking, 48kHz)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        output = cv(f.name, online_write=False)
        os.unlink(f.name)
    if isinstance(output, dict):
        audio = np.array(list(output.values())[0]).flatten().astype(np.float32)
    else:
        audio = np.array(output).flatten().astype(np.float32)
    # ClearVoice MossFormer2_SE_48K outputs at 48kHz, resample back to 24kHz
    audio = signal.resample_poly(audio, 1, 2).astype(np.float32)

    # Step 2: DeepFilterNet3 (fine-grained denoising)
    audio_48k = signal.resample_poly(audio, 2, 1).astype(np.float32)
    tensor = torch.from_numpy(audio_48k).unsqueeze(0)
    df_out = df_enhance(df_model, df_state, tensor).squeeze().numpy()
    audio = signal.resample_poly(df_out, 1, 2).astype(np.float32)

    # Step 3: Recipe E (high-pass + spectral denoise)
    sos = signal.butter(4, 80, btype='high', fs=sr, output='sos')
    hp = signal.sosfilt(sos, audio).astype(np.float32)
    enhanced = nr.reduce_noise(y=hp, sr=sr, stationary=True, prop_decrease=0.3).astype(np.float32)

    # Step 4: Presence boost (high shelf +~3dB at 3kHz) + peak cap
    b, a = signal.butter(2, 3000 / (sr / 2), btype='high')
    highs = signal.lfilter(b, a, enhanced).astype(np.float32)
    enhanced = (enhanced + highs * 0.3).astype(np.float32)
    peak = np.max(np.abs(enhanced))
    if peak > 0.95:
        enhanced = enhanced * (0.95 / peak)

    return enhanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=386)
    parser.add_argument("--end", type=int, default=475)
    args = parser.parse_args()

    BACKUP_DIR.mkdir(exist_ok=True)

    clips = []
    for i in range(args.start, args.end + 1):
        fname = f"clip_{i:04d}.wav"
        # Restore from backup if it exists (re-run safe)
        backup_path = BACKUP_DIR / fname
        audio_path = AUDIO_DIR / fname
        if backup_path.exists():
            shutil.copy2(backup_path, audio_path)
            clips.append((fname, audio_path))
        elif audio_path.exists():
            shutil.copy2(audio_path, backup_path)
            clips.append((fname, audio_path))

    print(f"Enhancing {len(clips)} clips ({args.start}–{args.end})")
    print(f"Pipeline: ClearVoice → DeepFilterNet3 → Recipe E (exact original)\n")

    for i, (fname, path) in enumerate(clips):
        audio, sr = sf.read(path)
        print(f"[{i+1}/{len(clips)}] {fname} ({len(audio)/sr:.2f}s)...", end=" ", flush=True)

        try:
            enhanced = enhance_one(audio, sr)
            sf.write(str(path), enhanced, sr)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone! Backups at {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
