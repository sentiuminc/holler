#!/usr/bin/env python3
"""Generate waveform images with confirmed pop regions highlighted."""
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SR = 24000
SRC = Path.home() / "Downloads/nora-g32-10para"
OUT = Path.home() / "Downloads/pop-visualizations"
OUT.mkdir(exist_ok=True)

# User-confirmed pops per file (time in seconds, description)
CONFIRMED = {
    "para_00.wav": [(29.0, "slight pop")],
    "para_01.wav": [(0.5, "pop"), (14.0, "slight"), (18.0, "slight pop"), (24.5, "slight crackle"), (29.0, "crackle")],
    "para_02.wav": [(3.5, "crackle"), (6.5, "crackle"), (9.0, "crackle"), (16.0, "crackle"), (22.0, "crackle"), (27.0, "crackle"), (30.0, "cracklepop"), (33.5, "crackle")],
    "para_03.wav": [(0.3, "clip"), (5.0, "pop crackle"), (12.0, "pop"), (15.0, "pop"), (18.0, "crackle"), (24.5, "crackle top voice")],
}


def load(fname):
    audio, sr = sf.read(str(SRC / fname))
    assert sr == SR
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def compute_rms_envelope(audio, window_ms=20):
    window = int(SR * window_ms / 1000)
    sq = audio ** 2
    cumsum = np.cumsum(sq)
    cumsum = np.insert(cumsum, 0, 0)
    rms = np.zeros(len(audio))
    half = window // 2
    for i in range(0, len(audio), half):
        lo = max(0, i - half)
        hi = min(len(audio), i + half)
        val = np.sqrt((cumsum[hi] - cumsum[lo]) / (hi - lo))
        for j in range(max(0, i - half), min(len(audio), i + half)):
            rms[j] = val
    return rms


def plot_full_waveform(fname, pops, audio):
    """Full file waveform with pop regions highlighted in red."""
    duration = len(audio) / SR
    t = np.arange(len(audio)) / SR

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f'{fname} — Nora g32 (confirmed pops in red)', fontsize=14, fontweight='bold')

    # Top: full waveform
    ax = axes[0]
    ax.plot(t, audio, color='#2196F3', linewidth=0.3, alpha=0.7)
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-0.8, 0.8)
    ax.axhline(y=0.7, color='orange', linewidth=0.5, linestyle='--', alpha=0.5, label='soft clip ceiling')
    ax.axhline(y=-0.7, color='orange', linewidth=0.5, linestyle='--', alpha=0.5)

    for pop_t, desc in pops:
        ax.axvspan(pop_t - 0.3, pop_t + 0.3, color='red', alpha=0.2)
        ax.annotate(desc, xy=(pop_t, 0.75), fontsize=7, color='red',
                   ha='center', va='bottom', rotation=45)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Waveform')

    # Middle: RMS envelope
    ax = axes[1]
    rms = compute_rms_envelope(audio, window_ms=50)
    ax.plot(t, rms, color='#4CAF50', linewidth=0.8)
    ax.set_ylabel('RMS')
    ax.set_ylim(0, 0.35)

    for pop_t, desc in pops:
        ax.axvspan(pop_t - 0.3, pop_t + 0.3, color='red', alpha=0.2)

    ax.set_title('RMS Envelope (50ms window)')

    # Bottom: sample-to-sample delta
    ax = axes[2]
    deltas = np.abs(np.diff(audio.astype(np.float64)))
    # Downsample for plotting (max in windows)
    ds = 100
    n_bins = len(deltas) // ds
    delta_ds = np.array([np.max(deltas[i*ds:(i+1)*ds]) for i in range(n_bins)])
    t_ds = np.arange(n_bins) * ds / SR
    ax.plot(t_ds, delta_ds, color='#FF9800', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Max |Δ|')
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.38, color='purple', linewidth=0.5, linestyle='--', alpha=0.5, label='proposed threshold (0.38)')

    for pop_t, desc in pops:
        ax.axvspan(pop_t - 0.3, pop_t + 0.3, color='red', alpha=0.2)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Max Sample Delta (per 100-sample window)')
    ax.set_xlabel('Time (seconds)')

    plt.tight_layout()
    out_path = OUT / f"{fname.replace('.wav', '')}_full.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_pop_closeups(fname, pops, audio):
    """Zoomed-in views of each confirmed pop."""
    n_pops = len(pops)
    if n_pops == 0:
        return

    cols = min(3, n_pops)
    rows = (n_pops + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_pops == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(f'{fname} — Pop Close-ups (±10ms)', fontsize=13, fontweight='bold')

    for idx, (pop_t, desc) in enumerate(pops):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        center = int(pop_t * SR)
        window = int(SR * 0.010)  # 10ms each side
        start = max(0, center - window)
        end = min(len(audio), center + window)
        segment = audio[start:end]
        t_local = (np.arange(len(segment)) - (center - start)) / SR * 1000  # ms relative to pop

        ax.plot(t_local, segment, color='#2196F3', linewidth=0.8)
        ax.axvline(x=0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvspan(-2, 2, color='red', alpha=0.08)
        ax.set_title(f'{pop_t}s: {desc}', fontsize=10)
        ax.set_xlabel('ms from pop center')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-0.8, 0.8)
        ax.grid(True, alpha=0.2)

        # Annotate peak and RMS
        peak = np.max(np.abs(segment))
        rms = np.sqrt(np.mean(segment ** 2))
        ax.text(0.02, 0.98, f'peak={peak:.2f} rms={rms:.3f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for idx in range(n_pops, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    out_path = OUT / f"{fname.replace('.wav', '')}_closeups.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(audio_pop, pop_t, audio_clean, clean_t, fname):
    """Side-by-side: pop region vs clean region from same file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'{fname} — Pop vs Clean Comparison (±10ms)', fontsize=13, fontweight='bold')

    for ax, audio, t, label, color in [
        (axes[0], audio_pop, pop_t, 'POP region', '#F44336'),
        (axes[1], audio_clean, clean_t, 'CLEAN region', '#4CAF50'),
    ]:
        center = int(t * SR)
        window = int(SR * 0.010)
        start = max(0, center - window)
        end = min(len(audio), center + window)
        segment = audio[start:end]
        t_local = (np.arange(len(segment)) - (center - start)) / SR * 1000

        ax.plot(t_local, segment, color=color, linewidth=0.8)
        ax.set_title(f'{label} at {t}s', fontsize=11)
        ax.set_xlabel('ms')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-0.8, 0.8)
        ax.grid(True, alpha=0.2)

        peak = np.max(np.abs(segment))
        rms = np.sqrt(np.mean(segment ** 2))
        ax.text(0.02, 0.98, f'peak={peak:.2f} rms={rms:.3f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out_path = OUT / "pop_vs_clean_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    # Generate full waveform plots for each file with pops
    for fname, pops in CONFIRMED.items():
        audio = load(fname)
        plot_full_waveform(fname, pops, audio)
        plot_pop_closeups(fname, pops, audio)

    # Comparison: para_02 at 22s (worst crackle) vs para_08 at 15s (known clean)
    audio_02 = load("para_02.wav")
    audio_08 = load("para_08.wav")
    plot_comparison(audio_02, 22.0, audio_08, 15.0, "para_02 vs para_08")

    import subprocess
    subprocess.run(["open", str(OUT)])
    print(f"\nAll images saved to {OUT}")


if __name__ == '__main__':
    main()
