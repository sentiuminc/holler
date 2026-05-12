#!/usr/bin/env python3
"""Extract short clips around detected pops for manual verification.

Produces 3-second WAV clips centered on each pop, plus a report.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

SR = 24000
CLIP_DURATION = 3.0  # seconds of context around each pop
CLIP_SAMPLES = int(SR * CLIP_DURATION)


def load_audio(path):
    audio, sr = sf.read(str(path))
    if sr != SR:
        raise ValueError(f"Expected {SR}Hz, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def rms_windowed(audio, window_samples=2400):
    n = len(audio)
    sq = audio ** 2
    cumsum = np.cumsum(sq)
    cumsum = np.insert(cumsum, 0, 0)
    half = window_samples // 2
    rms = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        rms[i] = np.sqrt((cumsum[hi] - cumsum[lo]) / (hi - lo))
    return rms


def find_audible_pops(audio, delta_threshold=0.30, min_rms=0.01):
    """Find pops ranked by audibility (delta/local_rms ratio)."""
    deltas = np.abs(np.diff(audio.astype(np.float64)))
    local_rms = rms_windowed(audio, 2400)

    events = []
    last_sample = -2400  # min 100ms between events

    for i in np.where(deltas > delta_threshold)[0]:
        if local_rms[i] < min_rms:
            continue
        if i - last_sample < 2400:
            continue
        last_sample = i

        # Classify
        pre_mean = np.mean(audio[max(0, i-10):i])
        post_mean = np.mean(audio[i+1:min(len(audio), i+11)])
        is_spike = abs(audio[i+1] - pre_mean) > abs(post_mean - pre_mean) * 1.5 if abs(post_mean - pre_mean) > 0.001 else False

        # Burst length
        burst_start = i
        while burst_start > 0 and deltas[burst_start - 1] > delta_threshold * 0.5:
            burst_start -= 1
        burst_end = i
        while burst_end < len(deltas) - 1 and deltas[burst_end + 1] > delta_threshold * 0.5:
            burst_end += 1

        events.append({
            'sample': int(i),
            'time_s': i / SR,
            'delta': float(deltas[i]),
            'local_rms': float(local_rms[i]),
            'audibility': float(deltas[i] / local_rms[i]) if local_rms[i] > 0 else 0,
            'type': 'spike' if is_spike else 'step',
            'burst_samples': int(burst_end - burst_start + 1),
            'before_5': [round(float(audio[max(0, i-4+j)]), 3) for j in range(5)],
            'after_5': [round(float(audio[min(len(audio)-1, i+1+j)]), 3) for j in range(5)],
        })

    return events


def main():
    src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/nora-g32-10para"
    out_dir = Path.home() / "Downloads/pop-verification"
    out_dir.mkdir(exist_ok=True)

    files = sorted(src_dir.glob("*.wav"))
    if not files:
        print(f"No WAV files in {src_dir}")
        return

    all_pops = []
    for f in files:
        audio = load_audio(f)
        pops = find_audible_pops(audio, delta_threshold=0.30)
        for p in pops:
            p['file'] = f.name
            p['audio'] = audio
            p['full_path'] = f
        all_pops.extend(pops)

    # Sort by audibility (how much it stands out from surroundings)
    all_pops.sort(key=lambda p: p['audibility'], reverse=True)

    # Pick top 10, preferring spikes (they sound more like clicks)
    # and spreading across files
    selected = []
    files_used = {}
    # First pass: pick spikes with high audibility, max 2 per file
    for p in all_pops:
        if len(selected) >= 10:
            break
        if p['type'] == 'spike' and files_used.get(p['file'], 0) < 2:
            selected.append(p)
            files_used[p['file']] = files_used.get(p['file'], 0) + 1

    # Second pass: fill remaining with steps if needed
    if len(selected) < 10:
        for p in all_pops:
            if len(selected) >= 10:
                break
            if p not in selected and files_used.get(p['file'], 0) < 3:
                selected.append(p)
                files_used[p['file']] = files_used.get(p['file'], 0) + 1

    # Sort by file then time for presentation
    selected.sort(key=lambda p: (p['file'], p['time_s']))

    print(f"Extracted {len(selected)} pops to {out_dir}/\n")
    print(f"{'#':<3} {'File':<12} {'Time':>7} {'Type':<6} {'Delta':>6} {'Audibility':>10} {'Burst':>5}  Description")
    print("-" * 90)

    for i, p in enumerate(selected):
        audio = p['audio']
        center = p['sample']
        half = CLIP_SAMPLES // 2
        start = max(0, center - half)
        end = min(len(audio), center + half)
        clip = audio[start:end]

        # The pop is at this offset within the clip
        pop_offset_s = (center - start) / SR

        out_name = f"pop_{i+1:02d}_{p['file'].replace('.wav', '')}_{p['time_s']:.1f}s.wav"
        sf.write(str(out_dir / out_name), clip, SR)

        # Describe what to listen for
        if p['type'] == 'spike':
            desc = f"Quick click/crackle ({p['burst_samples']} samples = {p['burst_samples']/SR*1000:.1f}ms)"
        else:
            desc = f"Pop/thud from sharp waveform jump (Δ={p['delta']:.2f})"

        print(f"{i+1:<3} {p['file']:<12} {p['time_s']:>6.2f}s {p['type']:<6} {p['delta']:>6.3f} "
              f"{p['audibility']:>10.1f}x  {desc}")
        print(f"    Pop is at ~{pop_offset_s:.1f}s in the clip. "
              f"Waveform: {p['before_5']} → {p['after_5']}")

    # Open folder
    import subprocess
    subprocess.run(["open", str(out_dir)])


if __name__ == '__main__':
    main()
