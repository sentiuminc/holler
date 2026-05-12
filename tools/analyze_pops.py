#!/usr/bin/env python3
"""Deep analysis of pop/crack artifacts in TTS audio.

Characterizes what pops actually ARE at the sample level:
- Sample-to-sample delta (slew rate) distribution
- Localized energy bursts (short windows of anomalous energy)
- Phase continuity analysis
- Plosive consonant correlation
- Per-chunk boundary analysis (streaming chunk boundaries)

Usage:
  python tools/analyze_pops.py ~/Downloads/nora-g32-10para/para_01.wav
  python tools/analyze_pops.py ~/Downloads/nora-g32-10para/ --all
  python tools/analyze_pops.py ~/Downloads/nora-g32-10para/ --all --voice nora --csv
"""
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
import json

SR = 24000


def load_audio(path):
    audio, sr = sf.read(str(path))
    if sr != SR:
        raise ValueError(f"Expected {SR}Hz, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def compute_deltas(audio):
    """Sample-to-sample differences (slew rate)."""
    return np.diff(audio)


def find_pop_candidates(audio, window_ms=0.5, hop_ms=0.1, z_threshold=4.0):
    """Find short bursts of anomalous energy using z-score on windowed energy.

    A pop is a very short (< 1ms) burst of energy that's way above the local average.
    We compute energy in tiny windows and flag outliers.

    Returns list of dicts with: sample index, magnitude, local_rms, z_score, duration_samples
    """
    window_samples = max(1, int(SR * window_ms / 1000))
    hop_samples = max(1, int(SR * hop_ms / 1000))

    # Compute energy in small windows
    n_windows = (len(audio) - window_samples) // hop_samples + 1
    energies = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples
        energies[i] = np.sqrt(np.mean(audio[start:end] ** 2))

    # Use a larger context window for local statistics (50ms)
    context_windows = int(50 / hop_ms)

    pops = []
    for i in range(n_windows):
        ctx_start = max(0, i - context_windows)
        ctx_end = min(n_windows, i + context_windows)
        ctx = energies[ctx_start:ctx_end]

        local_mean = np.mean(ctx)
        local_std = np.std(ctx)

        if local_std < 1e-6:
            continue

        z = (energies[i] - local_mean) / local_std
        if z > z_threshold:
            sample_idx = i * hop_samples
            pops.append({
                'sample': int(sample_idx),
                'time_ms': sample_idx / SR * 1000,
                'window_rms': float(energies[i]),
                'local_rms': float(local_mean),
                'z_score': float(z),
                'ratio': float(energies[i] / local_mean) if local_mean > 0 else 0,
            })

    return pops


def find_slew_rate_spikes(audio, percentile=99.9):
    """Find samples where the rate of change is extreme.

    Returns the threshold at the given percentile and all spikes above it.
    """
    deltas = np.abs(compute_deltas(audio))
    threshold = np.percentile(deltas, percentile)

    spike_indices = np.where(deltas > threshold)[0]

    spikes = []
    for idx in spike_indices:
        spikes.append({
            'sample': int(idx),
            'time_ms': idx / SR * 1000,
            'delta': float(deltas[idx]),
            'value_before': float(audio[idx]),
            'value_after': float(audio[idx + 1]),
        })

    return threshold, spikes


def analyze_delta_distribution(audio):
    """Full statistics on sample-to-sample deltas."""
    deltas = np.abs(compute_deltas(audio))
    return {
        'mean': float(np.mean(deltas)),
        'std': float(np.std(deltas)),
        'median': float(np.median(deltas)),
        'p95': float(np.percentile(deltas, 95)),
        'p99': float(np.percentile(deltas, 99)),
        'p999': float(np.percentile(deltas, 99.9)),
        'p9999': float(np.percentile(deltas, 99.99)),
        'max': float(np.max(deltas)),
        'max_sample': int(np.argmax(deltas)),
    }


def find_chunk_boundaries(audio, chunk_samples=None):
    """Check if pops correlate with streaming chunk boundaries.

    Default chunk size: 3 codec tokens * 2000 samples/token = 6000 samples at 24kHz.
    (Qwen3-TTS 12Hz codec: each token = 1/12 sec = 2000 samples)
    """
    if chunk_samples is None:
        chunk_samples = 3 * 2000  # 3 tokens * 2000 samples/token

    boundaries = list(range(chunk_samples, len(audio), chunk_samples))

    results = []
    margin = 50  # samples around boundary to check
    for b in boundaries:
        start = max(0, b - margin)
        end = min(len(audio), b + margin)
        segment = audio[start:end]

        if len(segment) < 2:
            continue

        deltas = np.abs(np.diff(segment))
        max_delta = np.max(deltas)
        max_idx = start + np.argmax(deltas)

        # Also check the exact boundary
        if b < len(audio) - 1:
            boundary_delta = abs(audio[b] - audio[b-1])
        else:
            boundary_delta = 0

        results.append({
            'boundary_sample': b,
            'time_ms': b / SR * 1000,
            'max_delta_near_boundary': float(max_delta),
            'exact_boundary_delta': float(boundary_delta),
            'max_delta_sample': int(max_idx),
        })

    return results


def characterize_pop_waveform(audio, pop_sample, context_ms=2.0):
    """Extract the exact waveform around a pop for visual inspection."""
    ctx_samples = int(SR * context_ms / 1000)
    start = max(0, pop_sample - ctx_samples)
    end = min(len(audio), pop_sample + ctx_samples)

    segment = audio[start:end]
    deltas = np.diff(segment)

    # Find the sharpest transition in this region
    abs_deltas = np.abs(deltas)
    peak_idx = np.argmax(abs_deltas)

    # Is this a single-sample spike or a multi-sample burst?
    # Check how many consecutive samples have high deltas (> 50% of peak)
    peak_val = abs_deltas[peak_idx]
    high_delta_mask = abs_deltas > (peak_val * 0.5)

    # Find contiguous run around peak
    burst_start = peak_idx
    while burst_start > 0 and high_delta_mask[burst_start - 1]:
        burst_start -= 1
    burst_end = peak_idx
    while burst_end < len(high_delta_mask) - 1 and high_delta_mask[burst_end + 1]:
        burst_end += 1

    burst_duration = burst_end - burst_start + 1

    return {
        'peak_delta': float(peak_val),
        'burst_duration_samples': int(burst_duration),
        'burst_duration_us': burst_duration / SR * 1e6,
        'is_single_sample': burst_duration <= 2,
        'is_short_burst': 2 < burst_duration <= 10,
        'is_sustained': burst_duration > 10,
        'waveform_before': [float(x) for x in audio[max(0, start + peak_idx - 5):start + peak_idx]],
        'waveform_at': float(audio[start + peak_idx]) if start + peak_idx < len(audio) else 0,
        'waveform_after': [float(x) for x in audio[start + peak_idx + 1:min(len(audio), start + peak_idx + 6)]],
    }


def analyze_file(path, verbose=True):
    """Full pop analysis on a single file."""
    audio = load_audio(path)
    duration_s = len(audio) / SR

    result = {
        'file': str(path.name),
        'duration_s': round(duration_s, 2),
        'peak': float(np.max(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio ** 2))),
    }

    # 1. Delta distribution
    delta_stats = analyze_delta_distribution(audio)
    result['delta_stats'] = delta_stats

    # 2. Find pop candidates (energy bursts)
    pops_z4 = find_pop_candidates(audio, z_threshold=4.0)
    pops_z6 = find_pop_candidates(audio, z_threshold=6.0)
    result['pop_count_z4'] = len(pops_z4)
    result['pop_count_z6'] = len(pops_z6)
    result['pops_per_sec_z4'] = round(len(pops_z4) / duration_s, 2) if duration_s > 0 else 0
    result['pops_per_sec_z6'] = round(len(pops_z6) / duration_s, 2) if duration_s > 0 else 0

    # 3. Slew rate spikes
    slew_threshold, slew_spikes = find_slew_rate_spikes(audio, percentile=99.95)
    result['slew_threshold_99_95'] = round(slew_threshold, 4)
    result['slew_spike_count'] = len(slew_spikes)

    # 4. Chunk boundary correlation
    boundaries = find_chunk_boundaries(audio)
    if boundaries:
        boundary_deltas = [b['exact_boundary_delta'] for b in boundaries]
        all_deltas = np.abs(compute_deltas(audio))
        avg_delta = np.mean(all_deltas)
        avg_boundary_delta = np.mean(boundary_deltas)
        result['boundary_delta_ratio'] = round(avg_boundary_delta / avg_delta, 2) if avg_delta > 0 else 0
        result['max_boundary_delta'] = round(max(boundary_deltas), 4)

    # 5. Characterize the top 5 worst pops
    worst_pops = []
    if pops_z4:
        sorted_pops = sorted(pops_z4, key=lambda p: p['z_score'], reverse=True)[:5]
        for pop in sorted_pops:
            char = characterize_pop_waveform(audio, pop['sample'])
            worst_pops.append({
                'time_ms': round(pop['time_ms'], 1),
                'z_score': round(pop['z_score'], 2),
                'ratio': round(pop['ratio'], 2),
                **char,
            })
    result['worst_pops'] = worst_pops

    if verbose:
        print(f"\n{'='*60}")
        print(f"File: {path.name} ({duration_s:.1f}s, peak={result['peak']:.3f}, rms={result['rms']:.4f})")
        print(f"{'='*60}")

        print(f"\nDelta distribution (sample-to-sample changes):")
        print(f"  mean={delta_stats['mean']:.4f}  median={delta_stats['median']:.4f}")
        print(f"  p95={delta_stats['p95']:.4f}  p99={delta_stats['p99']:.4f}")
        print(f"  p99.9={delta_stats['p999']:.4f}  p99.99={delta_stats['p9999']:.4f}  max={delta_stats['max']:.4f}")

        print(f"\nPop candidates (energy burst z-score):")
        print(f"  z>4: {len(pops_z4)} ({result['pops_per_sec_z4']}/sec)")
        print(f"  z>6: {len(pops_z6)} ({result['pops_per_sec_z6']}/sec)")

        print(f"\nSlew rate spikes (top 0.05% deltas > {slew_threshold:.4f}):")
        print(f"  count: {len(slew_spikes)}")

        if boundaries:
            print(f"\nChunk boundary analysis:")
            print(f"  boundary delta / avg delta ratio: {result['boundary_delta_ratio']}x")
            print(f"  max boundary delta: {result['max_boundary_delta']:.4f}")

        if worst_pops:
            print(f"\nTop {len(worst_pops)} worst pops:")
            for i, p in enumerate(worst_pops):
                pop_type = "SINGLE-SAMPLE" if p['is_single_sample'] else ("SHORT-BURST" if p['is_short_burst'] else "SUSTAINED")
                print(f"  #{i+1} at {p['time_ms']:.1f}ms: z={p['z_score']:.1f}, ratio={p['ratio']:.1f}x, "
                      f"peak_delta={p['peak_delta']:.4f}, burst={p['burst_duration_samples']}samp ({p['burst_duration_us']:.0f}µs) [{pop_type}]")
                print(f"       before: {[round(x,3) for x in p['waveform_before']]} → {round(p['waveform_at'],3)} → after: {[round(x,3) for x in p['waveform_after']]}")

    return result


def compare_files(results):
    """Cross-file comparison."""
    print(f"\n{'='*80}")
    print(f"CROSS-FILE COMPARISON ({len(results)} files)")
    print(f"{'='*80}")

    print(f"\n{'File':<15} {'Dur':>5} {'Peak':>6} {'RMS':>6} {'Pops/s':>7} {'MaxΔ':>7} {'p99.9Δ':>7} {'BndRatio':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['file']:<15} {r['duration_s']:>5.1f} {r['peak']:>6.3f} {r['rms']:>6.4f} "
              f"{r['pops_per_sec_z4']:>7.2f} {r['delta_stats']['max']:>7.4f} "
              f"{r['delta_stats']['p999']:>7.4f} {r.get('boundary_delta_ratio', 0):>8.2f}")

    # Aggregate pop characteristics
    all_pops = []
    for r in results:
        all_pops.extend(r['worst_pops'])

    if all_pops:
        single = sum(1 for p in all_pops if p['is_single_sample'])
        short = sum(1 for p in all_pops if p['is_short_burst'])
        sustained = sum(1 for p in all_pops if p['is_sustained'])
        print(f"\nPop type breakdown (top-5 per file, {len(all_pops)} total):")
        print(f"  Single-sample (≤2): {single} ({100*single/len(all_pops):.0f}%)")
        print(f"  Short-burst (3-10): {short} ({100*short/len(all_pops):.0f}%)")
        print(f"  Sustained (>10):    {sustained} ({100*sustained/len(all_pops):.0f}%)")

        peak_deltas = [p['peak_delta'] for p in all_pops]
        print(f"\nPop peak delta range: {min(peak_deltas):.4f} - {max(peak_deltas):.4f}")
        print(f"Pop peak delta mean: {np.mean(peak_deltas):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Deep pop/crack analysis for TTS audio')
    parser.add_argument('path', help='WAV file or directory')
    parser.add_argument('--all', action='store_true', help='Analyze all WAV files in directory')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--voice', default='', help='Voice name (for labeling)')
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        result = analyze_file(path, verbose=not args.json)
        if args.json:
            print(json.dumps(result, indent=2))
    elif path.is_dir() and args.all:
        files = sorted(path.glob('*.wav'))
        if not files:
            print(f"No WAV files in {path}")
            sys.exit(1)

        results = []
        for f in files:
            r = analyze_file(f, verbose=not args.json)
            results.append(r)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            compare_files(results)
    else:
        print(f"Pass a WAV file or directory with --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
