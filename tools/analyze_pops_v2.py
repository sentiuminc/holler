#!/usr/bin/env python3
"""Deep analysis of pop/crack artifacts in TTS audio — v2.

Focuses on what actually matters: large sample-to-sample jumps in SPEECH regions
(not silence). Characterizes each pop's waveform shape to determine what
kind of artifact it is and what fix would work.

Usage:
  python tools/analyze_pops_v2.py ~/Downloads/nora-g32-10para/ --all
  python tools/analyze_pops_v2.py file.wav --delta-threshold 0.15
"""
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

SR = 24000
CHUNK_SAMPLES = 3 * 2000  # 3 codec tokens * 2000 samples/token


def load_audio(path):
    audio, sr = sf.read(str(path))
    if sr != SR:
        raise ValueError(f"Expected {SR}Hz, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def rms_windowed(audio, window_samples=480):
    """Compute RMS in sliding windows (default 20ms)."""
    n = len(audio)
    rms = np.zeros(n)
    # Cumulative sum for efficient windowed RMS
    sq = audio ** 2
    cumsum = np.cumsum(sq)
    cumsum = np.insert(cumsum, 0, 0)
    half = window_samples // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        rms[i] = np.sqrt((cumsum[hi] - cumsum[lo]) / (hi - lo))
    return rms


def find_transients(audio, delta_threshold=0.15, min_local_rms=0.01):
    """Find sample-to-sample jumps that exceed threshold, only in speech regions.

    Returns list of transient events with full context.
    """
    deltas = np.abs(np.diff(audio.astype(np.float64)))
    local_rms = rms_windowed(audio, window_samples=2400)  # 100ms window

    events = []
    # Skip events that are within 50 samples of a previous event (dedup bursts)
    last_event_sample = -100

    for i in np.where(deltas > delta_threshold)[0]:
        if local_rms[i] < min_local_rms:
            continue  # skip silence
        if i - last_event_sample < 50:
            continue  # dedup

        last_event_sample = i

        # Get context: 20 samples before and after
        ctx_start = max(0, i - 20)
        ctx_end = min(len(audio), i + 21)
        context = audio[ctx_start:ctx_end]

        # Measure the burst: how many consecutive high-delta samples?
        burst_start = i
        while burst_start > 0 and deltas[burst_start - 1] > delta_threshold * 0.5:
            burst_start -= 1
        burst_end = i
        while burst_end < len(deltas) - 1 and deltas[burst_end + 1] > delta_threshold * 0.5:
            burst_end += 1
        burst_len = burst_end - burst_start + 1

        # Is this at a chunk boundary?
        nearest_boundary = round(i / CHUNK_SAMPLES) * CHUNK_SAMPLES
        dist_to_boundary = abs(i - nearest_boundary)

        # What does the waveform look like?
        # Check if it's a spike (goes up then right back down) or a step (stays at new level)
        pre_mean = np.mean(audio[max(0, i-10):i]) if i >= 1 else 0
        post_mean = np.mean(audio[i+1:min(len(audio), i+11)])
        immediate_delta = audio[i+1] - audio[i]
        recovery_delta = post_mean - audio[i+1]

        # Spike: large delta followed by return toward pre_mean
        # Step: large delta and stays at new level
        is_spike = abs(audio[i+1] - pre_mean) > abs(post_mean - pre_mean) * 1.5 if abs(post_mean - pre_mean) > 0.001 else False

        events.append({
            'sample': int(i),
            'time_s': i / SR,
            'delta': float(deltas[i]),
            'value_before': float(audio[i]),
            'value_after': float(audio[i + 1]),
            'local_rms': float(local_rms[i]),
            'delta_over_rms': float(deltas[i] / local_rms[i]) if local_rms[i] > 0 else 0,
            'burst_samples': int(burst_len),
            'burst_us': burst_len / SR * 1e6,
            'dist_to_chunk_boundary': int(dist_to_boundary),
            'at_chunk_boundary': dist_to_boundary < 10,
            'waveform_type': 'spike' if is_spike else 'step',
            'pre_mean': float(pre_mean),
            'post_mean': float(post_mean),
            'context_20': [round(float(x), 4) for x in context],
        })

    return events


def analyze_file(path, delta_threshold=0.15, verbose=True):
    audio = load_audio(path)
    duration_s = len(audio) / SR

    # Global stats
    deltas = np.abs(np.diff(audio.astype(np.float64)))
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Delta percentiles
    pcts = {}
    for p in [95, 99, 99.5, 99.9, 99.95, 99.99]:
        pcts[f'p{p}'] = float(np.percentile(deltas, p))

    # Find transients at multiple thresholds
    transients_015 = find_transients(audio, delta_threshold=0.15)
    transients_025 = find_transients(audio, delta_threshold=0.25)
    transients_050 = find_transients(audio, delta_threshold=0.50)

    # Chunk boundary correlation
    boundary_events = [e for e in transients_015 if e['at_chunk_boundary']]
    non_boundary = [e for e in transients_015 if not e['at_chunk_boundary']]

    result = {
        'file': path.name,
        'duration_s': round(duration_s, 2),
        'peak': round(peak, 3),
        'rms': round(rms, 4),
        'delta_percentiles': pcts,
        'max_delta': round(float(np.max(deltas)), 4),
        'max_delta_time_s': round(int(np.argmax(deltas)) / SR, 3),
        'transients_015': len(transients_015),
        'transients_025': len(transients_025),
        'transients_050': len(transients_050),
        'per_sec_015': round(len(transients_015) / duration_s, 2),
        'per_sec_025': round(len(transients_025) / duration_s, 2),
        'per_sec_050': round(len(transients_050) / duration_s, 2),
        'at_chunk_boundary': len(boundary_events),
        'not_at_boundary': len(non_boundary),
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"{path.name} — {duration_s:.1f}s, peak={peak:.3f}, rms={rms:.4f}")
        print(f"{'='*70}")

        print(f"\nDelta percentiles:")
        for k, v in pcts.items():
            print(f"  {k}: {v:.4f}")
        print(f"  max: {result['max_delta']:.4f} at {result['max_delta_time_s']:.3f}s")

        print(f"\nTransient counts (speech only, deduped):")
        print(f"  delta>0.15: {len(transients_015)} ({result['per_sec_015']}/sec)")
        print(f"  delta>0.25: {len(transients_025)} ({result['per_sec_025']}/sec)")
        print(f"  delta>0.50: {len(transients_050)} ({result['per_sec_050']}/sec)")

        print(f"\nChunk boundary correlation (delta>0.15):")
        print(f"  at boundary: {len(boundary_events)}, mid-chunk: {len(non_boundary)}")
        pct_boundary = len(boundary_events) / len(transients_015) * 100 if transients_015 else 0
        n_boundaries = len(audio) // CHUNK_SAMPLES
        expected_pct = n_boundaries * 20 / len(audio) * 100  # 20-sample window per boundary
        print(f"  boundary %: {pct_boundary:.1f}% (random expectation: {expected_pct:.2f}%)")

        # Show top 10 transients
        all_t = sorted(transients_015, key=lambda e: e['delta'], reverse=True)[:10]
        if all_t:
            print(f"\nTop {len(all_t)} transients by delta:")
            for i, e in enumerate(all_t):
                bnd = " [BOUNDARY]" if e['at_chunk_boundary'] else ""
                print(f"  #{i+1} {e['time_s']:.3f}s: Δ={e['delta']:.4f} ({e['waveform_type']}), "
                      f"burst={e['burst_samples']}samp ({e['burst_us']:.0f}µs), "
                      f"Δ/rms={e['delta_over_rms']:.1f}x{bnd}")
                # Show 5 samples before and after the jump
                idx = e['sample']
                before = [round(float(audio[max(0,idx-4+j)]), 3) for j in range(5)]
                after = [round(float(audio[min(len(audio)-1, idx+1+j)]), 3) for j in range(5)]
                print(f"       [{', '.join(f'{x:+.3f}' for x in before)}] → [{', '.join(f'{x:+.3f}' for x in after)}]")

        # Waveform type breakdown
        if transients_015:
            spikes = sum(1 for e in transients_015 if e['waveform_type'] == 'spike')
            steps = len(transients_015) - spikes
            print(f"\nWaveform type (delta>0.15):")
            print(f"  spike (goes up then back): {spikes} ({100*spikes/len(transients_015):.0f}%)")
            print(f"  step (stays at new level):  {steps} ({100*steps/len(transients_015):.0f}%)")

        # Burst duration distribution
        if transients_015:
            bursts = [e['burst_samples'] for e in transients_015]
            print(f"\nBurst duration distribution:")
            print(f"  1 sample:  {sum(1 for b in bursts if b == 1)}")
            print(f"  2-3 samp:  {sum(1 for b in bursts if 2 <= b <= 3)}")
            print(f"  4-10 samp: {sum(1 for b in bursts if 4 <= b <= 10)}")
            print(f"  >10 samp:  {sum(1 for b in bursts if b > 10)}")

    return result, transients_015


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='WAV file or directory')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--delta-threshold', type=float, default=0.15)
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        analyze_file(path, args.delta_threshold)
    elif path.is_dir() and args.all:
        files = sorted(path.glob('*.wav'))
        if not files:
            print(f"No WAV files found")
            sys.exit(1)

        all_results = []
        all_transients = []
        for f in files:
            r, t = analyze_file(f, args.delta_threshold)
            all_results.append(r)
            all_transients.extend(t)

        # Cross-file summary
        print(f"\n{'='*70}")
        print(f"SUMMARY ({len(all_results)} files)")
        print(f"{'='*70}")

        print(f"\n{'File':<12} {'Dur':>5} {'Peak':>5} {'MaxΔ':>6} {'T>0.15':>6} {'T>0.25':>6} {'T>0.50':>6} {'Bnd%':>5}")
        print("-" * 60)
        for r in all_results:
            bnd_pct = r['at_chunk_boundary'] / r['transients_015'] * 100 if r['transients_015'] > 0 else 0
            print(f"{r['file']:<12} {r['duration_s']:>5.1f} {r['peak']:>5.3f} {r['max_delta']:>6.4f} "
                  f"{r['transients_015']:>6} {r['transients_025']:>6} {r['transients_050']:>6} {bnd_pct:>5.1f}")

        if all_transients:
            # Global waveform type
            spikes = sum(1 for e in all_transients if e['waveform_type'] == 'spike')
            steps = len(all_transients) - spikes
            print(f"\nGlobal waveform type: spike={spikes} ({100*spikes/len(all_transients):.0f}%), step={steps} ({100*steps/len(all_transients):.0f}%)")

            # Global burst duration
            bursts = [e['burst_samples'] for e in all_transients]
            print(f"Global burst duration: 1samp={sum(1 for b in bursts if b==1)}, "
                  f"2-3={sum(1 for b in bursts if 2<=b<=3)}, "
                  f"4-10={sum(1 for b in bursts if 4<=b<=10)}, "
                  f">10={sum(1 for b in bursts if b>10)}")

            # Delta/RMS ratio distribution
            ratios = [e['delta_over_rms'] for e in all_transients]
            print(f"Delta/RMS ratio: mean={np.mean(ratios):.1f}x, median={np.median(ratios):.1f}x, "
                  f"max={max(ratios):.1f}x")


if __name__ == '__main__':
    main()
