#!/usr/bin/env python3
"""Stability benchmark for Holler checkpoints.

Generates N samples per voice, detects stuttering/instability artifacts
automatically, and reports per-voice artifact rates.

Artifact detection signals:
1. Mid-utterance silence gaps (model gets stuck mid-sentence)
2. Spectral discontinuities (sudden energy drops/spikes = garbled audio)
3. Repeated segments (spectral self-similarity in short windows)
4. STT round-trip (FluidAudio Parakeet) — if ASR can't transcribe it, something's wrong

Usage:
  # Generate + analyze (requires holler CLI binary)
  python benchmark_stability.py --checkpoint checkpoints/holler-6voice-v1-6bit --samples 20

  # Analyze existing samples only
  python benchmark_stability.py --analyze-dir ~/Downloads/holler-stability-test/6bit

  # Compare two checkpoints
  python benchmark_stability.py --checkpoint checkpoints/holler-6voice-v1-6bit --samples 20 --label 6bit
  python benchmark_stability.py --checkpoint checkpoints/holler-6voice-v1-bf16 --samples 20 --label bf16
  python benchmark_stability.py --compare ~/Downloads/holler-stability-test/6bit ~/Downloads/holler-stability-test/bf16
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sig

SR = 24000
FLUIDAUDIO_CLI = os.path.expanduser(
    "~/Desktop/Files/AI/ivi/audio-test/FluidAudio/.build/arm64-apple-macosx/debug/fluidaudiocli"
)

VOICES = ["kit", "dakota", "nora", "joe", "oliver", "tessa"]

SENTENCES = [
    "So I looked into it and here's what I found. The configuration was wrong the whole time, but it's fixed now.",
    "Wait, are you serious? That's actually amazing. I can't believe we didn't try that before.",
    "Okay, let me walk you through this step by step. First, you need to open the settings panel and find the network tab.",
    "I mean, honestly, I think the bigger question is whether we should even be doing this in the first place.",
    "Right, so the problem is that the server keeps timing out after about thirty seconds, and nobody knows why.",
    "Look, I already checked the numbers twice. They don't add up, and I'm not going to sugarcoat it for you.",
    "That's a great point actually. I hadn't considered it from that angle before. Let me think about it.",
    "The thing is, we could probably ship this tomorrow if we just cut the authentication feature for now.",
    "Hey, quick question. Do you know if the deployment pipeline supports rolling updates, or is it all or nothing?",
    "I was thinking about what you said earlier, and I think you're right. We should go with the simpler approach.",
    "Alright, here's the deal. We have three options, and none of them are perfect, but we need to pick one today.",
    "So the short answer is yes, but the long answer involves a lot of caveats that I think are worth discussing.",
    "Can you pull up the dashboard? I want to show you something interesting about the latency numbers from last week.",
    "Honestly, I'm not sure that's going to work. But hey, let's try it and see what happens.",
    "The fix is actually pretty straightforward once you understand what's going on under the hood.",
    "What if we just bypass the cache entirely? I know it sounds crazy, but hear me out for a second.",
    "I ran the tests three times and got different results each time. Something is definitely flaky here.",
    "Perfect, that's exactly what I was hoping to hear. Let's move forward with that plan then.",
    "No, no, no. That's not what I meant at all. Let me rephrase that so it's clearer.",
    "Sure, I can take a look at that. Give me about ten minutes and I'll have an answer for you.",
]


def detect_mid_silence(audio, sr, min_gap_ms=800, threshold_db=-40):
    """Detect abnormally long silence gaps in the middle of speech.

    Normal inter-sentence pauses are 200-700ms. We flag gaps > 800ms as
    potential artifacts (model getting stuck). Short pauses at sentence
    boundaries are expected and ignored.

    Returns list of (start_ms, duration_ms) for each abnormal gap.
    """
    frame_len = int(0.010 * sr)  # 10ms frames
    hop = int(0.005 * sr)  # 5ms hop
    n = len(audio)

    frame_rms = []
    for i in range(0, n - frame_len, hop):
        frame = audio[i : i + frame_len]
        rms = np.sqrt(np.mean(frame**2))
        frame_rms.append(20 * np.log10(rms + 1e-10))
    frame_rms = np.array(frame_rms)

    is_silent = frame_rms < threshold_db

    # Find speech boundaries (first and last non-silent frame)
    speech_frames = np.where(~is_silent)[0]
    if len(speech_frames) < 10:
        return []

    speech_start = speech_frames[0]
    speech_end = speech_frames[-1]

    # Find silent runs within the speech region
    gaps = []
    in_gap = False
    gap_start = 0
    for i in range(speech_start, speech_end + 1):
        if is_silent[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_silent[i] and in_gap:
            in_gap = False
            gap_ms = (i - gap_start) * (hop / sr) * 1000
            if gap_ms >= min_gap_ms:
                start_ms = gap_start * (hop / sr) * 1000
                gaps.append((round(start_ms), round(gap_ms)))

    return gaps


def estimate_expected_duration(text, wpm=160):
    """Estimate expected audio duration from text length.

    Average TTS speech rate is ~150-170 words per minute.
    Returns (min_s, expected_s, max_s).
    """
    words = len(text.split())
    expected_s = words / wpm * 60
    return (expected_s * 0.5, expected_s, expected_s * 2.0)


def detect_repetition(audio, sr, window_ms=200, hop_ms=50, threshold=0.97, min_consecutive=5):
    """Detect repeated audio segments via spectral self-similarity.

    Normal speech has moderate similarity between adjacent windows (~0.85-0.95).
    Stuttering/looping produces very high similarity (>0.97) across 3+ consecutive
    non-overlapping windows — the same spectral pattern repeating.

    Returns list of (time_ms, similarity_score, n_repetitions).
    """
    window_samples = int(window_ms / 1000 * sr)
    hop_samples = int(hop_ms / 1000 * sr)
    n = len(audio)

    features = []
    for i in range(0, n - window_samples, hop_samples):
        frame = audio[i : i + window_samples]
        rms = np.sqrt(np.mean(frame**2))
        if 20 * np.log10(rms + 1e-10) < -40:
            features.append(None)  # silence, skip
            continue
        fft = np.fft.rfft(frame * np.hanning(len(frame)))
        mag = np.abs(fft)
        n_fft = len(mag)
        bands = np.array_split(mag[: n_fft // 2], 20)
        mel = np.array([np.mean(b**2) for b in bands])
        mel_db = 10 * np.log10(mel + 1e-10)
        features.append(mel_db)

    if len(features) < 6:
        return []

    # Look for consecutive windows with very high similarity to their neighbor
    # (i.e., the model is stuck outputting the same pattern)
    sims = []
    for i in range(len(features) - 1):
        if features[i] is None or features[i + 1] is None:
            sims.append(0)
            continue
        a, b = features[i], features[i + 1]
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            sims.append(np.dot(a, b) / (norm_a * norm_b))
        else:
            sims.append(0)

    # Find runs of consecutive high-similarity windows
    repetitions = []
    run_start = None
    run_len = 0
    for i, sim in enumerate(sims):
        if sim > threshold:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= min_consecutive:
                time_ms = run_start * hop_ms
                avg_sim = np.mean(sims[run_start : run_start + run_len])
                repetitions.append((round(time_ms), round(avg_sim, 3), run_len))
            run_start = None
            run_len = 0

    if run_len >= min_consecutive and run_start is not None:
        time_ms = run_start * hop_ms
        avg_sim = np.mean(sims[run_start : run_start + run_len])
        repetitions.append((round(time_ms), round(avg_sim, 3), run_len))

    return repetitions


def transcribe_wav(wav_path):
    """Transcribe a WAV file using FluidAudio Parakeet. Returns (text, confidence)."""
    if not os.path.exists(FLUIDAUDIO_CLI):
        return None, None
    try:
        result = subprocess.run(
            [FLUIDAUDIO_CLI, "transcribe", str(wav_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None, None
        # Last non-empty line of stdout is the transcription
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        text = lines[-1] if lines else ""
        # Parse confidence from log output
        confidence = None
        for line in result.stderr.split("\n") + result.stdout.split("\n"):
            if "Confidence:" in line:
                try:
                    confidence = float(line.split("Confidence:")[1].strip())
                except (ValueError, IndexError):
                    pass
        return text, confidence
    except (subprocess.TimeoutExpired, Exception):
        return None, None


def word_error_rate(reference, hypothesis):
    """Compute WER between reference and hypothesis text. Returns (wer, details)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple Levenshtein on word level
    r = len(ref_words)
    h = len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            # Strip punctuation for comparison
            rw = ref_words[i - 1].strip(".,!?;:'\"")
            hw = hyp_words[j - 1].strip(".,!?;:'\"")
            cost = 0 if rw == hw else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    wer = d[r][h] / r if r > 0 else 0
    return round(wer, 3), {"ref_words": r, "hyp_words": h, "edit_distance": d[r][h]}


def analyze_clip(path, expected_text=None, run_stt=True):
    """Run all stability detectors on a single clip. Returns dict."""
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != SR:
        audio = sig.resample_poly(audio, SR, sr).astype(np.float32)
        sr = SR

    duration_s = len(audio) / sr

    gaps = detect_mid_silence(audio, sr)
    repetitions = detect_repetition(audio, sr)

    # Duration anomaly check
    duration_anomaly = False
    if expected_text:
        min_s, exp_s, max_s = estimate_expected_duration(expected_text)
        duration_anomaly = duration_s < min_s or duration_s > max_s

    # STT round-trip
    stt_text = None
    stt_confidence = None
    stt_wer = None
    if run_stt and expected_text:
        stt_text, stt_confidence = transcribe_wav(path)
        if stt_text and expected_text:
            stt_wer, _ = word_error_rate(expected_text, stt_text)

    # Overall quality: peak, RMS, LUFS estimate
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    peak_db = 20 * np.log10(peak + 1e-10)
    rms_db = 20 * np.log10(rms + 1e-10)

    # High WER = likely artifact (garbled speech)
    stt_failed = stt_wer is not None and stt_wer > 0.3

    has_artifact = len(gaps) > 0 or len(repetitions) > 0 or stt_failed or duration_anomaly

    return {
        "file": str(path),
        "duration_s": round(duration_s, 2),
        "peak_db": round(peak_db, 1),
        "rms_db": round(rms_db, 1),
        "mid_silence_gaps": gaps,
        "repetitions": repetitions,
        "duration_anomaly": duration_anomaly,
        "stt_text": stt_text,
        "stt_confidence": round(stt_confidence, 3) if stt_confidence else None,
        "stt_wer": stt_wer,
        "expected_text": expected_text,
        "has_artifact": has_artifact,
        "artifact_types": (
            (["silence_gap"] if gaps else [])
            + (["repetition"] if repetitions else [])
            + (["duration_anomaly"] if duration_anomaly else [])
            + (["stt_failed"] if stt_failed else [])
        ),
    }


def generate_samples(checkpoint, output_dir, holler_bin, n_samples=20, voices=None):
    """Generate samples using holler CLI. Also saves sentence map for STT verification."""
    voices = voices or VOICES
    os.makedirs(output_dir, exist_ok=True)

    sentences = SENTENCES[:n_samples]
    total = len(voices) * len(sentences)
    done = 0

    # Save sentence map so analysis can do STT round-trip
    sentence_map = {f"s{i:02d}": text for i, text in enumerate(sentences)}
    with open(os.path.join(output_dir, "sentences.json"), "w") as f:
        json.dump(sentence_map, f, indent=2)

    for voice in voices:
        voice_dir = os.path.join(output_dir, voice)
        os.makedirs(voice_dir, exist_ok=True)

        for i, text in enumerate(sentences):
            out_path = os.path.join(voice_dir, f"s{i:02d}.wav")
            if os.path.exists(out_path):
                done += 1
                continue

            cmd = [
                holler_bin,
                "--text", text,
                "--voice", voice,
                "--model", checkpoint,
                "--output", out_path,
                "--temperature", "0.6",
            ]
            done += 1
            print(f"  [{done}/{total}] {voice} s{i:02d}...", end=" ", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"FAILED: {result.stderr[:200]}")
            elif os.path.exists(out_path):
                audio, _ = sf.read(out_path)
                print(f"ok ({len(audio)/SR:.1f}s)")
            else:
                print("no output file")


def analyze_directory(sample_dir, run_stt=True):
    """Analyze all WAV files in a directory tree. Returns per-voice stats."""
    results = {}
    all_clips = []

    sample_path = Path(sample_dir)

    # Load sentence map if available (for STT round-trip)
    sentence_map = {}
    sentences_file = sample_path / "sentences.json"
    if sentences_file.exists():
        with open(sentences_file) as f:
            sentence_map = json.load(f)

    for voice_dir in sorted(sample_path.iterdir()):
        if not voice_dir.is_dir():
            continue
        voice = voice_dir.name
        results[voice] = []

        wavs = sorted(voice_dir.glob("*.wav"))
        total_wavs = len(wavs)
        for wi, wav in enumerate(wavs):
            stem = wav.stem  # e.g. "s00"
            expected_text = sentence_map.get(stem)
            print(f"  Analyzing {voice}/{wav.name} ({wi+1}/{total_wavs})...", end=" ", flush=True)
            clip = analyze_clip(wav, expected_text=expected_text, run_stt=run_stt)
            clip["voice"] = voice
            results[voice].append(clip)
            all_clips.append(clip)
            status = "ARTIFACT" if clip["has_artifact"] else "ok"
            wer_str = f" WER={clip['stt_wer']:.0%}" if clip.get("stt_wer") is not None else ""
            print(f"{status}{wer_str}")

    return results, all_clips


def print_report(results, all_clips, label=""):
    """Print stability report."""
    header = f"\n{'='*60}\nSTABILITY REPORT"
    if label:
        header += f": {label}"
    header += f"\n{'='*60}"
    print(header)

    total = len(all_clips)
    artifacts = [c for c in all_clips if c["has_artifact"]]
    artifact_rate = len(artifacts) / total * 100 if total > 0 else 0

    print(f"\nTotal clips: {total}")
    print(f"Artifacts detected: {len(artifacts)} ({artifact_rate:.1f}%)")

    # Per-type breakdown
    type_counts = {}
    for c in artifacts:
        for t in c["artifact_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        print(f"\nArtifact breakdown:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count} clips ({count/total*100:.1f}%)")

    # Per-voice
    print(f"\nPer-voice artifact rate:")
    print(f"  {'Voice':<10} {'Clips':>6} {'Artifacts':>10} {'Rate':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8}")
    for voice, clips in sorted(results.items()):
        n = len(clips)
        bad = sum(1 for c in clips if c["has_artifact"])
        rate = bad / n * 100 if n > 0 else 0
        marker = " <<<" if rate > 20 else ""
        print(f"  {voice:<10} {n:>6} {bad:>10} {rate:>7.1f}%{marker}")

    # STT round-trip stats
    stt_clips = [c for c in all_clips if c.get("stt_wer") is not None]
    if stt_clips:
        wers = [c["stt_wer"] for c in stt_clips]
        confs = [c["stt_confidence"] for c in stt_clips if c.get("stt_confidence") is not None]
        print(f"\nSTT round-trip (FluidAudio Parakeet):")
        print(f"  Clips tested: {len(stt_clips)}")
        print(f"  Mean WER: {np.mean(wers):.1%}  (median {np.median(wers):.1%})")
        print(f"  WER > 30%: {sum(1 for w in wers if w > 0.3)} clips ({sum(1 for w in wers if w > 0.3)/len(wers)*100:.1f}%)")
        if confs:
            print(f"  Mean confidence: {np.mean(confs):.3f}")

    # Worst clips (for spot-checking)
    if artifacts:
        print(f"\nFlagged clips (spot-check these):")
        for c in artifacts[:15]:
            types = ", ".join(c["artifact_types"])
            fname = os.path.basename(os.path.dirname(c["file"])) + "/" + os.path.basename(c["file"])
            details = []
            if c["mid_silence_gaps"]:
                details.append(f"gaps: {c['mid_silence_gaps']}")
            if c["repetitions"]:
                details.append(f"repeats: {len(c['repetitions'])}")
            if c.get("duration_anomaly"):
                exp = estimate_expected_duration(c.get("expected_text", ""))[1]
                details.append(f"dur={c['duration_s']:.1f}s (expect ~{exp:.1f}s)")
            if c.get("stt_wer") is not None and c["stt_wer"] > 0.3:
                details.append(f"WER={c['stt_wer']:.0%}")
            print(f"  {fname:<25} [{types}] {'; '.join(details)}")


def print_comparison(dir_a, dir_b, label_a, label_b, run_stt=True):
    """Compare two directories of samples."""
    results_a, clips_a = analyze_directory(dir_a, run_stt=run_stt)
    results_b, clips_b = analyze_directory(dir_b, run_stt=run_stt)

    print_report(results_a, clips_a, label_a)
    print_report(results_b, clips_b, label_b)

    # Side-by-side summary
    rate_a = sum(1 for c in clips_a if c["has_artifact"]) / len(clips_a) * 100 if clips_a else 0
    rate_b = sum(1 for c in clips_b if c["has_artifact"]) / len(clips_b) * 100 if clips_b else 0

    print(f"\n{'='*60}")
    print(f"COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"  {label_a}: {rate_a:.1f}% artifact rate")
    print(f"  {label_b}: {rate_b:.1f}% artifact rate")

    # Per-voice comparison
    all_voices = sorted(set(list(results_a.keys()) + list(results_b.keys())))
    print(f"\n  {'Voice':<10} {label_a:>12} {label_b:>12} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for voice in all_voices:
        clips_va = results_a.get(voice, [])
        clips_vb = results_b.get(voice, [])
        ra = sum(1 for c in clips_va if c["has_artifact"]) / len(clips_va) * 100 if clips_va else 0
        rb = sum(1 for c in clips_vb if c["has_artifact"]) / len(clips_vb) * 100 if clips_vb else 0
        delta = rb - ra
        print(f"  {voice:<10} {ra:>11.1f}% {rb:>11.1f}% {delta:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Stability benchmark for Holler checkpoints")
    parser.add_argument("--checkpoint", "-c", help="Path to checkpoint for generation")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Samples per voice (default: 20)")
    parser.add_argument("--label", "-l", default="", help="Label for this run")
    parser.add_argument("--analyze-dir", "-a", help="Analyze existing samples (skip generation)")
    parser.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), help="Compare two sample directories")
    parser.add_argument("--compare-labels", nargs=2, default=["A", "B"], help="Labels for comparison")
    parser.add_argument("--holler-bin", default="./holler", help="Path to holler CLI binary")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: ~/Downloads/holler-stability-test/<label>)")
    parser.add_argument("--voices", nargs="+", help="Voices to test (default: all 6)")
    parser.add_argument("--json", help="Save per-clip results to JSON file")
    parser.add_argument("--no-stt", action="store_true", help="Skip STT round-trip (faster)")
    args = parser.parse_args()

    run_stt = not args.no_stt

    if args.compare:
        print_comparison(args.compare[0], args.compare[1], args.compare_labels[0], args.compare_labels[1])
        return

    if args.analyze_dir:
        results, all_clips = analyze_directory(args.analyze_dir, run_stt=run_stt)
        print_report(results, all_clips, args.label or args.analyze_dir)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(all_clips, f, indent=2)
        return

    if not args.checkpoint:
        parser.error("Need --checkpoint for generation or --analyze-dir for analysis")

    label = args.label or Path(args.checkpoint).name
    output_dir = args.output_dir or os.path.expanduser(f"~/Downloads/holler-stability-test/{label}")

    print(f"Generating {args.samples} samples per voice from {args.checkpoint}")
    print(f"Output: {output_dir}\n")

    generate_samples(
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        holler_bin=args.holler_bin,
        n_samples=args.samples,
        voices=args.voices or VOICES,
    )

    print("\nAnalyzing...")
    results, all_clips = analyze_directory(output_dir, run_stt=run_stt)
    print_report(results, all_clips, label)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_clips, f, indent=2)
        print(f"\nPer-clip data saved to {args.json}")


if __name__ == "__main__":
    main()
