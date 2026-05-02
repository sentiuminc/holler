#!/usr/bin/env python3
"""[DEPRECATED — not part of the active pipeline. Keep for reference and future use.]

== WHY DEPRECATED ==

Thresholds were calibrated against Cartesia's CustomVoice built-in voices
(Serena = gold standard, real recorded audio). For clean synthetic cloner output
(1.7B-Base-bf16 + enhance_clean.py), they are too aggressive:

- HNR < 14 dB rejects ~25% of clips that sound fine to the ear. Low HNR on
  synthetic audio often just reflects certain phonemes or sentences, not actual
  voice quality problems. Tested on Nora 2026-05-02: 69/500 rejected by HNR,
  only ~6 of those were audibly rough. The rest sounded fine.
- Manual tinder curation (curate_clips.py) catches real problems by ear —
  wrong emphasis, pacing, voice drift, EOS cutoffs — which metrics can't detect.
  It makes this automated gate redundant.

The current pipeline is: generate → enhance_clean.py → curate_clips.py (manual).
Auto-curation sits between those two and adds noise, not signal.

May be useful again for: real mic recordings with actual noise, future cloners
with lower quality output, or bulk-processing many voices without time to listen.
The analyze_voice_quality.py tool still runs the same metrics without rejecting.

----------------------------------------------------------------------

== WHAT THIS DOES ==

This is Step 4 of the Holler training data pipeline:
  1. generate_training_data.py  → clones voice into 500 raw clips (audio-original/)
  2. enhance_clips.py           → DeepFilter + LUFS + de-ess + presence (audio/)
  3. [optional] curate_clips.py → manual Tinder-style listening pass
  4. auto_curate.py (THIS)      → automated quality gate based on metrics

It analyzes every enhanced clip in audio/ using 25+ acoustic metrics (spectral,
perceptual DNSMOS, voice quality via Praat), then rejects clips that fail any of
the fixed quality thresholds. The thresholds were derived from analyzing
Cartesia's CustomVoice built-in voices (Serena = gold standard) and validated
by ear at 90% volume on AirPods Pro 3.

== WHY FIXED THRESHOLDS ==

The same thresholds apply to all voices regardless of gender. Male voices
naturally have more 2-4kHz energy and lower HNR, so they reject more clips
(~40-45% vs ~25-30% for female). The solution is to generate more source clips
for male voices, NOT to loosen thresholds. Quality > quantity for training data.

== WHAT IT DOES NOT DO ==

- Does NOT delete any audio files. Ever.
- Does NOT modify audio-original/ or audio/.
- Only writes two files (when --apply is passed):
  - rejects_auto.txt: list of filenames that failed (one per line)
  - train_curated.jsonl: filtered copy of train.jsonl with rejects removed
- Always writes analysis.json (full per-clip metrics) regardless of --apply.

== THRESHOLDS ==

  Peak > -4 dBFS        — too hot, risks clipping downstream
  Harshness 2-4kHz > 2% — ear pain predictor (more than volume)
  Sibilance 4-10kHz > 3% — harsh S/T/Ch sounds
  HNR < 14 dB           — breathy, rough, or damaged voice quality
  DNSMOS OVRL < 3.0     — neural perceptual quality score (P.835 model)
  Silence > 50%         — clip is mostly silence, not useful
  Duration < 0.5s       — too short for the model to learn from

== AFTER THIS ==

The user should Tinder-curate (curate_clips.py) to catch prosody issues that
metrics can't detect: wrong emphasis, unnatural pacing, robotic monotone, or
voice identity drift. Metrics catch technical quality; ears catch naturalness.

== USAGE ==

  python auto_curate.py --voice nora          # report only (default)
  python auto_curate.py --voice joe           # report only
  python auto_curate.py --voice nora --apply  # write rejects_auto.txt + train_curated.jsonl

Requires .venv-enhance-audio/ (parselmouth, torchmetrics, onnxruntime, scipy).
"""
import argparse
import json
import sys
from pathlib import Path

VOICES_DIR = Path(__file__).parent.parent / "voices"

THRESHOLDS = {
    "peak_db": ("Peak > -0.5 dBFS", lambda v: v > -0.5),
    "harsh_2_4k": ("Harshness > 2%", lambda v: v > 0.02),
    "sib_4_10k": ("Sibilance > 3%", lambda v: v > 0.03),
    "hnr_db": ("HNR < 14 dB", lambda v: v < 14),
    "dnsmos_ovrl": ("DNSMOS OVRL < 3.0", lambda v: v < 3.0),
    "silence": ("Silence > 50%", lambda v: v > 0.5),
    "dur": ("Duration < 0.5s", lambda v: v < 0.5),
}


def main():
    parser = argparse.ArgumentParser(description="Auto-curate training data by quality thresholds")
    parser.add_argument("--voice", required=True, help="Voice name (directory under voices/)")
    parser.add_argument("--apply", action="store_true", help="Write rejects_auto.txt + train_curated.jsonl (default is report only)")
    args = parser.parse_args()

    voice_dir = VOICES_DIR / args.voice
    training_dir = voice_dir / "training-data"
    audio_dir = training_dir / "audio"
    jsonl_path = training_dir / "train.jsonl"
    rejects_path = training_dir / "rejects_auto.txt"
    curated_path = training_dir / "train_curated.jsonl"
    analysis_path = training_dir / "analysis.json"

    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} not found. Run enhance_clips.py first.")
        return

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found.")
        return

    # Step 1: Run analysis on all clips
    wavs = sorted(f.name for f in audio_dir.iterdir() if f.suffix == ".wav")
    print(f"Analyzing {len(wavs)} clips in {audio_dir}...")

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_voice_quality import analyze_clip

    clips = []
    for i, wav in enumerate(wavs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(wavs)}]")
        result = analyze_clip(audio_dir / wav)
        result["file"] = wav
        clips.append(result)

    print(f"  [{len(wavs)}/{len(wavs)}] done.")
    print()

    # Always save full analysis
    with open(analysis_path, "w") as f:
        json.dump(clips, f, indent=2)
    print(f"Analysis saved: {analysis_path}")
    print()

    # Step 2: Check thresholds
    criterion_fails = {key: [] for key in THRESHOLDS}
    failing_files = set()

    for clip in clips:
        for key, (label, check) in THRESHOLDS.items():
            if key in clip and check(clip[key]):
                criterion_fails[key].append(clip)
                failing_files.add(clip["file"])

    # Step 3: Report
    print(f"{'Criterion':<25} {'Failing':>8} {'%':>6}")
    print("-" * 42)
    for key, (label, _) in THRESHOLDS.items():
        n = len(criterion_fails[key])
        print(f"{label:<25} {n:>8} {n*100/len(clips):>5.1f}%")

    print()
    print(f"{'Total failing':<25} {len(failing_files):>8} ({len(failing_files)*100/len(clips):.1f}%)")
    print(f"{'Would keep':<25} {len(clips) - len(failing_files):>8} ({(len(clips)-len(failing_files))*100/len(clips):.1f}%)")

    # Show failing files (all if <=20, otherwise worst 20)
    if failing_files:
        failing_clips = [c for c in clips if c["file"] in failing_files]
        failing_clips.sort(key=lambda c: sum(1 for k, (_, chk) in THRESHOLDS.items() if k in c and chk(c[k])), reverse=True)

        show = failing_clips if len(failing_clips) <= 20 else failing_clips[:20]
        print(f"\n{'Worst failing clips' if len(failing_clips) > 20 else 'Failing clips'}:")
        for c in show:
            reasons = []
            for key, (label, check) in THRESHOLDS.items():
                if key in c and check(c[key]):
                    if key in ("harsh_2_4k", "sib_4_10k"):
                        reasons.append(f"{label.split('>')[0].strip()}={c[key]*100:.1f}%")
                    elif key == "dur":
                        reasons.append(f"dur={c[key]:.2f}s")
                    else:
                        reasons.append(f"{key}={c[key]:.1f}")

            print(f"  {c['file']}: {', '.join(reasons)}")

        if len(failing_clips) > 20:
            print(f"  ... and {len(failing_clips) - 20} more (see analysis.json for full list)")

    if not args.apply:
        print("\n(report only — pass --apply to write rejects_auto.txt + train_curated.jsonl)")
        return

    # Step 4: Write rejects list
    with open(rejects_path, "w") as f:
        for fname in sorted(failing_files):
            f.write(fname + "\n")
    print(f"\nWrote {len(failing_files)} rejects → {rejects_path}")

    # Step 5: Write curated JSONL (train.jsonl minus rejects)
    kept = 0
    with open(jsonl_path) as src, open(curated_path, "w") as dst:
        for line in src:
            entry = json.loads(line)
            fname = entry["audio"].split("/")[-1]
            if fname not in failing_files:
                dst.write(line)
                kept += 1

    print(f"Wrote {kept} curated entries → {curated_path}")


if __name__ == "__main__":
    main()
