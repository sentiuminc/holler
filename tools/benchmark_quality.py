#!/usr/bin/env python3
"""Quality benchmark for Holler checkpoints.

Uses Whisper word timestamps to detect mid-speech pauses, dropped words,
and timing artifacts. Tests the production path: multi-sentence paragraphs
generated via holler --session (carryover KV cache).

Artifact detection:
  1. Word-level gap analysis — flags abnormal pauses based on linguistic context
  2. Strict WER — flags ANY word mismatch (insertion, deletion, substitution)
  3. Duration sanity — total audio vs expected from word count

Usage:
  # Full run: generate + analyze
  python benchmark_quality.py -c checkpoints/holler-6voice-v1-6bit

  # Analyze existing samples only
  python benchmark_quality.py --analyze-dir ~/Downloads/holler-bench/v1-6bit

  # Compare two runs
  python benchmark_quality.py --compare ~/Downloads/holler-bench/v1 ~/Downloads/holler-bench/v2

  # Quick run (3 paragraphs instead of 10)
  python benchmark_quality.py -c checkpoints/holler-6voice-v1-6bit --quick
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
VOICES = ["kit", "dakota", "nora", "joe", "oliver", "tessa"]

PARAGRAPHS = [
    "So I looked into it and here's what I found. The configuration was completely wrong, but it's fixed now. Want me to walk you through what I changed?",
    "Wait, are you serious? That's actually amazing. I can't believe we didn't try that sooner. We should probably document this so nobody else wastes time on it.",
    "Okay, let me walk you through this step by step. First, open the settings panel and find the network tab. Then scroll down to the proxy section and toggle it off.",
    "I mean, honestly, I think the bigger question is whether we should even be doing this in the first place. The whole approach feels wrong to me.",
    "Right, so the problem is that the server keeps timing out after about thirty seconds. Nobody knows why. I've checked the logs and there's nothing useful in there.",
    "Look, I already checked the numbers twice. They don't add up, and I'm not going to sugarcoat it. We need to fix this before the meeting tomorrow morning.",
    "The fix is actually pretty straightforward once you understand what's going on. The database connection pool was exhausted because nobody was closing their connections properly.",
    "Can you pull up the dashboard? I want to show you something interesting about the latency numbers. Last week we had a spike that I think is related to this issue.",
    "What if we just bypass the cache entirely? I know it sounds crazy, but hear me out. The cache is causing more problems than it solves at this point.",
    "Perfect, that's exactly what I was hoping to hear. Let's move forward with that plan. I'll set up the staging environment and we can test it this afternoon.",
]

# Gap thresholds (ms) by context
GAP_THRESHOLDS = {
    "within_phrase": {"normal": 250, "suspect": 400, "artifact": 500},
    "comma": {"normal": 400, "suspect": 700, "artifact": 900},
    "sentence_end": {"normal": 800, "suspect": 1200, "artifact": 1500},
}


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}

CONTRACTION_EXPANSIONS = {
    "can't": "cannot", "cant": "cannot", "cannot": "cannot",
    "won't": "will not", "wont": "will not",
    "don't": "do not", "dont": "do not",
    "didn't": "did not", "didnt": "did not",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not",
    "wouldn't": "would not", "wouldnt": "would not",
    "shouldn't": "should not", "shouldnt": "should not",
    "couldn't": "could not", "couldnt": "could not",
    "i'm": "i am", "im": "i am",
    "i've": "i have", "ive": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are", "youre": "you are",
    "you've": "you have", "youve": "you have",
    "you'll": "you will", "youll": "you will",
    "you'd": "you would", "youd": "you would",
    "he's": "he is", "hes": "he is",
    "she's": "she is", "shes": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have", "weve": "we have",
    "we'll": "we will",
    "they're": "they are", "theyre": "they are",
    "they've": "they have", "theyve": "they have",
    "they'll": "they will", "theyll": "they will",
    "that's": "that is", "thats": "that is",
    "what's": "what is", "whats": "what is",
    "there's": "there is", "theres": "there is",
    "here's": "here is", "heres": "here is",
    "let's": "let us", "lets": "let us",
    "gonna": "going to", "gotta": "got to", "wanna": "want to",
    "kinda": "kind of", "sorta": "sort of",
}


def normalize_word(w):
    """Normalize a single word for comparison."""
    w = w.lower().strip()
    w = re.sub(r"[''']", "'", w)
    w = re.sub(r"[^\w']", "", w)
    return w


def expand_to_tokens(word):
    """Expand a word into normalized comparison tokens.
    Handles contractions, number words, and compound words so that
    'thirty' matches '30', 'can't' matches 'can not', 'sugarcoat' matches 'sugar coat'."""
    w = normalize_word(word)
    if not w:
        return []

    # Number normalization
    if w in NUMBER_WORDS:
        return [NUMBER_WORDS[w]]

    # Contraction expansion
    if w in CONTRACTION_EXPANSIONS:
        return CONTRACTION_EXPANSIONS[w].split()

    return [w]


def align_words(ref_words, hyp_words):
    """Align reference and hypothesis words using Levenshtein DP.
    Expands contractions and number words before comparison.
    Returns list of (op, ref_word, hyp_word) where op is 'match', 'sub', 'ins', 'del'."""
    # Expand both sides to normalized tokens
    ref_expanded = []
    ref_sources = []
    for rw in ref_words:
        tokens = expand_to_tokens(rw)
        for t in tokens:
            ref_expanded.append(t)
            ref_sources.append(rw)

    hyp_expanded = []
    hyp_sources = []
    for hw in hyp_words:
        tokens = expand_to_tokens(hw)
        for t in tokens:
            hyp_expanded.append(t)
            hyp_sources.append(hw)

    r = len(ref_expanded)
    h = len(hyp_expanded)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    ops = [[""] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
        ops[i][0] = "del"
    for j in range(h + 1):
        d[0][j] = j
        ops[0][j] = "ins"
    ops[0][0] = ""

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            cost = 0 if ref_expanded[i - 1] == hyp_expanded[j - 1] else 1
            candidates = [
                (d[i - 1][j] + 1, "del"),
                (d[i][j - 1] + 1, "ins"),
                (d[i - 1][j - 1] + cost, "match" if cost == 0 else "sub"),
            ]
            best = min(candidates, key=lambda x: x[0])
            d[i][j] = best[0]
            ops[i][j] = best[1]

    # Backtrace — report using original source words (not expanded tokens)
    alignment = []
    seen_ref = set()
    seen_hyp = set()
    i, j = r, h
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "match" or op == "sub":
            rs = ref_sources[i - 1]
            hs = hyp_sources[j - 1]
            key = (i - 1, j - 1)
            if op == "match":
                if rs not in seen_ref or hs not in seen_hyp:
                    alignment.append(("match", rs, hs))
                    seen_ref.add(rs)
                    seen_hyp.add(hs)
            else:
                alignment.append(("sub", rs, hs))
            i -= 1
            j -= 1
        elif op == "del":
            alignment.append(("del", ref_sources[i - 1], None))
            i -= 1
        elif op == "ins":
            alignment.append(("ins", None, hyp_sources[j - 1]))
            j -= 1
        else:
            break

    alignment.reverse()

    # Deduplicate: expanded tokens can produce multiple ops for the same source word.
    # Collapse consecutive matches/ops on the same source word pair.
    collapsed = []
    for op, rw, hw in alignment:
        if collapsed and collapsed[-1] == (op, rw, hw):
            continue
        collapsed.append((op, rw, hw))

    # Count real edit distance on original words (not expanded)
    edit_dist = sum(1 for op, _, _ in collapsed if op != "match")

    return collapsed, edit_dist


def classify_gap_context(prev_word_raw):
    """Determine gap context from the raw word preceding the gap."""
    if not prev_word_raw:
        return "within_phrase"
    stripped = prev_word_raw.strip()
    if stripped and stripped[-1] in ".!?":
        return "sentence_end"
    if stripped and stripped[-1] in ",;:—–":
        return "comma"
    return "within_phrase"


def judge_gap(duration_ms, context):
    """Judge a gap: 'normal', 'suspect', or 'artifact'."""
    thresholds = GAP_THRESHOLDS.get(context, GAP_THRESHOLDS["within_phrase"])
    if duration_ms <= thresholds["normal"]:
        return "normal"
    elif duration_ms <= thresholds["suspect"]:
        return "suspect"
    else:
        return "artifact"


def transcribe_with_timestamps(wav_path, whisper_model):
    """Transcribe audio with word-level timestamps using mlx-whisper."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=whisper_model,
        word_timestamps=True,
        language="en",
    )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    full_text = result.get("text", "").strip()
    return words, full_text


def analyze_clip(wav_path, expected_text, whisper_model):
    """Full analysis of one clip. Returns detailed result dict."""
    audio, sr = sf.read(str(wav_path))
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    duration_s = len(audio) / sr

    # Transcribe
    words, whisper_text = transcribe_with_timestamps(wav_path, whisper_model)

    if not words:
        return {
            "file": str(wav_path),
            "duration_s": round(duration_s, 2),
            "error": "no words detected",
            "has_artifact": True,
            "artifact_types": ["no_speech"],
            "artifacts": [],
            "word_errors": [],
            "gaps": [],
            "wer": 1.0,
            "whisper_text": whisper_text,
            "expected_text": expected_text,
        }

    # Word alignment for strict comparison
    ref_words = expected_text.split()
    hyp_words = [w["word"] for w in words]
    alignment, edit_distance = align_words(ref_words, hyp_words)

    wer = edit_distance / len(ref_words) if ref_words else 0
    word_errors = [
        {"op": op, "ref": rw, "hyp": hw}
        for op, rw, hw in alignment
        if op != "match"
    ]

    # Gap analysis using whisper word timestamps
    gaps = []
    artifacts = []

    for i in range(1, len(words)):
        gap_start = words[i - 1]["end"]
        gap_end = words[i]["start"]
        gap_ms = (gap_end - gap_start) * 1000

        if gap_ms < 0:
            gap_ms = 0

        prev_raw = words[i - 1]["word"]
        context = classify_gap_context(prev_raw)
        judgment = judge_gap(gap_ms, context)

        gap_info = {
            "after": words[i - 1]["word"],
            "before": words[i]["word"],
            "start_s": round(gap_start, 3),
            "end_s": round(gap_end, 3),
            "duration_ms": round(gap_ms),
            "context": context,
            "judgment": judgment,
        }
        gaps.append(gap_info)

        if judgment == "artifact":
            artifacts.append({
                "type": "pause",
                "time_s": round(gap_start, 3),
                "duration_ms": round(gap_ms),
                "context": context,
                "between": f"'{words[i-1]['word']}' → '{words[i]['word']}'",
            })

    # Compile artifact types
    artifact_types = []
    if artifacts:
        artifact_types.append("pause")
    if word_errors:
        artifact_types.append("word_error")

    has_artifact = len(artifacts) > 0 or len(word_errors) > 0

    return {
        "file": str(wav_path),
        "duration_s": round(duration_s, 2),
        "has_artifact": has_artifact,
        "artifact_types": artifact_types,
        "artifacts": artifacts,
        "word_errors": word_errors,
        "gaps": gaps,
        "wer": round(wer, 4),
        "n_words_ref": len(ref_words),
        "n_words_hyp": len(hyp_words),
        "edit_distance": edit_distance,
        "whisper_text": whisper_text,
        "expected_text": expected_text,
        "first_word_time_ms": round(words[0]["start"] * 1000) if words else None,
        "last_word_end_s": round(words[-1]["end"], 3) if words else None,
    }


def generate_samples(checkpoint, output_dir, holler_bin, paragraphs, voices=None, raw=False, temperature=0.6):
    """Generate samples using holler CLI.

    raw=False: --session mode (sentence-by-sentence with carryover, retry, silence abort)
    raw=True:  --text mode (direct synthesis, no guardrails — shows raw model behavior)
    """
    voices = voices or VOICES
    os.makedirs(output_dir, exist_ok=True)

    para_map = {f"p{i:02d}": text for i, text in enumerate(paragraphs)}
    with open(os.path.join(output_dir, "paragraphs.json"), "w") as f:
        json.dump(para_map, f, indent=2)

    mode_label = "raw --text" if raw else "--session"
    total = len(voices) * len(paragraphs)
    done = 0

    for voice in voices:
        voice_dir = os.path.join(output_dir, voice)
        os.makedirs(voice_dir, exist_ok=True)

        for i, text in enumerate(paragraphs):
            out_path = os.path.join(voice_dir, f"p{i:02d}.wav")
            done += 1
            if os.path.exists(out_path):
                audio, _ = sf.read(out_path)
                print(f"  [{done}/{total}] {voice}/p{i:02d} — exists ({len(audio)/SR:.1f}s)")
                continue

            print(f"  [{done}/{total}] {voice}/p{i:02d}...", end=" ", flush=True)
            cmd = [
                holler_bin,
                "--text", text,
                "--voice", voice,
                "--model", checkpoint,
                "--output", out_path,
                "--temperature", str(temperature),
            ]
            if not raw:
                cmd.insert(1, "--session")
            if raw:
                cmd.extend(["--no-retry", "--no-silence-trim"])

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            elapsed = time.time() - t0

            if result.returncode != 0:
                err = result.stderr[:300] if result.stderr else "no stderr"
                print(f"FAILED ({elapsed:.1f}s): {err}")
            elif os.path.exists(out_path):
                audio, _ = sf.read(out_path)
                print(f"ok ({len(audio)/SR:.1f}s, {elapsed:.1f}s)")
            else:
                print(f"no output file ({elapsed:.1f}s)")


def analyze_directory(sample_dir, whisper_model, voices=None):
    """Analyze all WAV files. Returns (per_voice_results, all_clips)."""
    results = {}
    all_clips = []
    sample_path = Path(sample_dir)

    para_file = sample_path / "paragraphs.json"
    if not para_file.exists():
        print(f"ERROR: {para_file} not found — can't do text comparison")
        sys.exit(1)

    with open(para_file) as f:
        para_map = json.load(f)

    voice_dirs = sorted(d for d in sample_path.iterdir() if d.is_dir())
    if voices:
        voice_dirs = [d for d in voice_dirs if d.name in voices]

    for voice_dir in voice_dirs:
        voice = voice_dir.name
        results[voice] = []
        wavs = sorted(voice_dir.glob("*.wav"))
        total = len(wavs)

        for wi, wav in enumerate(wavs):
            stem = wav.stem
            expected = para_map.get(stem)
            if not expected:
                print(f"  SKIP {voice}/{wav.name} — no matching paragraph")
                continue

            print(f"  [{wi+1}/{total}] {voice}/{wav.name}...", end=" ", flush=True)
            clip = analyze_clip(wav, expected, whisper_model)
            clip["voice"] = voice
            results[voice].append(clip)
            all_clips.append(clip)

            # Status line
            n_art = len(clip["artifacts"])
            n_werr = len(clip["word_errors"])
            parts = []
            if n_art > 0:
                parts.append(f"{n_art} pause artifact{'s' if n_art > 1 else ''}")
            if n_werr > 0:
                parts.append(f"{n_werr} word error{'s' if n_werr > 1 else ''}")
            if parts:
                print(f"ISSUES: {', '.join(parts)}")
            else:
                print("CLEAN")

    return results, all_clips


def print_timeline(clip):
    """Print detailed word-by-word timeline for a clip."""
    fname = os.path.basename(os.path.dirname(clip["file"])) + "/" + os.path.basename(clip["file"])
    print(f"\n  {fname}  ({clip['duration_s']}s)")
    print(f"  Expected: \"{clip['expected_text'][:80]}{'...' if len(clip['expected_text']) > 80 else ''}\"")
    print(f"  Whisper:  \"{clip['whisper_text'][:80]}{'...' if len(clip['whisper_text']) > 80 else ''}\"")

    if clip.get("error"):
        print(f"  ERROR: {clip['error']}")
        return

    # Word errors
    if clip["word_errors"]:
        print(f"  Word errors ({clip['edit_distance']}):")
        for we in clip["word_errors"]:
            if we["op"] == "sub":
                print(f"    SUB: \"{we['ref']}\" → \"{we['hyp']}\"")
            elif we["op"] == "del":
                print(f"    DEL: \"{we['ref']}\" (missing)")
            elif we["op"] == "ins":
                print(f"    INS: \"{we['hyp']}\" (extra)")

    # Gap timeline (only non-normal gaps to keep it readable)
    flagged = [g for g in clip["gaps"] if g["judgment"] != "normal"]
    if flagged:
        print(f"  Flagged gaps ({len(flagged)}):")
        for g in flagged:
            marker = "⚠ SUSPECT" if g["judgment"] == "suspect" else "✗ ARTIFACT"
            print(
                f"    {g['start_s']:>6.2f}s  {g['duration_ms']:>4d}ms  "
                f"{g['context']:<16s} '{g['after']}' → '{g['before']}'  [{marker}]"
            )


def print_report(results, all_clips, label=""):
    """Print full quality report."""
    header = f"\n{'='*70}\nQUALITY REPORT"
    if label:
        header += f": {label}"
    header += f"\n{'='*70}"
    print(header)

    total = len(all_clips)
    if total == 0:
        print("  No clips analyzed.")
        return

    # Overall stats
    n_clean = sum(1 for c in all_clips if not c["has_artifact"])
    n_artifact = total - n_clean
    rate = n_artifact / total * 100

    print(f"\n  Total clips:  {total}")
    print(f"  Clean:        {n_clean} ({n_clean/total*100:.0f}%)")
    print(f"  With issues:  {n_artifact} ({rate:.0f}%)")

    # Artifact breakdown
    pause_clips = sum(1 for c in all_clips if "pause" in c.get("artifact_types", []))
    werr_clips = sum(1 for c in all_clips if "word_error" in c.get("artifact_types", []))
    both_clips = sum(1 for c in all_clips if "pause" in c.get("artifact_types", []) and "word_error" in c.get("artifact_types", []))

    if n_artifact > 0:
        print(f"\n  Breakdown:")
        print(f"    Pause artifacts:  {pause_clips} clips")
        print(f"    Word errors:      {werr_clips} clips")
        if both_clips:
            print(f"    Both:             {both_clips} clips")

    # Total pause artifacts
    all_artifacts = []
    for c in all_clips:
        all_artifacts.extend(c.get("artifacts", []))
    if all_artifacts:
        durations = [a["duration_ms"] for a in all_artifacts]
        print(f"\n  Pause artifact stats ({len(all_artifacts)} total):")
        print(f"    Median: {int(np.median(durations))}ms, "
              f"Mean: {int(np.mean(durations))}ms, "
              f"Max: {max(durations)}ms")
        # By context
        by_ctx = {}
        for a in all_artifacts:
            ctx = a["context"]
            by_ctx.setdefault(ctx, []).append(a["duration_ms"])
        for ctx, durs in sorted(by_ctx.items()):
            print(f"    {ctx}: {len(durs)} (median {int(np.median(durs))}ms)")

    # Gap distribution across ALL clips (not just artifacts)
    all_within = [g["duration_ms"] for c in all_clips for g in c.get("gaps", []) if g["context"] == "within_phrase"]
    all_comma = [g["duration_ms"] for c in all_clips for g in c.get("gaps", []) if g["context"] == "comma"]
    all_sent = [g["duration_ms"] for c in all_clips for g in c.get("gaps", []) if g["context"] == "sentence_end"]

    print(f"\n  Gap distribution (all clips):")
    if all_within:
        print(f"    Within phrase:    n={len(all_within)}, median={int(np.median(all_within))}ms, "
              f"p90={int(np.percentile(all_within, 90))}ms, max={max(all_within)}ms")
    if all_comma:
        print(f"    At comma:         n={len(all_comma)}, median={int(np.median(all_comma))}ms, "
              f"p90={int(np.percentile(all_comma, 90))}ms, max={max(all_comma)}ms")
    if all_sent:
        print(f"    Sentence boundary: n={len(all_sent)}, median={int(np.median(all_sent))}ms, "
              f"p90={int(np.percentile(all_sent, 90))}ms, max={max(all_sent)}ms")

    # WER stats
    wers = [c["wer"] for c in all_clips if c.get("wer") is not None]
    if wers:
        perfect = sum(1 for w in wers if w == 0)
        print(f"\n  Word accuracy:")
        print(f"    Perfect transcription: {perfect}/{len(wers)} ({perfect/len(wers)*100:.0f}%)")
        if any(w > 0 for w in wers):
            print(f"    Mean WER (imperfect only): {np.mean([w for w in wers if w > 0]):.1%}")

    # Per-voice table
    print(f"\n  Per-voice:")
    print(f"  {'Voice':<10} {'Clips':>6} {'Clean':>6} {'Pauses':>7} {'WErr':>5} {'Rate':>7}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*7}")
    for voice in sorted(results.keys()):
        clips = results[voice]
        n = len(clips)
        clean = sum(1 for c in clips if not c["has_artifact"])
        pauses = sum(1 for c in clips if "pause" in c.get("artifact_types", []))
        werrs = sum(1 for c in clips if "word_error" in c.get("artifact_types", []))
        issue_rate = (n - clean) / n * 100 if n > 0 else 0
        marker = " <<<" if issue_rate > 30 else ""
        print(f"  {voice:<10} {n:>6} {clean:>6} {pauses:>7} {werrs:>5} {issue_rate:>6.0f}%{marker}")

    # Detailed timelines for clips with artifacts
    problem_clips = [c for c in all_clips if c["has_artifact"]]
    if problem_clips:
        print(f"\n{'─'*70}")
        print(f"FLAGGED CLIPS ({len(problem_clips)})")
        print(f"{'─'*70}")
        for c in problem_clips:
            print_timeline(c)


def print_comparison(dir_a, dir_b, label_a, label_b, whisper_model):
    """Compare two benchmark runs."""
    print(f"Analyzing {label_a}...")
    results_a, clips_a = analyze_directory(dir_a, whisper_model)
    print(f"\nAnalyzing {label_b}...")
    results_b, clips_b = analyze_directory(dir_b, whisper_model)

    print_report(results_a, clips_a, label_a)
    print_report(results_b, clips_b, label_b)

    rate_a = sum(1 for c in clips_a if c["has_artifact"]) / len(clips_a) * 100 if clips_a else 0
    rate_b = sum(1 for c in clips_b if c["has_artifact"]) / len(clips_b) * 100 if clips_b else 0

    print(f"\n{'='*70}")
    print(f"COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*70}")
    print(f"  {label_a}: {rate_a:.0f}% issue rate")
    print(f"  {label_b}: {rate_b:.0f}% issue rate")

    all_voices = sorted(set(list(results_a.keys()) + list(results_b.keys())))
    print(f"\n  {'Voice':<10} {label_a:>12} {label_b:>12} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for voice in all_voices:
        ca = results_a.get(voice, [])
        cb = results_b.get(voice, [])
        ra = sum(1 for c in ca if c["has_artifact"]) / len(ca) * 100 if ca else 0
        rb = sum(1 for c in cb if c["has_artifact"]) / len(cb) * 100 if cb else 0
        delta = rb - ra
        print(f"  {voice:<10} {ra:>11.0f}% {rb:>11.0f}% {delta:>+7.0f}%")

    # Gap distribution comparison
    for label, clips in [(label_a, clips_a), (label_b, clips_b)]:
        within = [g["duration_ms"] for c in clips for g in c.get("gaps", []) if g["context"] == "within_phrase"]
        if within:
            print(f"\n  {label} within-phrase gaps: median={int(np.median(within))}ms, p90={int(np.percentile(within, 90))}ms, max={max(within)}ms")


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark for Holler checkpoints")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint path for generation")
    parser.add_argument("--analyze-dir", "-a", help="Analyze existing samples (skip generation)")
    parser.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), help="Compare two runs")
    parser.add_argument("--compare-labels", nargs=2, default=["A", "B"])
    parser.add_argument("--label", "-l", default="", help="Label for this run")
    parser.add_argument("--holler-bin", default="./holler", help="Path to holler CLI")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--voices", nargs="+", help="Voices to test (default: all 6)")
    parser.add_argument("--json", help="Save results to JSON")
    parser.add_argument("--quick", action="store_true", help="Quick run (3 paragraphs)")
    parser.add_argument("--raw", action="store_true",
                        help="Raw synthesis (--text, no retry/silence-trim). Default: --session mode with guardrails.")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-medium-mlx",
                        help="Whisper model for transcription")
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1],
                         args.compare_labels[0], args.compare_labels[1],
                         args.whisper_model)
        return

    if args.analyze_dir:
        print(f"Analyzing {args.analyze_dir}...")
        results, all_clips = analyze_directory(args.analyze_dir, args.whisper_model,
                                                voices=args.voices)
        print_report(results, all_clips, args.label or args.analyze_dir)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(all_clips, f, indent=2, default=str)
            print(f"\nResults saved to {args.json}")
        return

    if not args.checkpoint:
        parser.error("Need --checkpoint or --analyze-dir or --compare")

    label = args.label or Path(args.checkpoint).name
    output_dir = args.output_dir or os.path.expanduser(f"~/Downloads/holler-bench/{label}")
    paragraphs = PARAGRAPHS[:3] if args.quick else PARAGRAPHS
    mode_str = "raw --text (no guardrails)" if args.raw else "--session (with guardrails)"

    print(f"Benchmark: {label}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {mode_str}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_dir}")
    print(f"Paragraphs: {len(paragraphs)}, Voices: {args.voices or VOICES}")
    print(f"Whisper: {args.whisper_model}")
    print()

    # Generate
    print("=" * 70)
    print(f"GENERATING SAMPLES ({mode_str}, temp={args.temperature})")
    print("=" * 70)
    generate_samples(
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        holler_bin=args.holler_bin,
        paragraphs=paragraphs,
        voices=args.voices or VOICES,
        raw=args.raw,
        temperature=args.temperature,
    )

    # Analyze
    print()
    print("=" * 70)
    print("ANALYZING (Whisper word timestamps)")
    print("=" * 70)
    results, all_clips = analyze_directory(output_dir, args.whisper_model,
                                            voices=args.voices)
    print_report(results, all_clips, label)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_clips, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
