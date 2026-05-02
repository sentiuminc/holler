#!/usr/bin/env python3
"""Stress test for decoder stuttering on KV cache carryover.

Generates hundreds of multi-sentence sequences with KV cache carryover.
Automatically detects stutter patterns (repeated transients like "kkk")
and saves flagged audio for manual review.

Carryover flow in server.py:
  S1 (reset=True):  talker=fresh, decoder=fresh.  Cache NOT saved to global.
  S2 (reset=False): talker=fresh, decoder=S1 state. Cache saved to global.
  S3 (reset=False): talker=S2 cache, decoder=S1+S2. FULL carryover.

So true full carryover (talker + decoder) only starts at sentence 3+.
This script generates 4-sentence sequences and tests S2-S4 for stutter.

Usage:
    .venv/bin/python inference/stress_carryover.py
    .venv/bin/python inference/stress_carryover.py -c path/to/checkpoint --voice dakota -n 300
"""

import argparse
import json
import os
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as srv

SAMPLE_RATE = 24000

# Short sentences — trigger silence abort quickly, maximize carryover transitions
SHORT = [
    "Hey!", "Yes.", "Got it.", "No.", "Sure.", "OK.", "Right.", "Hmm.",
    "Wait.", "Yeah.", "Thanks.", "Fine.", "Wow.", "Oh.", "Great.", "Nope.",
    "Cool.", "True.", "Huh.", "Nice.",
]

# Longer sentences — produce meaningful speech onset for stutter analysis
LONG = [
    "What do you think about that?",
    "I completely agree with your assessment on this.",
    "Let me explain what I mean by that.",
    "That is not what I was trying to say at all.",
    "Could you tell me a little more about that?",
    "Keep going, I want to hear the rest of it.",
    "The problem here is really quite clear to me.",
    "But I think we actually need to reconsider this.",
    "Actually, now that I think about it more carefully.",
    "I just realized something really important about this.",
    "Pretty impressive work you have done here.",
    "Basically, the whole approach needs rethinking.",
    "Tell me what you think about this idea.",
    "Going back to your original question about it.",
    "Did you know that was even possible to do?",
    "The key insight is that simplicity always wins.",
    "People tend to overthink these kinds of problems.",
    "I believe the answer is simpler than we think.",
    "Consider what would happen if we tried it differently.",
    "That reminds me of something I read recently.",
]


def detect_stutter(audio, word_count, sr=SAMPLE_RATE):
    """Detect stutter by checking if audio is abnormally short for the text.

    Normal speech: ~80-120ms per word. A stuttered generation produces
    much less audio than expected (model hits EOS early or generates
    mostly silence that gets trimmed).

    Returns dict with detection signals:
      - short: True if audio duration is < 50% of expected
      - ratio: actual_duration / expected_duration
      - duration_s: actual audio duration
      - expected_s: expected duration based on word count
    """
    duration_s = len(audio) / sr
    expected_s = word_count * 0.10  # ~100ms per word is conservative floor
    ratio = duration_s / expected_s if expected_s > 0 else 999

    return {
        "short": ratio < 0.5,
        "ratio": round(ratio, 2),
        "duration_s": round(duration_s, 3),
        "expected_s": round(expected_s, 3),
    }


def save_wav(path, audio, sr=SAMPLE_RATE):
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def generate_sentence(model, text, voice, reset_decoder):
    """Generate one sentence, collecting all audio chunks."""
    chunks = []
    for chunk in srv.generate_audio(model, text, voice=voice,
                                    reset_decoder=reset_decoder):
        chunks.append(chunk)
    if chunks:
        return np.concatenate(chunks)
    return np.array([], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Stress test for carryover stuttering")
    parser.add_argument("-c", "--checkpoint", default=srv.DEFAULT_CHECKPOINT)
    parser.add_argument("-v", "--voice", default=None)
    parser.add_argument("-n", "--count", type=int, default=200,
                        help="Number of sequences to generate")
    parser.add_argument("-o", "--output", default="samples/stutter-debug",
                        help="Output directory for samples")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_audio.tts import load

    # Load model
    print(f"[stress] Loading {args.checkpoint}...", flush=True)
    t0 = time.time()
    model = load(args.checkpoint)
    mx.set_cache_limit(2 * 1024 ** 3)

    config = model.config.talker_config
    eos = config.codec_eos_token_id
    srv.suppress_indices_cache = mx.array(
        [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos],
        dtype=mx.int32
    )
    srv.zero_token_cache = mx.zeros((1, 1), dtype=mx.int32)

    spk_id = getattr(config, 'spk_id', None)
    voices = sorted(spk_id.keys()) if spk_id and isinstance(spk_id, dict) else ["default"]
    voice = args.voice or voices[0]

    # Warmup
    print(f"[stress] Warming up (voice={voice})...", flush=True)
    for _ in srv.generate_audio(model, "Hello world.", voice=voice, max_tokens=50):
        pass
    model.speech_tokenizer.decoder.reset_streaming_state()
    mx.clear_cache()
    srv._carry_over_state = {}

    print(f"[stress] Ready in {time.time() - t0:.1f}s", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous run
    for f in output_dir.glob("*.wav"):
        f.unlink()
    for f in output_dir.glob("*.json"):
        f.unlink()

    flagged = []
    all_results = []
    total_sentences_tested = 0
    total_sequences = 0

    print(f"[stress] Generating {args.count} sequences of 4 sentences each", flush=True)
    print(f"[stress] Saving ALL carryover sentences (S3-S4 = full carryover)", flush=True)
    print(flush=True)

    for seq_idx in range(args.count):
        # Build a 4-sentence sequence: 1 short starter + 3 longer sentences
        texts = [random.choice(SHORT)]
        texts += random.sample(LONG, min(3, len(LONG)))

        # Reset everything between sequences
        srv._carry_over_state = {}
        model.speech_tokenizer.decoder.reset_streaming_state()
        mx.clear_cache()

        sentence_audios = []
        skip_sequence = False

        for s_idx, text in enumerate(texts):
            reset = (s_idx == 0)
            audio = generate_sentence(model, text, voice, reset_decoder=reset)

            if len(audio) < 100:
                skip_sequence = True
                break

            sentence_audios.append((text, audio, s_idx))

        if skip_sequence or len(sentence_audios) < 2:
            continue

        total_sequences += 1

        # Test S2, S3, S4 for stutter
        for text, audio, s_idx in sentence_audios[1:]:
            total_sentences_tested += 1
            carryover_type = "decoder-only" if s_idx == 1 else "full"
            word_count = len(text.split())

            detection = detect_stutter(audio, word_count)

            entry = {
                "sequence": seq_idx,
                "sentence_pos": s_idx + 1,
                "carryover_type": carryover_type,
                "text": text,
                "word_count": word_count,
                **detection,
            }
            all_results.append(entry)

            # Save combined WAV for ALL full-carryover sentences (S3-S4)
            # and flagged decoder-only sentences
            should_save = (carryover_type == "full") or detection["short"]

            if should_save:
                pause = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
                parts = []
                for i, (_, a, _) in enumerate(sentence_audios[:s_idx + 1]):
                    if i > 0:
                        parts.append(pause)
                    parts.append(a)
                combined = np.concatenate(parts)

                tag = "SHORT" if detection["short"] else "ok"
                fname = (f"seq{seq_idx:03d}_s{s_idx + 1}_{carryover_type}"
                         f"_{detection['ratio']:.1f}x_{tag}.wav")
                save_wav(output_dir / fname, combined)

            if detection["short"]:
                flagged.append(entry)
                print(f"  SHORT #{len(flagged):3d}  ratio={detection['ratio']:.2f}  "
                      f"dur={detection['duration_s']:.2f}s  [{carryover_type}]  "
                      f"\"{text[:50]}\"", flush=True)

        if (seq_idx + 1) % 50 == 0:
            print(f"[stress] {seq_idx + 1}/{args.count} sequences, "
                  f"{len(flagged)} short / {total_sentences_tested} tested", flush=True)

    # Summary
    durations = [r["duration_s"] for r in all_results]
    ratios = [r["ratio"] for r in all_results]

    print(f"\n{'=' * 65}")
    print(f"  CARRYOVER STUTTER STRESS TEST")
    print(f"{'=' * 65}")
    print(f"  Sequences:       {total_sequences}")
    print(f"  Sentences tested:{total_sentences_tested} (S2-S4)")
    print(f"  Abnormally short:{len(flagged)} ({len(flagged) / max(total_sentences_tested, 1) * 100:.1f}%)")
    print(f"  Duration range:  {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"  Ratio range:     {min(ratios):.2f}x - {max(ratios):.2f}x")
    print(f"  Voice:           {voice}")
    print(f"  Output:          {output_dir}/")

    if flagged:
        decoder_only = [f for f in flagged if f["carryover_type"] == "decoder-only"]
        full = [f for f in flagged if f["carryover_type"] == "full"]
        print(f"\n  Short breakdown:")
        print(f"    decoder-only (S2): {len(decoder_only)}")
        print(f"    full carryover (S3-S4): {len(full)}")
    else:
        print(f"\n  No abnormally short generations detected.")

    # Sort all results by ratio (shortest first) for manual review
    all_results.sort(key=lambda r: r["ratio"])

    n_saved = len(list(output_dir.glob("*.wav")))
    print(f"\n  Saved {n_saved} WAV files for manual review.")
    print(f"  Shortest 10 by duration/word ratio:")
    for r in all_results[:10]:
        print(f"    ratio={r['ratio']:.2f}  dur={r['duration_s']:.2f}s  "
              f"[{r['carryover_type']}]  \"{r['text'][:50]}\"")

    print(f"{'=' * 65}")

    meta_path = output_dir / "results.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "total_sequences": total_sequences,
            "total_tested": total_sentences_tested,
            "flagged_short": flagged,
            "all_results_sorted": all_results[:50],
            "voice": voice,
            "checkpoint": args.checkpoint,
        }, f, indent=2)
    print(f"\n  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
