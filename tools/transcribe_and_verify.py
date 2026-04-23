"""Runs on Mac after inference samples are downloaded. Transcribes every WAV via
openai-whisper and compares to the expected text. Flags any significant mismatch.

Usage:
  python3 transcribe_and_verify.py --manifest ~/Downloads/ivi-v7-samples/manifest.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

def normalize(t):
    import re
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip()


def wer(ref, hyp):
    # Simple word error rate via edit distance
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_size", default="base.en")
    args = ap.parse_args()

    import whisper
    print(f"Loading whisper model '{args.model_size}'...")
    model = whisper.load_model(args.model_size)

    rows = []
    for line in open(args.manifest):
        entry = json.loads(line)
        if "error" in entry:
            rows.append({**entry, "transcribed": None, "wer": None})
            continue
        audio_path = entry["path"]
        if not Path(audio_path).exists():
            print(f"  skip (missing file): {audio_path}")
            continue
        result = model.transcribe(audio_path, language="en", fp16=False)
        transcribed = result["text"].strip()
        w = wer(normalize(entry["text"]), normalize(transcribed))
        rows.append({**entry, "transcribed": transcribed, "wer": w})
        flag = "⚠" if w > 0.2 else "✓"
        print(f"  {flag} [{entry['voice']}] {entry['label']} WER={w:.2f}")
        print(f"     expected:    {entry['text']}")
        print(f"     transcribed: {transcribed}")

    # Summary
    by_voice = {}
    for r in rows:
        if r.get("wer") is None:
            continue
        by_voice.setdefault(r["voice"], []).append(r["wer"])

    print("\n=== Summary ===")
    for voice, wers in sorted(by_voice.items()):
        avg = sum(wers) / len(wers)
        bad = sum(1 for w in wers if w > 0.2)
        print(f"  {voice}: {len(wers)} clips | avg WER {avg:.2f} | {bad} clips >0.2 WER")

    # Write verdict
    verdict_path = Path(args.manifest).parent / "transcription_verdict.jsonl"
    with open(verdict_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nVerdict: {verdict_path}")


if __name__ == "__main__":
    main()
