#!/usr/bin/env python3
"""Build combined multi-voice train.jsonl for Katie + Joe.
Rewrites audio/ref_audio paths to absolute on-instance paths and tags each
sample with `voice_name`. Output is ready for Qwen3-TTS tokenize step.

Run on the INSTANCE after both datasets are uploaded.
"""
import json
import os
import sys

# Expected layout on instance after upload:
#   /workspace/training-data/katie/audio/clip_XXXX.wav
#   /workspace/training-data/katie/ref.wav
#   /workspace/training-data/katie/train.jsonl
#   /workspace/training-data/joe/...

BASE = "/workspace/training-data"
VOICES = [
    ("katie", os.path.join(BASE, "katie")),
    ("joe",   os.path.join(BASE, "joe")),
]

out_path = os.path.join(BASE, "train_multivoice.jsonl")
count = 0
per_voice = {}

with open(out_path, "w") as out:
    for voice_name, voice_dir in VOICES:
        jsonl_path = os.path.join(voice_dir, "train.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"ERROR: missing {jsonl_path}", file=sys.stderr)
            sys.exit(1)

        n = 0
        for line in open(jsonl_path):
            entry = json.loads(line)
            audio_rel = entry["audio"].lstrip("./")
            ref_rel = entry["ref_audio"].lstrip("./")
            rewritten = {
                "audio": os.path.join(voice_dir, audio_rel),
                "text": entry["text"],
                "ref_audio": os.path.join(voice_dir, ref_rel),
                "voice_name": voice_name,
            }
            out.write(json.dumps(rewritten) + "\n")
            count += 1
            n += 1
        per_voice[voice_name] = n
        print(f"  {voice_name}: {n} samples")

print(f"\nWrote {count} combined samples → {out_path}")
print(f"Per voice: {per_voice}")
