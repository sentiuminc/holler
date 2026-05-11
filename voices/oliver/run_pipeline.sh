#!/bin/bash
set -e
cd ~/Desktop/Files/AI/ivi/holler

REF_TEXT="Okay, so I looked into it and here is what I found. The file you were working on got saved to your Downloads folder, not your Desktop. Want me to move it over, or would you rather keep it where it is?"

echo "=== STEP 1: Generate 500 training clips ==="
.venv/bin/python tools/generate_training_data.py \
  --voice oliver \
  --ref-text "$REF_TEXT"

echo ""
echo "=== STEP 2: Enhance with skip-notch ==="
.venv-enhance-audio/bin/python tools/enhance_clean.py \
  --input voices/oliver/training-data/audio-original \
  --output voices/oliver/training-data/audio \
  --skip-notch

echo ""
echo "=== STEP 3: Full analysis on enhanced clips ==="
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py \
  --source voices/oliver/training-data/audio \
  --label "oliver-enhanced-final" \
  --n 100

echo ""
echo "=== PIPELINE COMPLETE ==="
echo "Raw clips: voices/oliver/training-data/audio-original/"
echo "Enhanced clips: voices/oliver/training-data/audio/"
echo "Manifest: voices/oliver/training-data/train.jsonl"
