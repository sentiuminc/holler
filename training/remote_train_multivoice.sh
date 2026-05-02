#!/bin/bash
# Multi-voice training on remote GPU instance.
# Expects remote_setup.sh already run + training data uploaded.
#
# Usage:
#   bash remote_train_multivoice.sh kit:3000 dakota:3001 nora:3002 joe:3003
#
# Each arg is voice_name:slot. The combined JSONL should already be at
# /workspace/training-data/train_multivoice.jsonl (built locally by
# training/build_combined_jsonl.py, then uploaded).
#
# Directory structure on remote:
#   /workspace/training-data/
#     kit/audio/*.wav, kit/ref.wav
#     dakota/audio/*.wav, dakota/ref.wav
#     ...
#     train_multivoice.jsonl
#   /workspace/sft_12hz_multivoice.py
set -e

if [ $# -eq 0 ]; then
  echo "Usage: bash remote_train_multivoice.sh voice:slot [voice:slot ...]"
  echo "Example: bash remote_train_multivoice.sh kit:3000 dakota:3001 nora:3002 joe:3003"
  exit 1
fi

PY=/workspace/.venv/bin/python3

# Build voice_slot_map JSON and symlinks from args
SLOT_MAP="{"
FIRST=1
for arg in "$@"; do
  voice="${arg%%:*}"
  slot="${arg##*:}"

  if [ $FIRST -eq 1 ]; then FIRST=0; else SLOT_MAP+=","; fi
  SLOT_MAP+="\"${voice}\":${slot}"

  # Symlink so JSONL relative paths resolve from finetuning dir
  ln -sf /workspace/training-data/"$voice" /workspace/Qwen3-TTS/finetuning/"$voice"
  echo "  $voice → slot $slot"
done
SLOT_MAP+="}"

cd /workspace/Qwen3-TTS/finetuning

echo "=== Tokenizing audio (adds audio_codes to each sample) ==="
$PY prepare_data.py \
  --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
  --input_jsonl /workspace/training-data/train_multivoice.jsonl \
  --output_jsonl /workspace/training-data/train_with_codes.jsonl

cp /workspace/sft_12hz_multivoice.py .

echo ""
echo "=== Multi-voice training (lr=1e-7, 2 epochs) ==="
echo "  Voice slots: $SLOT_MAP"
$PY sft_12hz_multivoice.py \
  --init_model_path /workspace/models/0.6B-Base \
  --output_model_path /workspace/output \
  --train_jsonl /workspace/training-data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 1e-7 \
  --num_epochs 2 \
  --voice_slot_map_json "$SLOT_MAP"

echo ""
echo "=== Training complete ==="
ls -la /workspace/output/
