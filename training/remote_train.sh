#!/bin/bash
# Run on the Vast instance after setup + data upload.
# Assumes remote_setup.sh has been run (venv, models, Qwen3-TTS repo exist).
#
# Single-voice usage:
#   bash remote_train.sh katie
#
# Multi-voice usage:
#   bash remote_train.sh --multi
#
# Expects:
#   /workspace/.venv/                  (created by remote_setup.sh)
#   /workspace/training-data/          (audio/, ref.wav, train_curated.jsonl — uploaded by you)
#   /workspace/Qwen3-TTS/finetuning/  (repo cloned by remote_setup.sh)
#   /workspace/models/0.6B-Base/      (downloaded by remote_setup.sh)
#   /workspace/models/Tokenizer-12Hz/ (downloaded by remote_setup.sh)
#   /workspace/sft_12hz_patched.py    (uploaded by you)
set -e

PY=/workspace/.venv/bin/python3
VOICE="${1:-katie}"

cd /workspace/Qwen3-TTS/finetuning

# Symlink training data so relative paths in JSONL resolve
ln -sf /workspace/training-data/ref.wav ./ref.wav
ln -sf /workspace/training-data/audio ./audio

echo "=== Tokenizing audio (adds audio_codes to each sample) ==="
$PY prepare_data.py \
  --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
  --input_jsonl /workspace/training-data/train_curated.jsonl \
  --output_jsonl /workspace/training-data/train_with_codes.jsonl

cp /workspace/sft_12hz_patched.py .

echo ""
echo "=== Training $VOICE (lr=1e-7, 2 epochs, fractional checkpoints) ==="
$PY sft_12hz_patched.py \
  --init_model_path /workspace/models/0.6B-Base \
  --output_model_path /workspace/output \
  --train_jsonl /workspace/training-data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 1e-7 \
  --num_epochs 2 \
  --speaker_name "$VOICE" \
  --save_every_steps 45

echo ""
echo "=== Training complete ==="
ls -la /workspace/output/
