#!/bin/bash
# Multi-voice training on remote GPU instance.
# Expects remote_setup.sh already run.
#
# Directory structure on remote:
#   /workspace/training-data/
#     nora/audio/*.wav, nora/ref.wav
#     joe/audio/*.wav, joe/ref.wav
#     multivoice_nora_joe.jsonl
#   /workspace/sft_12hz_multivoice.py
set -e

PY=/workspace/.venv/bin/python3

cd /workspace/Qwen3-TTS/finetuning

# Symlink per-voice dirs so JSONL relative paths resolve
ln -sf /workspace/training-data/nora ./nora
ln -sf /workspace/training-data/joe ./joe

echo "=== Tokenizing audio (adds audio_codes to each sample) ==="
$PY prepare_data.py \
  --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
  --input_jsonl /workspace/training-data/multivoice_nora_joe.jsonl \
  --output_jsonl /workspace/training-data/train_with_codes.jsonl

cp /workspace/sft_12hz_multivoice.py .

echo ""
echo "=== Multi-voice training: nora + joe (lr=1e-7, 2 epochs) ==="
$PY sft_12hz_multivoice.py \
  --init_model_path /workspace/models/0.6B-Base \
  --output_model_path /workspace/output \
  --train_jsonl /workspace/training-data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 1e-7 \
  --num_epochs 2 \
  --voice_slot_map_json '{"nora":3002,"joe":3001}'

echo ""
echo "=== Training complete ==="
ls -la /workspace/output/
