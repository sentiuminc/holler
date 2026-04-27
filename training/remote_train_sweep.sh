#!/bin/bash
# Sweep multi-voice training across different dataset sizes.
# Expects subsets already uploaded to /workspace/training-data/
set -e

PY=/workspace/.venv/bin/python3
cd /workspace/Qwen3-TTS/finetuning

# Symlink per-voice dirs
ln -sf /workspace/training-data/nora ./nora
ln -sf /workspace/training-data/joe ./joe

for N in 50 100 200; do
  JSONL="/workspace/training-data/multivoice_nora_joe_${N}.jsonl"
  echo ""
  echo "=========================================="
  echo "=== ${N} clips per voice ==="
  echo "=========================================="

  echo "--- Tokenizing ---"
  $PY prepare_data.py \
    --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
    --input_jsonl "$JSONL" \
    --output_jsonl "/workspace/training-data/train_${N}_codes.jsonl"

  cp /workspace/sft_12hz_multivoice.py .

  echo "--- Training (1 epoch only for speed) ---"
  $PY sft_12hz_multivoice.py \
    --init_model_path /workspace/models/0.6B-Base \
    --output_model_path "/workspace/output-${N}" \
    --train_jsonl "/workspace/training-data/train_${N}_codes.jsonl" \
    --batch_size 2 \
    --lr 1e-7 \
    --num_epochs 1 \
    --voice_slot_map_json '{"nora":3002,"joe":3001}'

  echo "--- Done: ${N} clips ---"
  ls -la "/workspace/output-${N}/"
done

echo ""
echo "=== All sweeps complete ==="
