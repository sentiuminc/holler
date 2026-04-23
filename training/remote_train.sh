#!/bin/bash
# Run on the Vast instance after data is uploaded. Assumes:
#   /workspace/training-data/katie/  (audio/, ref.wav, train.jsonl)
#   /workspace/training-data/joe/    (audio/, ref.wav, train.jsonl)
#   /workspace/Qwen3-TTS/finetuning/ (repo cloned)
#   /workspace/models/0.6B-Base/ + /workspace/models/Tokenizer-12Hz/
#   Local helper scripts uploaded to /workspace/: build_combined_jsonl.py, sft_12hz_multivoice.py
set -e

cd /workspace

echo "=== Building combined multi-voice JSONL ==="
python3 build_combined_jsonl.py

echo ""
echo "=== Tokenizing audio (adds audio_codes to each sample) ==="
cp sft_12hz_multivoice.py Qwen3-TTS/finetuning/
cd Qwen3-TTS/finetuning
python3 prepare_data.py \
  --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
  --input_jsonl  /workspace/training-data/train_multivoice.jsonl \
  --output_jsonl /workspace/training-data/train_multivoice_with_codes.jsonl

echo ""
echo "=== Kicking off multi-voice training (lr=1e-7, 2 epochs) ==="
python3 sft_12hz_multivoice.py \
  --init_model_path /workspace/models/0.6B-Base \
  --output_model_path /workspace/output \
  --train_jsonl /workspace/training-data/train_multivoice_with_codes.jsonl \
  --batch_size 2 \
  --lr 1e-7 \
  --num_epochs 2 \
  --voice_slot_map_json '{"katie": 3000, "joe": 3001}'

echo ""
echo "=== Training complete ==="
ls -la /workspace/output/
