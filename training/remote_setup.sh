#!/bin/bash
# Run on the Vast instance after SSH is up. Assumes /workspace exists.
# Installs deps, clones Qwen3-TTS, downloads models.
set -e

echo "=== Installing qwen-tts + deps ==="
pip install -q -U qwen-tts huggingface_hub

echo "=== Installing flash-attn (MAX_JOBS=4) ==="
MAX_JOBS=4 pip install -q -U flash-attn --no-build-isolation || echo "flash-attn install may have warnings; continuing"

echo "=== Cloning Qwen3-TTS ==="
cd /workspace
if [ ! -d Qwen3-TTS ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi

echo "=== Downloading 0.6B-Base ==="
mkdir -p models
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base --quiet

echo "=== Downloading Tokenizer-12Hz ==="
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Tokenizer-12Hz --quiet

echo "=== Creating dataset root ==="
mkdir -p training-data/katie training-data/joe

echo "=== Setup done ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
