#!/bin/bash
# Run on the Vast instance after SSH is up. Assumes /workspace exists.
# Installs deps into a venv, clones Qwen3-TTS, downloads models.
set -e

VENV=/workspace/.venv
PIP="$VENV/bin/pip"
PY="$VENV/bin/python3"

echo "=== Creating venv ==="
python3 -m venv "$VENV"
$PIP install -q -U pip

echo "=== Installing torch 2.6 + torchaudio ==="
$PIP install -q torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing qwen-tts + deps ==="
$PIP install -q -U qwen-tts huggingface_hub

echo "=== Installing build deps for flash-attn ==="
$PIP install -q wheel setuptools

echo "=== Installing flash-attn 2.7.3 (MAX_JOBS=4) ==="
MAX_JOBS=4 $PIP install -q flash-attn==2.7.3 --no-build-isolation --no-cache-dir || echo "flash-attn build failed; training will fall back to sdpa"

echo "=== Installing sox (needed by tokenizer) ==="
apt-get install -y -qq sox libsox-fmt-all 2>/dev/null

echo "=== Installing safetensors ==="
$PIP install -q safetensors

echo "=== Cloning Qwen3-TTS ==="
cd /workspace
if [ ! -d Qwen3-TTS ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi

echo "=== Downloading 0.6B-Base ==="
mkdir -p models
$PY -m huggingface_hub.commands.hf_cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base --quiet

echo "=== Downloading Tokenizer-12Hz ==="
$PY -m huggingface_hub.commands.hf_cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Tokenizer-12Hz --quiet

echo "=== Creating dataset root ==="
mkdir -p training-data

echo "=== Setup done ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
echo "Python: $($PY --version)"
echo "Torch: $($PY -c 'import torch; print(torch.__version__)')"
