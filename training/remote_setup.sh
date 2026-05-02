#!/bin/bash
# Vast.ai instance setup for Holler: model training (SFT).
# Run once after SSH is up. Assumes /workspace exists (Vast.ai default).
#
#   ssh root@<instance> 'bash -s' < remote_setup.sh
#
set -e

echo "══════════════════════════════════════════════════════"
echo "  Holler — Remote Instance Setup"
echo "══════════════════════════════════════════════════════"
echo

# ── System tools ──────────────────────────────────────────────────────
echo "=== Installing system tools ==="
apt-get update -qq && apt-get install -y -qq bmon nvtop htop sox libsox-fmt-all

VENV=/workspace/.venv
PIP="$VENV/bin/pip"
PY="$VENV/bin/python3"

# ── Python venv ───────────────────────────────────────────────────────
echo "=== Creating venv ==="
python3 -m venv "$VENV"
$PIP install -q -U pip

# ── PyTorch (pinned for flash-attn compat) ────────────────────────────
echo "=== Installing torch 2.6 + torchaudio ==="
$PIP install -q torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ── Training: qwen-tts + flash-attn ──────────────────────────────────
echo "=== Installing qwen-tts + training deps ==="
$PIP install -q -U qwen-tts huggingface_hub safetensors
$PIP install -q wheel setuptools

echo "=== Installing flash-attn 2.7.3 (takes a few minutes) ==="
MAX_JOBS=4 $PIP install -q flash-attn==2.7.3 --no-build-isolation --no-cache-dir \
  || echo "  ⚠ flash-attn build failed; training will fall back to sdpa"

# ── Models ────────────────────────────────────────────────────────────
cd /workspace
mkdir -p models

echo "=== Downloading 0.6B-Base (training starting checkpoint) ==="
$VENV/bin/hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base --quiet

echo "=== Downloading Tokenizer-12Hz ==="
$VENV/bin/hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Tokenizer-12Hz --quiet

# ── Qwen3-TTS repo (training scripts reference) ──────────────────────
echo "=== Cloning Qwen3-TTS ==="
if [ ! -d Qwen3-TTS ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi

# ── CUDA MPS (enables true parallel execution across processes) ───────
echo "=== Enabling CUDA MPS ==="
nvidia-cuda-mps-control -d 2>/dev/null || true
echo start_server | nvidia-cuda-mps-control 2>/dev/null \
  && echo "  CUDA MPS enabled (multi-process parallel GPU execution)" \
  || echo "  ⚠ CUDA MPS not available — multi-process will time-slice"

# ── Directories ───────────────────────────────────────────────────────
mkdir -p training-data voices output

# ── Status ────────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════════"
echo "  Setup complete"
echo "══════════════════════════════════════════════════════"
echo
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Disk:    $(df -h /workspace | tail -1 | awk '{print $4 " free / " $2 " total"}')"
echo "Python:  $($PY --version 2>&1 | awk '{print $2}')"
echo "Torch:   $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $($PY -c 'import torch; print(torch.version.cuda)')"
echo
echo "Next steps:"
echo "  1. Upload training data:  scp -r voices/<name>/training-data root@\$(hostname):/workspace/training-data/<name>/"
echo "  2. Upload training scripts: scp training/sft_12hz_multivoice.py root@\$(hostname):/workspace/"
echo "  3. Train:                 bash /workspace/remote_train_multivoice.sh"
echo "  4. Download checkpoint:   scp -r root@\$(hostname):/workspace/checkpoints/ ~/Downloads/"
echo
