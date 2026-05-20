#!/bin/bash
# Vast.ai instance setup for Holler: model training (SFT).
# Run once after SSH is up. Assumes /workspace exists (Vast.ai default).
#
# Pulls cached pip wheels + models from R2 first, so pip installs
# and model downloads are local (no PyPI/HuggingFace dependency).
#
#   ssh root@<instance> 'bash -s' < remote_setup.sh
#
set -e

echo "══════════════════════════════════════════════════════"
echo "  Holler — Remote Instance Setup"
echo "══════════════════════════════════════════════════════"
echo

VENV=/workspace/.venv
PIP="$VENV/bin/pip"
PY="$VENV/bin/python3"

# ── System tools ──────────────────────────────────────────────────────
echo "=== Installing system tools ==="
apt-get update -qq && apt-get install -y -qq bmon nvtop htop sox libsox-fmt-all unzip

# ── rclone + R2 config (needed early for cache pull) ─────────────────
echo "=== Installing rclone ==="
if ! command -v rclone &>/dev/null; then
  curl -sSL https://rclone.org/install.sh | bash
fi

if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ]; then
  echo "ERROR: R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set"
  echo "  export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=..."
  exit 1
fi

mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<CONF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://bc32378f7dcfe01a255b7a152f9c2319.r2.cloudflarestorage.com
no_check_bucket = true
CONF

# ── Pull cached dependencies from R2 ────────────────────────────────
cd /workspace
mkdir -p models

echo "=== Pulling cached deps from R2 ==="
rclone copy r2:holler/instance-cache/ /workspace/_cache/ --transfers 8 --stats 10s --stats-one-line 2>&1

if [ -f /workspace/_cache/pip-cache.tar.gz ]; then
  echo "  Restoring pip cache..."
  mkdir -p /root/.cache
  tar xzf /workspace/_cache/pip-cache.tar.gz -C /root/.cache/
  echo "  pip cache restored"
fi

if [ -f /workspace/_cache/models-cache.tar.gz ]; then
  echo "  Restoring models..."
  tar xzf /workspace/_cache/models-cache.tar.gz -C /workspace/
  echo "  models restored"
fi

rm -rf /workspace/_cache/

# ── Python venv ───────────────────────────────────────────────────────
echo "=== Creating venv ==="
python3 -m venv "$VENV"
$PIP install -q -U pip

# ── PyTorch (cached → instant, otherwise downloads from PyTorch index)
echo "=== Installing torch 2.6 + torchaudio ==="
$PIP install -q torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ── Training deps (cached → instant) ────────────────────────────────
echo "=== Installing qwen-tts + training deps ==="
$PIP install -q -U qwen-tts huggingface_hub safetensors
$PIP install -q wheel setuptools

echo "=== Installing flash-attn 2.7.3 (cached = instant, otherwise builds ~5 min) ==="
MAX_JOBS=4 $PIP install -q flash-attn==2.7.3 --no-build-isolation

# ── Models (skip if already restored from cache) ────────────────────
export HF_TOKEN="REDACTED"

if [ ! -f models/0.6B-Base/model.safetensors ]; then
  echo "=== Downloading 0.6B-Base (not in cache) ==="
  $VENV/bin/hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base --quiet
else
  echo "=== 0.6B-Base: cached ✓ ==="
fi

if [ ! -f models/Tokenizer-12Hz/model.safetensors ]; then
  echo "=== Downloading Tokenizer-12Hz (not in cache) ==="
  $VENV/bin/hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Tokenizer-12Hz --quiet
else
  echo "=== Tokenizer-12Hz: cached ✓ ==="
fi

# ── Qwen3-TTS repo (for prepare_data.py + dataset.py) ───────────────
echo "=== Cloning Qwen3-TTS ==="
if [ ! -d Qwen3-TTS ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi
ln -sf /workspace/Qwen3-TTS/finetuning/dataset.py /workspace/dataset.py

# ── CUDA MPS ─────────────────────────────────────────────────────────
echo "=== Enabling CUDA MPS ==="
nvidia-cuda-mps-control -d 2>/dev/null || true
echo start_server | nvidia-cuda-mps-control 2>/dev/null \
  && echo "  CUDA MPS enabled" \
  || echo "  ⚠ CUDA MPS not available — will time-slice"

# ── Training data from R2 ────────────────────────────────────────────
mkdir -p training-data output

echo "=== Downloading training data from R2 ==="
rclone copy r2:holler/training-data/ training-data/ --transfers 32 --stats 5s --stats-one-line 2>&1

echo "  Verifying..."
for voice in kit dakota nora joe oliver tessa; do
  n=$(ls training-data/$voice/audio/*.wav 2>/dev/null | wc -l)
  echo "  $voice: $n clips, ref=$([ -f training-data/$voice/ref.wav ] && echo 'YES' || echo 'NO')"
done
echo "  JSONL: $(wc -l < training-data/train_multivoice.jsonl) entries"

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
echo "  1. Upload training script: scp training/sft_12hz_multivoice.py root@\$(hostname):/workspace/"
echo "  2. Train:                  bash /workspace/remote_train_multivoice.sh kit:3000 dakota:3001 nora:3002 joe:3003 oliver:3004 tessa:3005"
echo "  3. Download checkpoint:    rclone copy from R2 or rsync from instance"
echo
