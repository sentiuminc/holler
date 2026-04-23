# Handover: Qwen3-TTS 0.6B Custom Voice Fine-Tuning

**Date:** 2026-04-21
**Status:** Ready for full SFT training run. Training data generated. Previous LoRA attempt failed. Need to rent GPU and run official pipeline.

## Goal

Fine-tune Qwen3-TTS 0.6B Base with a custom female voice for ivi. The voice comes from a Cartesia-generated reference audio clip. After fine-tuning, convert to MLX 8-bit and run locally at 77ms TTFA.

## What's Done

### Training Data (READY — do not regenerate)

**Location:** `~/Downloads/ivi-tts-training-data/`
- `train.jsonl` — 385 entries, each with `audio`, `text`, `ref_audio` fields
- `audio/clip_0001.wav` through `clip_0385.wav` — 24kHz mono WAV, 1s silence appended to each
- `ref.wav` — 10s reference audio (Cartesia female voice, 24kHz, padded)
- Total audio: 34.5 minutes
- Content: diverse conversational text — short responses, questions, technical, emotional, nature, science, stories, etc.

**How it was made:** Used `Qwen3-TTS-12Hz-1.7B-Base-8bit` (MLX, local) to voice-clone from a Cartesia reference clip. Each text was generated with `model.generate(text=..., ref_audio=ref_v2_24k.wav, ref_text="Hey, so I've been thinking...", language="en", temperature=0.6)`. Non-streaming mode. Scripts: `audio-test/generate_training_data.py` (batch 1, 160 clips) and `audio-test/generate_training_data_batch2.py` (batch 2, 225 clips).

**Reference audio source:** `/Users/nagy/Downloads/cartesia_audio_2026-04-21T12_36_35+04_00.wav` (9.6s, 44.1kHz). Resampled to 24kHz, 100ms front pad, 300ms back pad with fade-out → saved as `ref_v2_24k.wav`.

**Ref text (exact transcript):** "Hey, so I've been thinking about this for a while now. The thing is, you don't really notice how much it matters until you actually try it yourself. It's one of those subtle differences that just clicks."

### Previous Attempt: LoRA on RTX 4060 Ti 16GB — FAILED

- Full SFT OOMed on 16GB (AdamW optimizer states for 900M params)
- Switched to LoRA (1.3% params trainable) as workaround
- Loss converged nicely (12.6 → 3.4 over 10 epochs) but output sounded nothing like target voice
- **Why it failed:** LoRA only modifies attention/MLP weights. Voice identity requires co-training the speaker embedding (codec_embedding slot 3000) with the full model. LoRA doesn't touch codec_embedding. The embedding was injected manually at inference but wasn't co-trained.
- **Lesson:** Full SFT is the correct and only proven approach for Qwen3-TTS custom voices.

### Instance (DESTROYED)

The RTX 4060 Ti instance (35359349) has been destroyed by the user. Need to rent a new GPU.

## What To Do Now

### Step 1: Rent a GPU

**Requirements:**
- 24GB+ VRAM (full SFT on 0.6B OOMed at 16GB)
- CUDA, bf16 support (Ampere or newer)
- Prefer RTX 4090, A100, or other modern cards (user explicitly doesn't want 3090)

**Vast.ai credentials:**
- API key: `c24e873e09b69b7fd09208bfe4b47320ee51981d3b57c81119371dfbe200a35d`
- CLI: `/Users/nagy/.local/bin/vastai` (already configured)
- SSH key: `~/.ssh/runpod`
- HuggingFace token: `REDACTED`

**Search command:**
```bash
vastai search offers 'gpu_ram>=24 num_gpus=1 dph_total<0.40 reliability>0.95 rentable=True verified=True' -o 'dph_total' --limit 15
```

### Step 2: Set Up Instance

```bash
# SSH in
ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=no -p <PORT> root@<HOST>

# Install deps
pip install transformers accelerate peft datasets soundfile librosa flash-attn --no-build-isolation

# Clone official repo (check if text_projection fix is in HEAD first!)
git clone https://github.com/QwenLM/Qwen3-TTS.git
pip install -e ./Qwen3-TTS

# Install sox
apt-get update -qq && apt-get install -y -qq sox libsox-fmt-all libsndfile1

# Download models
export HF_TOKEN=REDACTED
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='/workspace/models/0.6B-Base')
snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='/workspace/models/Tokenizer-12Hz')
"
```

### Step 3: Upload Training Data

```bash
rsync -avz -e "ssh -i ~/.ssh/runpod -o StrictHostKeyChecking=no -p <PORT>" \
  ~/Downloads/ivi-tts-training-data/ root@<HOST>:/workspace/training-data/
```

### Step 4: Tokenize

```bash
cd /workspace/training-data
python3 /workspace/Qwen3-TTS/finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path /workspace/models/Tokenizer-12Hz \
  --input_jsonl train.jsonl \
  --output_jsonl train_with_codes.jsonl
```

### Step 5: Check for Bugs BEFORE Training

**IMPORTANT:** Before applying any patches, check if the fixes are already in the current HEAD:

```bash
# Check text_projection fix (PR #188)
grep -n "text_projection" /workspace/Qwen3-TTS/finetuning/sft_12hz.py
# If line ~89 already has text_projection() call → fix is merged, no patch needed
# If it just has model.talker.model.text_embedding() without projection → need to patch

# Check default LR
grep -n "default=" /workspace/Qwen3-TTS/finetuning/sft_12hz.py | grep lr
# If default is 2e-5, override with --lr 2e-6 on command line
```

**Only apply the text_projection fix if it's NOT already in the code:**
```python
# Line ~89, change:
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
# To:
input_text_embedding = model.talker.text_projection(model.talker.model.text_embedding(input_text_ids)) * text_embedding_mask
```

### Step 6: Train (Full SFT)

```bash
cd /workspace/training-data

# Remove tensorboard requirement if present
sed -i 's/log_with="tensorboard"//' /workspace/Qwen3-TTS/finetuning/sft_12hz.py

PYTHONUNBUFFERED=1 nohup python3 /workspace/Qwen3-TTS/finetuning/sft_12hz.py \
  --init_model_path /workspace/models/0.6B-Base \
  --output_model_path /workspace/output \
  --train_jsonl /workspace/training-data/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name ivi_female \
  > /workspace/train.log 2>&1 &

# Monitor
tail -f /workspace/train.log
```

**Expected:** ~90s per epoch based on LoRA timing (full SFT will be slower but not dramatically). Total: maybe 20-40 minutes.

### Step 7: Download & Convert

The training script saves a full checkpoint per epoch at `/workspace/output/checkpoint-epoch-{N}/`. Each is a complete model (~1.2GB) with:
- `model.safetensors` — all weights including speaker embedding at slot 3000
- `config.json` — updated with `tts_model_type: "custom_voice"` and `spk_id: {"ivi_female": 3000}`

```bash
# Download best epoch (probably epoch-9 or test several)
rsync -avz -e "ssh -i ~/.ssh/runpod -p <PORT> -i ~/.ssh/runpod" \
  root@<HOST>:/workspace/output/checkpoint-epoch-9/ \
  ~/Downloads/ivi-tts-finetuned/
```

### Step 8: Convert to MLX & Test

This is the uncertain step — need to verify the right conversion command:
```bash
# Option A: mlx_lm.convert (may work if model is standard enough)
cd ~/Desktop/Files/AI/ivi/audio-test
.venv-tts-bench/bin/python -m mlx_lm.convert \
  --hf-path ~/Downloads/ivi-tts-finetuned/ \
  --mlx-path ~/Downloads/ivi-tts-finetuned-mlx/ \
  -q --q-bits 8

# Option B: mlx-audio may have its own converter — check docs
# Option C: manually convert safetensors (framework-agnostic format)
```

Then test with mlx-audio:
```python
from mlx_audio.tts import load
model = load("~/Downloads/ivi-tts-finetuned-mlx/")
for result in model.generate(text="Hello!", voice="ivi_female", temperature=0.6, stream=True, streaming_interval=0.1):
    play(result.audio)  # should be 77ms TTFA
```

## Key Technical Details

### How Qwen3-TTS Custom Voices Work
- `codec_embedding` table has ~4000 slots, each a 1024-dim vector
- CustomVoice model: speakers at known slots (Aiden=3001, etc.)
- Fine-tuning: trains ALL params, extracts speaker embedding from ref audio via `speaker_encoder`, writes it to slot 3000 at checkpoint save
- At inference: `speaker="ivi_female"` → lookup slot 3000 → inject embedding → generate

### Why Full SFT is Required
- Voice identity = speaker embedding + LLM weights co-trained together
- LoRA only touches 1.3% of weights, doesn't train codec_embedding
- All community success stories use full SFT
- Full SFT needs ~10GB for model + ~5GB for optimizer states + ~5-8GB for activations = ~20-23GB

### Training Hyperparameters (community-validated)
- LR: 2e-6 (NOT default 2e-5 which causes noise)
- Epochs: 10 (with double-label-shift bug fixed, might go higher)
- Batch size: 2
- Gradient accumulation: 4 (built into script)
- Mixed precision: bf16
- Gradient clipping: 1.0

### Multi-Speaker (Future)
Can train 6+ voices in one model. Each gets its own codec_embedding slot. JSONL entries would have per-speaker `ref_audio`. The Mozi EasyFinetuning tool supports this. Official script needs modification (track multiple speaker embeddings).

## Files Reference

| File | Description |
|------|-------------|
| `~/Downloads/ivi-tts-training-data/` | Training data (385 clips + manifest) |
| `~/Downloads/ivi-tts-training-data/ref.wav` | Reference audio (10s, 24kHz) |
| `audio-test/generate_training_data.py` | Batch 1 generator script |
| `audio-test/generate_training_data_batch2.py` | Batch 2 generator script |
| `audio-test/bench_clone_1.7b.py` | Voice clone test script |
| `audio-test/.venv-tts-bench/` | Python venv with mlx-audio installed |
| `docs/voice-tts-research.md` | Full TTS research document |

## Don't Repeat These Mistakes

1. **Don't use LoRA for Qwen3-TTS voice fine-tuning** — it doesn't train the codec_embedding table
2. **Don't rent <24GB VRAM** — full SFT on 0.6B OOMs at 16GB
3. **Don't use `language="en"`** in PyTorch API — use `language="english"` (full name)
4. **Don't forget PYTHONUNBUFFERED=1** when logging to files
5. **Don't apply patches without checking if they're already merged in HEAD**
6. **Don't generate test audio on the CUDA instance** — PyTorch inference is absurdly slow (~14 min per clip). Download model, convert to MLX, test locally.
7. **ref_text must exactly match ref_audio content** — mismatched transcript produces garbage clips
