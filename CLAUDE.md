# Holler

Open-source American English voice pack for Qwen3-TTS 0.6B. By Sentium.

**What this is:** A fine-tuned Qwen3-TTS-12Hz-0.6B model with 30 high-quality American English voices, optimized for local inference on Apple Silicon via mlx-audio. ~139ms TTFA streaming, ~1.7GB RAM at 6-bit. The fastest high-quality local TTS available on Mac.

**What this is not:** A new TTS architecture. This is a fine-tune of Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) with better English voices and a fully open training pipeline. All credit for the base model goes to the Qwen team at Alibaba. Our contribution is the voices, the Mac-focused inference setup, and the open training pipeline.

**License:** The base Qwen3-TTS model is Apache 2.0. Our fine-tune scripts and voices will also be Apache 2.0. All releases must include proper attribution to Qwen/Alibaba.

**Focus:** Mac. Local inference on Apple Silicon specifically. The training can happen on any CUDA GPU, but the inference target is mlx-audio on M-series Macs.

**Origin:** Started as the voice component of [ivi](https://ivi.computer), a macOS notch AI assistant by Sentium. We open-sourced it because Qwen3-TTS is SOTA for local inference but ships with only 2 mediocre English voices — and nothing else fills that gap.

**Session logs:** All holler session logs go in the parent ivi repo at `ivi/logs/`, not in `holler/logs/`. Old session logs have been moved there already. `holler/logs/runs/` still holds raw training/inference output logs.

## Current State (2026-04-26)

- **Recipe:** Proven. lr=1e-7, 2 epochs, text_projection patch only. Now with `--save_every_steps` for fractional epoch checkpoints.
- **Katie v6 (current production):** Clean. 6-bit affine g64, RTF 0.38, TTFA 139ms.
- **Katie v8 training data (ready):** 452 curated clips, 31.3 min, enhanced (ClearVoice→DeepFilter→RecipeE→presence boost). At `voices/katie/training-data/train_curated.jsonl`. Mixed prosody: 18% emotional, 6% questions, 76% statements. Next step: rent GPU and train.
- **Quantization:** 6-bit affine g64 is the pick.
  - `checkpoints/katie-v6/` — bf16, 1.7GB disk (source for quantization)
  - `checkpoints/quant-experiment/affine-6bit-g64/` — **6-bit affine (the pick)**, 1094MB disk, 1.7GB Metal RAM, RTF 0.38, TTFA 139ms
- **Multi-voice (Katie+Joe, v7):** bf16 only at `checkpoints/katie-joe-v7/`. Quality issues. Not production-grade.
- **Voices designed:** 2/30 (Katie, Joe). 28 more needed.
- **Inference runtime:** Custom fast inference server (`inference/server.py`). RTF 0.38, TTFA 139ms on 6-bit.
- **Training data tools:** Full pipeline automated — `tools/regenerate_rejects.py`, `tools/enhance_clips.py`, `tools/trim_and_merge.py`, `tools/curate_clips.py`.
- **Python venv:** `.venv` (Python 3.13, torch 2.6, torchaudio 2.6, mlx-audio, clearvoice, deepfilternet, noisereduce, scipy, pyloudnorm).

## Structure

```
holler/
├── CLAUDE.md              — This file (project instructions + state)
├── .gitignore
├── training/              — Training scripts (single-voice + multi-voice SFT)
├── inference/             — MLX + PyTorch inference, TTFA benchmarks, live demo
│   ├── server.py                   — Fast inference server (--checkpoint, --port flags)
│   ├── experiment_quant_methods.py — Quantize + benchmark all quant variants
│   ├── benchmark_katie_v6.py       — bf16 benchmark (10 texts, TTFA + peaks)
│   ├── benchmark_quant_proper.py   — all-quant benchmark with MLX Metal memory
│   ├── benchmark_sentence_queue.py — sentence-queue benchmark simulating ivi pattern
│   ├── live_demo.py                — Web UI: type text, hear it spoken
│   ├── test_mlx_inference.py       — basic MLX test
│   ├── test_mlx_streaming.py       — streaming TTFA test
│   ├── test_mlx_multivoice.py      — multi-voice test
│   ├── test_pytorch_inference.py   — GPU ground-truth test
│   └── test_epoch_sweep.py         — compare multiple epoch checkpoints
├── tools/                 — Voice design, training data generation, verification
├── voices/                — Per-voice reference audio + training data
│   ├── katie/             — Female voice (Cartesia-sourced, slot 3000)
│   │   ├── ref.wav        — 10s reference audio
│   │   ├── cartesia_original.wav — original source
│   │   └── training-data/ — 385 clips + train.jsonl
│   │       ├── audio/           — enhanced clips (ClearVoice→DeepFilter→E)
│   │       └── audio-original/  — pre-enhancement backups
│   └── joe/               — Male voice (VoiceDesign-sourced, slot 3001)
│       ├── ref.wav
│       ├── candidates/    — 28 voice design candidates + index.txt
│       └── training-data/ — 452 curated clips (v2) + train_curated.jsonl
│           ├── audio/           — enhanced clips (ClearVoice→DeepFilter→RecipeE→presence)
│           ├── audio-original/  — pre-enhancement backups (v1 originals)
│           ├── audio-enhanced-backup/ — pre-enhancement of emotional batch
│           ├── curation.json    — Tinder decisions (452 keep, 23 reject)
│           └── train_curated.jsonl — TRAINING FILE (452 entries)
├── checkpoints/           — Model checkpoints (not in git — large)
│   ├── katie-v6/          — 1.7GB bf16 (source for quantization)
│   ├── katie-v6-4bit/     — 960MB 4-bit affine (previous pick)
│   ├── katie-joe-v7/      — 2.3GB bf16, multi-voice, quality unresolved
│   └── quant-experiment/  — All quantization variants tested 2026-04-24
│       ├── affine-6bit-g64/ — 1094MB ← THE PICK
│       ├── affine-4bit-g64/ — 960MB (baseline comparison)
│       ├── affine-4bit-g32/, affine-4bit-g128/ — group size variants
│       ├── affine-3bit-g64/ — 893MB (BROKEN — EOS lost)
│       ├── affine-8bit-g64/ — 1210MB
│       ├── mxfp4/, nvfp4/, mxfp8/ — alternative formats
├── samples/               — Audio samples organized by version/voice/precision
│   ├── quant-experiment/  — A/B samples from quantization experiment
│   ├── benchmark-katie-v6-{bf16,4bit}/ — older benchmark clips
│   ├── v6-mlx/, v6-pytorch/, v5/   — earlier samples
│   └── v7-{mlx,pytorch}-{katie,joe}/ — multi-voice samples
├── logs/                  — DEPRECATED: session logs now live in ivi repo at ivi/logs/
│   ├── sessions/          — Old session logs (moved to ivi/logs/)
│   └── runs/              — Raw training/inference logs
└── docs/
    ├── ivi-session-notes.md        — Full debugging journey + v7 addendum
    └── handover-lora-failure.md    — Historical: why LoRA doesn't work
```

## The Winning Recipe

```bash
# Standard (epoch-only checkpoints)
python3 sft_12hz_patched.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /path/to/output \
  --train_jsonl /path/to/train_curated.jsonl \
  --batch_size 2 --lr 1e-7 --num_epochs 2 \
  --speaker_name katie

# Fine-grained (fractional epoch checkpoints every 45 steps ≈ 0.2 epochs)
python3 sft_12hz_patched.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /path/to/output \
  --train_jsonl /path/to/train_curated.jsonl \
  --batch_size 2 --lr 1e-7 --num_epochs 2 \
  --speaker_name katie --save_every_steps 45
```

With 452 clips at batch_size=2: 226 steps/epoch. `--save_every_steps 45` gives checkpoints at ~0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 epochs. Each checkpoint is a full model copy (~1.7GB bf16), so budget ~17GB disk.

- **lr=1e-7 is critical.** Higher LRs (2e-6, 2e-5) destroy EOS token — model generates until max_new_tokens.
- **Only patch needed:** Wrap `text_embedding` with `text_projection` on line ~89 of upstream `sft_12hz.py` (fixes 0.6B dimension mismatch). Script: `training/sft_12hz_patched.py`.
- **Do NOT apply** the "double label shift" fix or "remove sub-codebook loop" fix at this LR — they break training.
- **Epoch 1 was the pick for v6.** With v8's improved data, sweep 0.6–1.4 to find optimal.
- **Loss stays at ~12-13.** That's correct. Low loss at higher LR = overfitting, not quality.

## Quantization

Done via **mlx-audio's own converter** (NOT `mlx_lm.convert` — that doesn't support qwen3_tts model type):

```python
from mlx_audio.convert import convert
convert(
    hf_path='checkpoints/katie-v6',
    mlx_path='checkpoints/quant-experiment/affine-6bit-g64',
    quantize=True, q_bits=6, q_group_size=64, q_mode='affine',
)
```

- Affine quantization: per-group (64 weights) scale + zero-point, weights stored as N-bit integers
- The converter **selectively keeps critical layers at full precision** (codec_embedding, speaker embeddings, small layers). That's why "4-bit" averages 8.9 bits/weight and "8-bit" averages 11.4 bits/weight.
- MLX dequantizes on-the-fly in Metal GPU kernels during matmul — no separate unpack step.
### Quantization Experiment Results (2026-04-24, M1 Pro)

Tested all MLX quantization modes against Katie v6 bf16 source. Script: `inference/experiment_quant_methods.py`.

| Variant | Disk | RAM | TTFA | RTF | Bits/wt | Notes |
|---------|------|-----|------|-----|---------|-------|
| **affine 6-bit g64** | **1094M** | **1744M** | **84ms** | **0.67x** | **10.1** | **THE PICK — best voice presence** |
| affine 4-bit g64 | 960M | 1611M | 77ms | 0.63x | 8.9 | Previous pick. Still good, less presence |
| affine 4-bit g32 | 994M | 1644M | 77ms | 0.65x | 9.2 | Finer groups, one EOS overshoot |
| affine 4-bit g128 | 943M | 1594M | 80ms | 0.66x | 8.7 | Coarser groups, slightly slower |
| nvfp4 | 960M | 1611M | 78ms | 0.65x | 8.9 | Clean, competitive with affine 4-bit |
| mxfp4 | 943M | 1594M | 82ms | 0.66x | 8.7 | EOS issue on "Okay." (6s generated) |
| mxfp8 | 1210M | 1861M | 90ms | 0.84x | 11.2 | Worst RTF, TTFA spikes, EOS issues |
| affine 3-bit g64 | 893M | 1544M | 80ms | 0.64x | 8.3 | BROKEN — all samples clip to 1.0, EOS lost |

**6-bit affine wins on voice quality.** Noticeably more presence and vocal dynamics than 4-bit. Cost: +134MB RAM, +0.04x RTF — negligible for the quality gain. 3-bit is broken. mxfp8 is surprisingly worse than affine 4-bit despite more bits.

**Important:** Use `mx.metal.get_active_memory()` / `mx.metal.get_peak_memory()` for measuring MLX memory — NOT `psutil.Process().memory_info().rss` which gives garbage numbers for MLX workloads.

## Inference Architecture (updated 2026-04-24)

**Holler is a model, not an inference library.** Ship weights on HuggingFace, users run via our fast inference server (`inference/server.py`). Uses mlx-audio for model loading and codec decoding, with a custom generate loop for 2.3x faster streaming.

### Quick Start

```bash
# 1. Create venv and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download checkpoint (or use local)
# TODO: huggingface-cli download sentium/holler-0.6b-6bit --local-dir checkpoints/holler-6bit

# 3. Run server (default checkpoint or specify one)
python3 inference/server.py
python3 inference/server.py --checkpoint path/to/checkpoint --port 8100
# → http://localhost:8100

# 4. Test
curl "http://localhost:8100/tts?text=Hello+world" -o test.wav
curl http://localhost:8100/benchmark
```

### API

```
POST /tts        — streaming float32 PCM (24kHz mono), chunked transfer encoding
  Body: {"text": "...", "voice": "katie", "temperature": 0.6, "n_codebooks": 12}
GET  /tts?text=  — returns WAV file
GET  /benchmark  — runs 6-sentence benchmark, returns text report
GET  /health     — {"status": "ok"}
```

### Performance (M1 Pro, Katie v6 6-bit affine)

| Metric | Value |
|--------|-------|
| RTF (streaming) | **0.38 avg**, 0.46 max |
| TTFA | **139ms avg** |
| Throughput | 2.6x real-time |
| Metal RAM (idle) | 1.7 GB |
| CPU during generation | ~8% |
| Model on disk | 1094 MB |

### Why not just use mlx-audio directly?

mlx-audio's `model.generate(stream=True, streaming_interval=0.1)` gives RTF ~0.76. Our server gives RTF ~0.34. The difference:
1. mlx-audio calls `mx.eval()` + `mx.clear_cache()` after every streaming chunk (Metal pipeline thrashing)
2. We use a custom generate loop with one `mx.eval()` per token, chunked decode with two-phase TTFA
3. We use 12 of 16 codebooks by default (configurable), skipping highest-frequency acoustic detail
4. Full details: see `RESEARCH.md` (27 experiments logged)

## Inference Pipeline

**Stack:** `inference/server.py` → custom generate loop → mlx-audio model/codec → MLX → Metal GPU

1. `mlx_audio.tts.load(checkpoint_path)` loads model + speech tokenizer
2. Custom generate loop: talker LLM → code predictor → codec tokens at 12Hz
3. Chunked streaming decode: first chunk at 3 tokens (~120ms TTFA), then 40-token chunks
4. HTTP chunked transfer encoding streams float32 PCM to client
5. `mx.clear_cache()` after each complete generation to prevent memory accumulation

**Live demo:** `inference/live_demo.py` — HTTP server on port 8099, type text in browser, audio plays from Mac speakers.

**Python venv:** `.venv` exists (Python 3.13). Has mlx-audio for inference, plus clearvoice/deepfilternet/noisereduce for audio enhancement. torch 2.6 + torchaudio 2.6 (pinned for DeepFilterNet compatibility).

## Known: 220ms Leading Silence

The first 2-3 streaming chunks (~220ms) from any Qwen3-TTS generation are near-silence. This is a **known codec LM architecture behavior** — confirmed across Qwen3-TTS, CosyVoice, VALL-E descendants. The codec decoder needs initial context tokens before producing meaningful audio.

Our training data also has 25-212ms of leading silence per clip, which reinforces this behavior. Future training runs should trim leading silence from training clips.

**Mitigations (not yet implemented):**
- Trim leading silence from training data before next training run
- For ivi integration: sentence-level streaming from LLM overlaps codec warmup with text generation
- Study `rekuenkdr/Qwen3-TTS-streaming` — two-phase streaming fork that buffers past the silence before first emit (208ms first audible vs 570ms baseline)

## Training Data Pipeline (updated 2026-04-25)

### Step 0: Enhance reference audio FIRST

Before cloning training data, run the reference audio through the enhancement pipeline. This is critical — source audio quality (especially from services like Cartesia) often has noise, peaks, and artifacts that the 1.7B model will faithfully clone into every training clip. A/B test the enhanced ref vs original and pick the better one. Enhanced ref saved alongside original (e.g. `ref_enhanced.wav`).

### Step 1: Generate clips via 1.7B voice cloning

Use `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` with the (enhanced) reference audio. Temperature 0.9 recommended (1.0 causes clipping in ~30% of clips). Append 1s silence for cutoff detection (will be trimmed).

### Step 2: Trim silence

100ms lead + 100ms trail padding, 20ms fade-out. Reject clips with abrupt cutoffs (peak > 0.03 in last 50ms before the 1s silence pad). Auto-retry once on cutoff. Script: `tools/trim_and_merge.py`.

### Step 3: Enhance clips

Peak normalize clippers only (>= 0.999 peak → scale to -1dB), then run through enhancement pipeline. Script: `tools/enhance_clips.py`.

### Step 4: Curate via Clip Tinder

`tools/curate_clips.py --voice <name>` — swipe through clips, keep/reject/maybe. Regenerate rejects with `tools/regenerate_rejects.py`.

### Enhancement Pipeline: ClearVoice → DeepFilterNet3 → Recipe E

Applied to both reference audio and training clips.

1. **ClearVoice MossFormer2_SE_48K** — Alibaba's 50M-param speech enhancer. Uses spectral masking (NOT vocoding — preserves voice identity). Outputs at 48kHz, resample back to 24kHz via `signal.resample_poly`.
2. **DeepFilterNet3** — 2.1M-param neural denoiser. Fine-grained noise removal in 2-6kHz range (ear fatigue zone). Resample 24k→48k→24k via `signal.resample_poly`.
3. **Recipe E** — 80Hz high-pass filter (scipy butter order 4) + 30% **stationary** spectral denoise (noisereduce). `stationary=True` is critical — without it, noisereduce is too aggressive and "boxes in" the sound.
4. **Peak normalize clippers only** — only scale down clips where peak >= 0.999. Don't normalize everything (reduces overall volume by ~1dB for no reason).

**Why this order:** ClearVoice does broad spectral cleanup (understands speech structure), DeepFilterNet does fine-grained residual noise, Recipe E catches low-frequency rumble. Each pass is complementary. Reversed order (DF→CV→E) sounds worse.

**What was tested and rejected:** Resemble Enhance (needs deepspeed, destroys speaker similarity), VoiceFixer (checkpoint corrupt, uses vocoder), AudioSR (ancient deps), AP-BWE (manual weight download), spectral denoise alone at 80-100% (hollows voice), DSP with normalization+trim+de-essing (too much).

**Performance:** ~0.5-0.7s/clip on M1 Pro. All 770 clips in ~8 minutes.

**Result:** Cleaner audio, reduced 2-6kHz noise that causes ear fatigue on AirPods, voice character fully preserved. "Light years better than Cartesia" on same sentences.

**Dependencies:** holler `.venv` (Python 3.13, torch 2.6, torchaudio 2.6, clearvoice, deepfilternet, noisereduce, scipy). Note: torchaudio must be 2.6 (not newer) for DeepFilterNet compatibility.

## Key Technical Details

- Qwen3-TTS `codec_embedding.weight` has 3072 slots (1024-dim each). Slots 0-2047 are active codec tokens (DON'T overwrite). 3000-3071 is the 72-slot custom voice region.
- Fine-tuning writes a speaker embedding to a slot and co-trains the full model. At inference, voice name → config lookup → slot → embedding injection at codec position 6.
- Training data: ~385 clips per voice, voice-cloned from a reference through Qwen3-TTS-1.7B-Base. 24kHz mono WAV with 1s trailing silence.
- Multi-voice: same recipe, but JSONL has per-sample `voice_name` field, and training script tracks embeddings per voice. See `training/sft_12hz_multivoice.py`.
- Voice name in v6 config is `katie` at slot 3000 (nested under `talker_config.spk_id`).

## GPU Training Runbook

**Requirements:** 24GB+ VRAM, CUDA 12.x, bf16 support. A100 SXM4 40GB (~$0.56/hr) or RTX 4090 (~$0.30/hr) on Vast.ai. 80GB disk is enough for single-voice + all checkpoints (~17GB).

**Docker image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`

**SSH key:** `~/.ssh/runpod` works for Vast.ai instances.

### Step 1: Rent + Setup

```bash
# Rent instance (search for cheapest A100 or 4090)
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 rentable=true' -o 'dph_total' --limit 5
vastai create instance <ID> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 80

# Wait for running, get SSH details
vastai show instances

# Upload and run setup script
scp -i ~/.ssh/runpod -P <PORT> holler/training/remote_setup.sh root@<HOST>:/workspace/
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "bash /workspace/remote_setup.sh"
```

Setup installs into `/workspace/.venv`: torch 2.6, qwen-tts, flash-attn 2.7.3, sox. Downloads 0.6B-Base model + tokenizer. Clones Qwen3-TTS repo. Takes ~5 min.

### Step 2: Upload Training Data

```bash
# Create remote dir, upload ONLY what's needed
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "mkdir -p /workspace/training-data"
scp -i ~/.ssh/runpod -P <PORT> -r voices/<voice>/training-data/audio root@<HOST>:/workspace/training-data/
scp -i ~/.ssh/runpod -P <PORT> voices/<voice>/training-data/ref.wav root@<HOST>:/workspace/training-data/
scp -i ~/.ssh/runpod -P <PORT> voices/<voice>/training-data/train_curated.jsonl root@<HOST>:/workspace/training-data/
scp -i ~/.ssh/runpod -P <PORT> holler/training/sft_12hz_patched.py root@<HOST>:/workspace/

# Verify checksums — entire audio folder + JSONL must match local
# Local (macOS):
cd voices/<voice>/training-data && find audio -name '*.wav' -type f | sort | xargs md5 -q | md5 -q && md5 -q train_curated.jsonl
# Remote:
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "cd /workspace/training-data && find audio -name '*.wav' -type f | sort | xargs md5sum | md5sum && md5sum train_curated.jsonl"
```

Both hashes must match (note: md5 vs md5sum output format differs, compare the hex digest only). Do NOT upload backup dirs (`audio-original/`, `audio-enhanced-backup/`).

### Step 3: Train

```bash
scp -i ~/.ssh/runpod -P <PORT> holler/training/remote_train.sh root@<HOST>:/workspace/
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "bash /workspace/remote_train.sh katie"
```

`remote_train.sh` handles tokenization, symlinks, and training. ~5 min for 452 clips on A100.

### Step 4: Verify with PyTorch Inference

```bash
# Upload sweep script, run on GPU
scp -i ~/.ssh/runpod -P <PORT> holler/inference/test_epoch_sweep.py root@<HOST>:/workspace/
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "/workspace/.venv/bin/python3 /workspace/test_epoch_sweep.py"

# Download samples
scp -i ~/.ssh/runpod -P <PORT> -r root@<HOST>:/workspace/samples/ ~/Downloads/
```

Check: voice sounds right, EOS terminates (no runaways), audio lengths proportional to text.

### Step 5: Download Winner + Quantize Locally

```bash
# Download checkpoint
scp -i ~/.ssh/runpod -P <PORT> -r root@<HOST>:/workspace/output/checkpoint-epoch-<N>/ holler/checkpoints/<voice>-v<X>/

# Destroy instance
vastai destroy instance <ID>

# Quantize on Mac (6-bit affine g64)
python3 -c "
from mlx_audio.convert import convert
convert(hf_path='holler/checkpoints/<voice>-v<X>', mlx_path='holler/checkpoints/<voice>-v<X>-6bit', quantize=True, q_bits=6, q_group_size=64, q_mode='affine')
"
```

### Hard-Won Environment Lessons

- **flash-attn 2.7.3** works with torch 2.6. Version 2.8.3 does NOT (ABI symbol mismatch).
- **Never install into system python.** Always venv. `--force-reinstall` on system python cascades into torch version hell.
- **Install `wheel` + `setuptools` before flash-attn** — it builds from source and needs them.
- **`sox` must be installed via apt** — `prepare_data.py` needs the binary, not a Python package.
- **sdpa is a valid fallback** if flash-attn won't build. Training produces identical results. Inference should use flash_attention_2 when available.
- **JSONL relative paths** (`./audio/`, `./ref.wav`) resolve from cwd. Symlink into the working directory.

## Community References

- **rekuenkdr** — anonymous hobbyist, posted the winning lr=1e-7 recipe (Issue #39), built the two-phase streaming fork (72 stars). Also works on "OVA" (local voice assistant pipeline).
- **Our GitHub comment** documenting findings: https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999

## HuggingFace Release Target

- `sentium/holler-0.6b` (bf16 — full precision, source for custom quantization)
- `sentium/holler-0.6b-6bit` (affine 6-bit g64 — the pick, best quality-to-size ratio)

## Hard-Won Lessons

**Training:**
- **LR is the dominant knob.** Not loss, not epochs, not community patches.
- **EOS termination is the diagnostic.** If PyTorch inference hits max_new_tokens on a short sentence, the model is broken.
- **Always verify with PyTorch inference on GPU first.** It's the ground truth. MLX issues are separable from training issues.
- **Loss decreasing ≠ quality.** Loss 1.0 produced noise; loss 12.8 produced clean voice.
- **ref_mel shape is [1, T, 128]** after upstream .transpose(1,2), NOT [1, 128, T]. Pad dim=1 for multi-voice.
- **Use mlx-audio's converter, not mlx_lm** — mlx_lm doesn't support the qwen3_tts model type.

**Inference (2026-04-24 research session — see RESEARCH.md):**
- **mlx-audio's streaming mode is the bottleneck, not the model.** `mx.eval()` + `mx.clear_cache()` per chunk thrashes Metal pipeline. Custom generate loop = 2.3x faster.
- **Code predictor is 71% of generation time.** 15 sequential 5-layer transformer calls per speech token. The main 28-layer talker is only 24%.
- **12 of 16 codebooks is the sweet spot.** Codebooks 13-16 are highest-frequency acoustic detail. Skipping them = 18% faster, same EOS reliability (96% vs 98%), negligible quality loss.
- **Two-phase streaming:** first chunk at 3 tokens (~120ms TTFA), then 40-token chunks. Balances latency vs decode overhead.
- **`mx.clear_cache()` once after generation, not during.** Prevents memory accumulation without hurting performance.
- **psutil RSS is garbage for MLX memory** — use `mx.get_active_memory()`.
- **6-bit affine is the quantization sweet spot for TTS.** Noticeably more vocal presence than 4-bit. 4-bit still good, 3-bit destroys EOS. Novel finding — no prior audio model quant benchmarks exist.
- **Leading silence is architectural** — all codec LMs do it. Trim at inference or overlap with LLM streaming.
- **Hann crossfade at chunk boundaries:** tested 10ms/20ms/40ms overlap. No audible difference — streaming decoder's internal state already handles continuity.
- **3-bit quantization destroys EOS.** Model generates 4-25s for short sentences, all samples clip to 1.0. Same failure mode as high LR during training.
- **mxfp8 is worse than affine 6-bit** despite more bits (11.2 vs 10.1). Slower RTF (0.84x vs 0.67x), TTFA spikes, EOS issues. Float format doesn't help here.
- **mxfp4 has EOS issues on short utterances.** "Okay." generated 6s. nvfp4 is cleaner but no quality advantage over affine.
- **Always benchmark through server.py**, never mlx-audio's `model.generate()`. The latter shows RTF 0.63-0.84x; our server shows 0.38x. The difference is `mx.eval()`+`mx.clear_cache()` per chunk.

## What's NOT Known / Unresolved

- Whether joint training at 30 voices holds up (only tested at 2)
- Root cause of v7 quality issues (Katie noise, Joe clipping) — see docs/ivi-session-notes.md "Observations" section
- Whether sequential training (one voice at a time, cumulative checkpoints) works better than joint
- Optimal voice count per training run
- Whether mixed precision (e.g. higher bits for code_predictor, lower for talker) could improve quality at same average bits
- Whether trimming leading silence from training data reduces the 220ms codec warmup
- EOS failure ~2-4% of the time (model-level, both 12cb and 16cb) — mitigated by safety cap but not eliminated
- Server crashes during long idle — needs process supervisor for production

## What's Next

1. ~~**Wire Katie v6 into ivi**~~ — ✅ DONE. `inference/server.py` + `sidecar/tts-sidecar-fast.py`. RTF 0.38, TTFA 139ms (6-bit).
2. ~~**Try alternative quantization**~~ — ✅ DONE (2026-04-24). Tested 8 variants: affine 3/4/6/8-bit, mxfp4, mxfp8, nvfp4, multiple group sizes. 6-bit affine g64 wins on voice quality. See Quantization section.
3. **Enhance training audio** — IN PROGRESS (2026-04-24). ClearVoice→DeepFilter→E pipeline applied to all 770 clips (Katie + Joe). Audio quality enhanced. Still need manual curation: listen to all clips, remove bad prosody/wonky ones (~30% estimated). Automated filters can't catch prosody issues — only ears can.
3b. **Retrain Katie+Joe with curated+enhanced data** — blocked on manual curation (step 3).
4. **Solve multi-voice quality** — v7 Katie+Joe has issues (short utterances generate silence). Root cause still unknown.
5. **Design and train more voices** — 28 more needed. Names and characters in `voices/VOICES.md`.
6. **Standardize GPU runbook** — from-scratch instance setup → training → quantization → 6-bit conversion
7. **HuggingFace release** under `sentium/` with full docs — bf16 + 6-bit affine only
8. **PR to mlx-audio** — reduce `mx.clear_cache()` frequency in their streaming loop. Easy win for the community.
