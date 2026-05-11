# Holler Training Runbook

Authoritative doc for training Qwen3-TTS 0.6B custom voices. Covers single-voice and multi-voice SFT, GPU setup, data pipeline, quantization, and all hard-won lessons.

## Training Recipe

**Model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
**Method:** Full-model SFT (not LoRA — LoRA doesn't work, see `docs/handover-lora-failure.md`)
**Hyperparameters:**
- **Single-voice:** lr=1e-7, batch_size=2, gradient_accumulation=4, AdamW weight_decay=0.01, bf16
- **Multi-voice (6+):** lr=5e-7 with cosine warmup (10% of steps), same batch/accum/optimizer. lr=1e-7 is too low for multi-voice — model can't learn distinct voice patterns.
**Epochs:** 2 (pick best by ear). 3 epochs overfit (tested 2026-05-11). With `--save_every_steps 45` you get fractional checkpoints.
**Loss:** 12-15 at lr=1e-7, 12-14 at lr=5e-7. Low loss at higher LR = overfitting, not quality.
**Cosine scheduler (FIXED 2026-05-11):** `total_optimizer_steps = (steps_per_epoch × num_epochs) // grad_accum`. The Accelerator steps the scheduler once per optimizer step (every `grad_accum` batches). Without this fix, the scheduler sees raw batch count and only completes ~25% of its cosine curve — lr barely decays.
**Embedding normalization (added 2026-05-11):** L2-normalize all ECAPA-TDNN speaker embeddings to magnitude 10.0 before injection. Strips loudness from embeddings while preserving voice identity. Without this, voices with higher embedding norms produce hotter output, and LUFS changes cascade unpredictably through shared weights.
**LUFS:** All voices at **uniform -22 LUFS** (refs and clips). Do NOT use per-voice LUFS adjustments — they cause unpredictable cross-voice interference through shared transformer weights ("whack-a-mole"). Tested extensively 2026-05-11 across 12 training runs.
**Inference temperature:** 0.7 (not 0.6). Training data was generated at 0.85 — inference at 0.6 flattens prosody. 0.7 is the sweet spot.

### Single-Voice

```bash
python3 sft_12hz.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /path/to/output \
  --train_jsonl /path/to/train_curated.jsonl \
  --batch_size 2 --lr 1e-7 --num_epochs 2 \
  --speaker_name <voice>
```

With `--save_every_steps 45` you get fractional epoch checkpoints (~0.2 epoch intervals). Each is a full model copy (~1.7GB bf16), budget ~17GB disk.

### Multi-Voice

```bash
python3 sft_12hz_multivoice.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /path/to/output \
  --train_jsonl /path/to/merged_multivoice.jsonl \
  --batch_size 2 --lr 1e-7 --num_epochs 2 \
  --voice_slot_map_json '{"nora":3002,"joe":3001}'
```

**JSONL format:** Each entry must have a `voice_name` field. Audio paths are per-voice prefixed (`./nora/audio/clip_0001.wav`, `./joe/audio/clip_0001.wav`). Ref audio similarly (`./nora/ref.wav`, `./joe/ref.wav`).

**Building the merged JSONL:** Read each voice's `train_curated.jsonl`, add `voice_name` field, rewrite `audio` and `ref_audio` paths with per-voice prefixes, concatenate, shuffle.

**Slot assignments:** Voices get slots in the 3000-3071 custom voice region of `codec_embedding.weight` (3072 × 1024). Slots 0-2047 are active codec tokens — never overwrite. Adjacent slots are fine (CustomVoice has adjacent built-in voices).

### Critical Rules

- **lr=1e-7 is non-negotiable.** Higher LRs (2e-6, 2e-5) destroy EOS token — model generates until max_new_tokens.
- **Only code patch needed:** Wrap `text_embedding` with `text_projection` in the forward pass (fixes 0.6B dimension mismatch where text_embed=2048 but codec_embed=1024). Already in `sft_12hz.py` and `sft_12hz_multivoice.py`.
- **Do NOT apply** the "double label shift" fix or "remove sub-codebook loop" fix at this LR — they break training.
- **EOS termination is THE diagnostic.** If PyTorch inference hits max_new_tokens on a short sentence, the model is broken.
- **Always verify with PyTorch inference on GPU first.** It's ground truth. MLX issues are separable from training issues.

### Multi-Voice Specific

- **Embedding injection bug (fixed 2026-04-27):** The training loop must inject cached per-voice embeddings at position 6, NOT live speaker_encoder output. In mixed-voice batches, ref_mel zero-padding corrupts the shorter voice's live embedding extraction. The fix is in `sft_12hz_multivoice.py`:
  ```python
  for b_idx, vname in enumerate(voice_names):
      input_codec_embedding[b_idx, 6, :] = target_speaker_embeddings[vname][0]
  ```
- **Embedding norms:** Should be naturally close (~1-2% difference) for voices with well-matched reference audio. If >5% different, investigate reference audio quality. Norms are logged at end of epoch 0 (`[diag] <voice> embedding norm: X.XXXX`).
- **Mixed batches are fine.** Standard practice in multi-speaker TTS (VITS, YourTTS, CosyVoice 2). Don't segregate by voice.
- **Sub-talker loss weight 0.3 is the upstream default.** Keep it.
- **No one else has publicly documented successful multi-voice Qwen3-TTS fine-tuning.** We're first (as of 2026-04-27).

## Voice Data Pipeline

End-to-end: voice design → reference audio → 500 cloned clips → enhance → curate → train-ready JSONL.

### 1. Voice Design (~30 min, interactive)

Pick character from `voices/VOICES.md`. Generate 8 candidates via VoiceDesign model:

```python
from mlx_audio.tts import load
model = load("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit")
for result in model.generate_voice_design(
    text=TARGET_TEXT, instruct=instruct, language="english", temperature=0.9, stream=False
):
    chunks.append(result.audio)
```

**Proven target texts:**
- Assistant-style: `"Okay, so I looked into it and here's what I found. The file you were working on got saved to your Downloads folder, not your Desktop. Want me to move it over, or would you rather keep it where it is?"`
- Character-style: `"Look, I already checked the numbers twice. They don't add up, and I'm not gonna sugarcoat it for you. We need to fix this before the meeting, or we're in trouble."`

Use assistant-style for voices meant to be assistants. Character-style makes voices sound like movie characters.

**Prompting tips (from 800+ candidate analysis):**
- "Velvety", "smooth", "warm", "natural" produce cleaner male voices (40% pass vs 13% for "gravelly"/"rough")
- High-pitched female voices all sound identical — vary texture/pacing, not just energy
- VoiceDesign **cannot produce regional accents** — it controls timbre only. For accented voices, find real accented ref audio and clone.
- Temp 0.8 has slightly better quality pass rate than 0.9 or 0.75

**Enhance VoiceDesign output** with `tools/enhance_voicedesign.py` (trim → LUFS → IIR notch), NOT `enhance_clips.py`. The full pipeline's STFT de-esser creates chirping/musical noise on already-clean synthetic audio.

```bash
.venv-enhance-audio/bin/python tools/enhance_voicedesign.py --input <raw_dir> --output <enhanced_dir>
```

Refine → narrow → pick winner → optionally clone through 1.7B-Base-bf16 for cleaner version.

### 1b. Prepare Reference Audio

Clean the reference with DeepFilterNet3 only (single pass) + LUFS normalize to match training clip LUFS. **Do NOT cascade enhancers** (ClearVoice + DeepFilter + noisereduce was proven harmful — adds noise to silence, doubles sibilance). Use `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (full precision) for cloning.

**⚠️ CRITICAL: All voices at uniform LUFS.** The ECAPA-TDNN speaker encoder bakes loudness into the embedding. Per-voice LUFS adjustments cause unpredictable cross-voice interference through shared transformer weights ("whack-a-mole"). Tested extensively 2026-05-11: changing one voice's LUFS by 3 dB shifted other voices' output by up to 5+ dB.

**⚠️ Use embedding normalization.** L2-normalize speaker embeddings to magnitude 10.0 before injection during training. This strips the loudness signal from embeddings while preserving voice identity. Without it, some voices (e.g., Nora) consistently produce output 5-9 dB hotter than training data regardless of LUFS adjustments.

**⚠️ Ref audio length matters.** Longer refs (7-11s) produce more stable voice identity at inference. Short refs (4-5s) give the ECAPA-TDNN less context → vaguer embedding → variable voice across generations. Use the longest good clip available.

**Current LUFS target: -22** (was -20, changed 2026-05-11). All voices at the same level. Both refs and training clips must match.

```bash
# Quick check — all refs should be close to -20 RMS
.venv-enhance-audio/bin/python -c "
import soundfile as sf; import numpy as np
data, sr = sf.read('voices/<name>/ref.wav')
print(f'RMS: {20*np.log10(np.sqrt(np.mean(data**2))+1e-10):.1f} dBFS')
print(f'Peak: {20*np.log10(np.max(np.abs(data))+1e-10):.1f} dBFS')
"
```

### 2. Generate Training Data

500 clips via 1.7B-Base voice cloning. Temperature **0.85** (0.6 = monotone — F0 range 168 Hz vs 235 Hz at 0.85). Do NOT append trailing silence — it masks abrupt cutoffs instead of exposing them.

**Generate locally on Mac:**

```bash
cd holler && .venv/bin/python tools/generate_training_data.py \
  --voice <name> --ref-text "<exact transcript of ref.wav>"
```

Uses `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (NOT 8-bit) via mlx-audio. Outputs to `voices/<name>/training-data/audio-original/` + `train.jsonl`. ~25 min per voice on M1 Pro.

**Legacy:** `training/remote_generate_training_data.py` exists for GPU generation on Vast.ai but is no longer used — local generation is the same speed with less hassle.

### 3. Enhance Clips (~3 min)

```bash
.venv-enhance-audio/bin/python tools/enhance_clips.py --voice <name> --gender <male|female>
```

Pipeline: Trim silence → DeepFilterNet3 → LUFS normalize (-20 LUFS, was -18) → Spectral de-ess → Dynamic presence.

Gender affects de-esser band (male: 4500-7000Hz, female: 6000-9000Hz).

**Do NOT cascade enhancers.** Single-pass DeepFilterNet3 only. ClearVoice + DeepFilter + noisereduce cascading adds artifacts to clean TTS audio. Paper: "Amplifying Artifacts with Speech Enhancement" (arXiv:2506.11542). Proven by measurement: same 10 clips through old cascade vs new single-pass — sibilance dropped 73% (3.7% → 1.0%), spectral tilt normalized 2 dB warmer, LUFS consistency improved 3.5x.

### 4. Auto-Curate

```bash
.venv-enhance-audio/bin/python tools/auto_curate.py --voice <name>          # report only
.venv-enhance-audio/bin/python tools/auto_curate.py --voice <name> --apply  # write files
```

Rejects clips failing: DNSMOS OVRL <3.0, peak >-1 dBFS, duration <0.5s, silence >50%, HNR <10, sibilance >5%, harshness >2%.

Writes `rejects_auto.txt` + `train_curated.jsonl`. Never deletes audio files.

**Expected rejection rates:** ~25-30% for female voices, ~40-90% for male voices (male cloned audio runs hotter/harsher — generate more, don't loosen thresholds).

### 5. Optional Manual Curate

```bash
.venv-enhance-audio/bin/python tools/curate_clips.py --voice <name>
```

Tinder-style swipe for prosody issues metrics can't catch.

### 6. Verify Before Training

Run analysis on the curated clips and share results with the user before moving to GPU training:

```bash
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py \
  --source voices/<name>/training-data/audio --label "<name>-curated" --n 80
```

**Check these against targets below:**
- Harshness < 2% (if >2%, data is too hot — check enhancement pipeline or regenerate)
- Sibilance < 3%
- LUFS between -20 and -16
- Peak between -4 and -1 dBFS
- HNR > 14 dB
- DNSMOS OVRL > 3.0

**Also verify:**
- `train_curated.jsonl` exists and has enough clips (target 300+, minimum ~50)
- `ref.wav` exists in `training-data/`
- Auto-curate rejection rate is reasonable (<35% for female, <50% for male — if >50% male, data quality is suspect, consider regenerating)

Share results with the user before training — a quick sanity check avoids wasting GPU time on bad data (learned the hard way with Joe's 89% rejection rate).

### Audio Quality Targets

Reference: Serena (built-in CustomVoice). Based on ear testing at 90% AirPods Pro 3 volume.

| Metric | Target | Why |
|--------|--------|-----|
| Peak dBFS | -4 to -1 | Headroom at -22 LUFS target |
| LUFS | -20 to -16 | Voice assistant loudness (podcast/Siri level) |
| Harshness 2-4kHz | < 2% | Predicts ear pain better than volume |
| Sibilance 4-10kHz | < 3% | Excessive = fatiguing |
| Presence 1-5kHz | 8-20% | Too low = muffled, too high = sharp |
| HNR | > 14 dB | Voice clarity |
| F0 range | > 200 Hz | Expressiveness |

**Key insight:** Harshness and presence predict ear pain better than volume. Aiden peaks at -3.6 dBFS (hot) but doesn't hurt (1.6% harshness). Vivian peaks at -3.1 (similar) but hurts (5.7% harshness).

Reference comparison:

| Metric | Serena (gold standard) | Vivian (hurts) |
|--------|------------------------|----------------|
| Peak dBFS | -10.2 | -3.1 |
| LUFS | -25.7 | -20.0 |
| Harshness | 1.4% | 5.7% |
| Presence | 15.4% | 35.4% |

### Crest Factor and Per-Voice Loudness

LUFS target and peak ceiling interact differently per voice. Voices with high crest factor (sharp transients — hard plosives, deep fundamentals with explosive consonants) will land below the LUFS target because the peak ceiling clamps them down.

**Measured crest factors (2026-05-02):**

| Voice | Crest Factor | LUFS after -18 target | % clips clamped by -1.0 ceiling |
|-------|-------------|----------------------|--------------------------------|
| Kit | 16.3 dB | -18.2 (on target) | 16% |
| Nora | 16.0 dB | -18.2 (on target) | 26% |
| Joe | 16.9 dB | -18.1 (on target) | 4% |
| Dakota | 19.3 dB | -19.6 (1.5 dB short) | 90% |

**Rule of thumb:** Check crest factor from originals before enhancing. If >17 dB, the voice will land 1-2 dB below LUFS target — this is fine, don't fight it. Raising the ceiling (tested -0.3 dBFS on Dakota) barely helps and is inaudible. The transients are what make the voice sound like itself.

### Enhancement Tunable Knobs

**De-esser gender presets:**

| Knob | Male | Female |
|------|------|--------|
| `deess_low` | 4500 Hz | 6000 Hz |
| `deess_high` | 7000 Hz | 9000 Hz |
| `deess_threshold` | -6.0 dB | -4.0 dB |
| `deess_max_reduction` | 8.0 dB | 6.0 dB |

**Presence:** 3500-8000 Hz range, max boost 2.5 dB, sensitivity 1.0.

**LUFS:** target -22.0, peak ceiling -1.0 dBFS.

### Analysis Tools

- `tools/analyze_voice_quality.py` — 20+ metric voice quality analysis (requires `.venv-enhance-audio/`)
- `tools/benchmark_quality.py` — **Quality benchmark for trained models** (requires `.venv` with mlx-whisper). Uses Whisper medium word timestamps for gap analysis + strict WER. Run after every training run.
- `tools/noise_profile.py` — before/after noise comparison by frequency band
- `tools/deess.py` — standalone de-esser (also integrated in enhance_clips.py)

```bash
# Analyze a directory of clips
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source voices/joe/training-data/audio --label "joe-v1" --n 80
# Export to JSON
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source <dir> --label "name" --n 80 --json output.json

# Benchmark a checkpoint (raw mode = no guardrails, true model quality)
.venv/bin/python tools/benchmark_quality.py -c checkpoints/<name> --holler-bin ./holler --raw --temperature 0.7 --label <name>
# Compare two checkpoints
.venv/bin/python tools/benchmark_quality.py --compare ~/Downloads/holler-bench/v1 ~/Downloads/holler-bench/v2 --compare-labels v1 v2
```

**Metrics:** peak_db, rms_db, crest_db, lufs, dc_offset | centroid_hz, harsh_2_4k, sib_4_10k, presence_1_5k, low_80_300, air_10k, tilt_db_oct, flatness, rolloff_hz | silence_ratio, dyn_range_db | f0_mean/std/range, voiced_ratio, jitter_pct, shimmer_pct, hnr_db, f1/f2/f3_hz | dnsmos_sig/bak/ovrl. Uses parselmouth (Praat) for voice quality, torchmetrics for DNSMOS P.835.

## GPU Training Runbook

### Requirements

- **24GB VRAM required.** Training uses ~10GB, but tokenization (`prepare_data.py`) peaks at 23GB. 16GB cards will fail at tokenization.
- CUDA 12.x, bf16 support. **NVIDIA driver ≥550 required** for the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` Docker image. Driver 535 fails with "forward compatibility not supported on GeForce" (error 804).
- RTX 3090 (~$0.12-0.16/hr), RTX 4090 (~$0.30/hr), A100 (~$0.50/hr) on Vast.ai.
- 80GB disk minimum
- Docker image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- SSH key: `~/.ssh/runpod`

### Step 1: Rent + Setup

**⚠️ Blacklisted machines (unreliable boot):**
- mach_id 43503 (host 155125, California) — gets stuck on "verifying checksum" for 5+ min, never boots

```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 rentable=true disk_space>=80' -o 'dph_total' --limit 10
vastai create instance <ID> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 80 --ssh

vastai show instances  # wait for "running"

scp -i ~/.ssh/runpod -P <PORT> holler/training/remote_setup.sh root@<HOST>:/workspace/
ssh -i ~/.ssh/runpod -p <PORT> root@<HOST> "bash /workspace/remote_setup.sh"
```

Setup pulls cached pip wheels + models from R2 first (`r2:holler/instance-cache/`), so pip installs are local and model downloads are skipped. Then installs remaining deps and pulls training data. Takes ~5-8 min (vs ~15-25 min without cache, depending on region).

**R2 dependency cache** (`r2:holler/instance-cache/`):
- `pip-cache.tar.gz` (~3GB) — all pip wheels (torch, torchaudio, qwen-tts, etc.). Extracted to `/root/.cache/pip/` so `pip install` finds them locally.
- `models-cache.tar.gz` (~2.5GB) — 0.6B-Base + Tokenizer-12Hz. Extracted to `/workspace/models/`, HF download steps detect existing files and skip.

**Updating the cache:** If you bump torch, flash-attn, or Python versions, rebuild the cache from a fresh instance:
```bash
# On the instance after setup completes:
tar czf /workspace/pip-cache.tar.gz -C /root/.cache pip/
tar czf /workspace/models-cache.tar.gz -C /workspace models/
rclone copyto /workspace/pip-cache.tar.gz r2:holler/instance-cache/pip-cache.tar.gz
rclone copyto /workspace/models-cache.tar.gz r2:holler/instance-cache/models-cache.tar.gz
```

**Note:** flash-attn builds from source (~3-5 min) even with the pip cache, since the original install used `--no-build-isolation`. The source tarball is cached but compilation still runs. Could cache a pre-built wheel to skip this — see "What's Unknown" section.

### Step 2: Training Data

Training data lives on R2 at `holler-data.sentium.one`. The setup script (`remote_setup.sh`) pulls it automatically via rclone.

**To update R2 data locally:**
```bash
# Build combined JSONL
python training/build_combined_jsonl.py --voices kit dakota nora joe oliver tessa

# Upload to R2 (rclone remote 'r2-sentium' must be configured)
for voice in kit dakota nora joe oliver tessa; do
  rclone copy "voices/$voice/training-data/audio/" "r2-sentium:holler/training-data/$voice/audio/" --transfers 16
  rclone copyto "voices/$voice/training-data/ref.wav" "r2-sentium:holler/training-data/$voice/ref.wav"
done
rclone copyto training/train_multivoice.jsonl r2-sentium:holler/training-data/train_multivoice.jsonl
```

**⚠️ Upload refs from `training-data/ref.wav`, NOT from `voices/<name>/ref.wav` (root).** The root refs may not be normalized. The training-data refs are the prepared ones. This was a bug in the first 6-voice train (2026-05-04).

**Still need to SCP the training script:**
```bash
scp -i ~/.ssh/runpod -P <PORT> training/sft_12hz_multivoice.py root@<HOST>:/workspace/
```

### Step 3: Train

**Single-voice:** `bash /workspace/remote_train.sh <voice>`

**Multi-voice:** `bash /workspace/remote_train_multivoice.sh kit:3000 dakota:3001 nora:3002 joe:3003`

Both scripts handle tokenization (via `prepare_data.py`) and training. ~5 min for 400 clips on A100, ~10-15 min on 3090.

### Step 4: Verify on GPU

Run PyTorch inference on the checkpoint before downloading. Use `inference/test_multivoice_gpu.py` or write a quick script. Check:
- Voices sound distinct
- EOS terminates (no runaways)
- Audio lengths proportional to text
- Peaks < 1.0 (no hard clipping)

### Step 5: Download + Quantize Locally

Upload checkpoint to R2 from the instance, then pull from R2 to Mac (faster than direct rsync, especially from non-US instances):
```bash
# On instance: upload to R2
rclone copy /workspace/output/checkpoint-epoch-1/ r2:holler/checkpoints/<name>/ --transfers 4 -v

# On Mac: pull from R2
rclone copy r2-sentium:holler/checkpoints/<name>/ holler/checkpoints/<name>/ --transfers 8 --progress

# Direct rsync fallback (slow from some regions):
# rsync -avz root@<HOST>:/workspace/output/checkpoint-epoch-1/ holler/checkpoints/<name>/

# Quantize (6-bit affine g64)
python3 -c "
from mlx_audio.convert import convert
convert(hf_path='checkpoints/<name>', mlx_path='checkpoints/<name>-6bit', quantize=True, q_bits=6, q_group_size=64, q_mode='affine')
"

# CRITICAL: copy ENTIRE speech_tokenizer directory (mlx_audio.convert bug)
# The speech tokenizer is identical across ALL checkpoints — reuse a canonical copy.
mkdir -p checkpoints/<name>-6bit/speech_tokenizer
cp checkpoints/<canonical>/speech_tokenizer/* checkpoints/<name>-6bit/speech_tokenizer/
# Files needed: config.json, configuration.json, preprocessor_config.json, model.safetensors
```

### Step 6: Serve

```bash
.venv/bin/python3 inference/server.py --checkpoint checkpoints/<name>-6bit
# → http://localhost:8100 with voice selector dropdown
```

Server auto-detects available voices from checkpoint config. DEFAULT_VOICE is first alphabetically.

## Quantization

6-bit affine g64 is the pick. Done via mlx-audio's converter (NOT `mlx_lm.convert`).

```python
from mlx_audio.convert import convert
convert(hf_path='<bf16_path>', mlx_path='<output_path>', quantize=True, q_bits=6, q_group_size=64, q_mode='affine')
```

**CRITICAL BUG:** `mlx_audio.convert()` does NOT copy the `speech_tokenizer/` directory (682MB codec decoder + config files). Without it, model either decodes to silence or crashes with "Speech tokenizer not loaded". Always copy the **entire directory** (model.safetensors + config.json + configuration.json + preprocessor_config.json) from a canonical source after quantization. The speech tokenizer is identical across all checkpoints — reuse the same copy, never re-download. Also copy it into bf16 checkpoints downloaded from R2 (which exclude speech_tokenizer to save bandwidth).

| Variant | Disk | Bits/wt | Notes |
|---------|------|---------|-------|
| **affine 6-bit g64** | **1094M** | **10.1** | **THE PICK** |
| affine 4-bit g64 | 960M | 8.9 | Good, less presence |
| affine 3-bit g64 | 893M | 8.3 | BROKEN — EOS destroyed |
| mxfp8 | 1210M | 11.2 | Worse than 6-bit despite more bits |

Codec_embedding and speaker embedding layers stay bf16 automatically (converter keeps small/critical layers full precision).

## Technical Details

- `codec_embedding.weight`: 3072 slots × 1024-dim. Slots 0-2047 = active codec tokens. 3000-3071 = custom voice region (72 slots).
- Fine-tuning writes speaker embedding to a slot and co-trains the full model. At inference: voice name → config `talker_config.spk_id` → slot → embedding injection at codec position 6.
- Training data: ~50-500 clips per voice (minimum viable untested below 50), 24kHz mono WAV, cloned from reference through 1.7B-Base-bf16 at temp 0.85.
- Speaker encoder is ECAPA-TDNN. Embeddings are detached (not trained through). Norms ~10.0-10.5 for well-prepared reference audio.

## Environment Gotchas

- **flash-attn 2.7.3** works with torch 2.6. Version 2.8.3 does NOT (ABI symbol mismatch).
- **Never install into system python.** Always venv.
- **Install `wheel` + `setuptools` before flash-attn** — builds from source.
- **`sox` via apt** — `prepare_data.py` needs the binary.
- **sdpa is a valid fallback** if flash-attn won't build. Same training results.
- **JSONL relative paths** resolve from cwd. Symlink `audio/` and `ref.wav` into the working directory.
- **`generate_custom_voice()` return type varies** across qwen-tts versions — can be tensor, tuple, or list. Always handle all three.
- **Python stdout buffering over SSH.** Remote scripts produce no output until completion because Python buffers stdout when not attached to a TTY. Always set `PYTHONUNBUFFERED=1` or use `python -u` in remote training/generation scripts.
- **SSH key for Vast.ai:** `~/.ssh/runpod`

## Sample Size Findings (2026-04-27)

Tested 50, 100, 200 clips per voice (multi-voice Nora+Joe, 1 epoch each on 3090):
- All three train successfully with loss in correct 14-17 range
- Zero EOS failures across all runs
- Embedding norms identical across all runs (10.125/10.250)
- Quality comparison pending (samples in ~/Downloads/holler-sweep-samples/)

Auto-curate rejection rates:
- Nora (female): 26.8% rejected → 366 kept from 500
- Joe (male): 89.4% rejected → 53 kept from 500 (data quality issue, needs regeneration)

## What's Unknown

- Minimum viable clip count for acceptable quality (50 trains, but how does it sound?)
- Whether 30 voices in one checkpoint holds up (tested at 2)
- Sequential vs joint training for many voices
- Optimal epoch count vs clip count tradeoff
- Whether trimming leading silence from training data reduces 220ms codec warmup
- EOS failure rate (~2-4%, model-level) — mitigated by safety cap, not eliminated
