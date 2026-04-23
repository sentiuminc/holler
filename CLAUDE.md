# Holler

Open-source American English voice pack for Qwen3-TTS 0.6B. By Sentium.

**What this is:** A fine-tuned Qwen3-TTS-12Hz-0.6B model with 30 high-quality American English voices, optimized for local inference on Apple Silicon via mlx-audio. ~80ms TTFA streaming, ~2GB RAM at 4-bit. The fastest high-quality local TTS available on Mac.

**What this is not:** A new TTS architecture. This is a fine-tune of Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) with better English voices and a fully open training pipeline. All credit for the base model goes to the Qwen team at Alibaba. Our contribution is the voices, the Mac-focused inference setup, and the open training pipeline.

**License:** The base Qwen3-TTS model is Apache 2.0. Our fine-tune scripts and voices will also be Apache 2.0. All releases must include proper attribution to Qwen/Alibaba.

**Focus:** Mac. Local inference on Apple Silicon specifically. The training can happen on any CUDA GPU, but the inference target is mlx-audio on M-series Macs.

**Origin:** Started as the voice component of [ivi](https://ivi.computer), a macOS notch AI assistant by Sentium. We open-sourced it because Qwen3-TTS is SOTA for local inference but ships with only 2 mediocre English voices — and nothing else fills that gap.

## Current State (2026-04-23)

- **Recipe:** Proven. lr=1e-7, 2 epochs, text_projection patch only on upstream `sft_12hz.py`.
- **Single-voice (Katie, v6):** Clean. Production-ready. Checkpoint at `checkpoints/katie-v6/` (bf16 source) and `checkpoints/katie-v6-4bit/` (production pick).
- **Quantization:** 4-bit affine is the pick. 8-bit deleted (redundant).
  - `checkpoints/katie-v6/` — bf16, 2.3GB disk (source for quantization)
  - `checkpoints/katie-v6-4bit/` — **4-bit affine (the pick)**, 960MB disk, 1.6GB Metal RAM, 64-80ms streaming TTFA
- **Multi-voice (Katie+Joe, v7):** bf16 only at `checkpoints/katie-joe-v7/`. Quality issues: short utterances like "Okay." generate silence. Can re-quantize to 4-bit anytime. Not production-grade.
- **Voices designed:** 2/30 (Katie, Joe). 28 more needed.
- **Inference runtime:** Python mlx-audio. Tested, clean audio, no artifacts. Streaming TTFA 64-80ms. See Inference Architecture section below.
- **Swift evaluation:** soniqo/speech-swift tested 2026-04-23. Rejected — audio pops, end cutoff, streaming crash after ~13 calls. Weight formats are identical to mlx-audio (verified key-by-key). Reference code is at `speech-swift/` for future use.

## Structure

```
holler/
├── CLAUDE.md              — This file (project instructions + state)
├── .gitignore
├── training/              — Training scripts (single-voice + multi-voice SFT)
├── inference/             — MLX + PyTorch inference, TTFA benchmarks, live demo
│   ├── benchmark_katie_v6.py       — bf16 benchmark (10 texts, TTFA + peaks)
│   ├── benchmark_quant_compare.py  — single-quant benchmark with psutil (deprecated measurement)
│   ├── benchmark_quant_proper.py   — all-quant benchmark with MLX Metal memory (use this one)
│   ├── benchmark_sentence_queue.py — sentence-queue benchmark simulating ivi pattern (use this one)
│   ├── live_demo.py                — Web UI: type text, hear it spoken (~80ms TTFA)
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
│   └── joe/               — Male voice (VoiceDesign-sourced, slot 3001)
│       ├── ref.wav
│       ├── candidates/    — 28 voice design candidates + index.txt
│       └── training-data/ — 385 clips + train.jsonl
├── checkpoints/           — Model checkpoints (not in git — large)
│   ├── katie-v6/          — 1.7GB bf16 (source for quantization)
│   ├── katie-v6-8bit/     — 1.2GB 8-bit affine (q_group_size=64)
│   ├── katie-v6-4bit/     — 960MB 4-bit affine (q_group_size=64) ← THE PICK
│   └── katie-joe-v7/      — 2.3GB bf16, multi-voice, quality unresolved
├── samples/               — Audio samples organized by version/voice/precision
│   ├── benchmark-katie-v6-bf16/    — 10 benchmark clips
│   ├── benchmark-katie-v6-8bit/    — 10 benchmark clips
│   ├── benchmark-katie-v6-4bit/    — 10 benchmark clips
│   ├── v6-mlx/, v6-pytorch/, v5/   — earlier samples
│   └── v7-{mlx,pytorch}-{katie,joe}/ — multi-voice samples
├── logs/
│   ├── sessions/          — Session logs (the full journey)
│   └── runs/              — Raw training/inference logs
└── docs/
    ├── ivi-session-notes.md        — Full debugging journey + v7 addendum
    └── handover-lora-failure.md    — Historical: why LoRA doesn't work
```

## The Winning Recipe

```bash
python3 sft_12hz.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /path/to/output \
  --train_jsonl /path/to/train_with_codes.jsonl \
  --batch_size 2 --lr 1e-7 --num_epochs 2 \
  --speaker_name voice_name
```

- **lr=1e-7 is critical.** Higher LRs (2e-6, 2e-5) destroy EOS token — model generates until max_new_tokens.
- **Only patch needed:** Wrap `text_embedding` with `text_projection` on line ~89 of upstream `sft_12hz.py` (fixes 0.6B dimension mismatch). Script: `training/sft_12hz_patched.py`.
- **Do NOT apply** the "double label shift" fix or "remove sub-codebook loop" fix at this LR — they break training.
- **Epoch 1** is the pick. Epoch 0 is underfit, epoch 2+ can drift.
- **Loss stays at ~12-13.** That's correct. Low loss at higher LR = overfitting, not quality.

## Quantization

Done via **mlx-audio's own converter** (NOT `mlx_lm.convert` — that doesn't support qwen3_tts model type):

```python
from mlx_audio.convert import convert
convert(
    hf_path='checkpoints/katie-v6',
    mlx_path='checkpoints/katie-v6-4bit',
    quantize=True, q_bits=4, q_group_size=64, q_mode='affine',
)
```

- Affine quantization: per-group (64 weights) scale + zero-point, weights stored as N-bit integers
- The converter **selectively keeps critical layers at full precision** (codec_embedding, speaker embeddings, small layers). That's why "4-bit" averages 8.9 bits/weight and "8-bit" averages 11.4 bits/weight.
- MLX dequantizes on-the-fly in Metal GPU kernels during matmul — no separate unpack step.
- Other quantization modes available but untested: `mxfp4`, `mxfp8`, `nvfp4`. Mixed precision recipes: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`. Could improve quality at same size — worth exploring later.

### Benchmark Results (2026-04-22, M-series Mac)

| | bf16 | 8-bit | 4-bit |
|--|------|-------|-------|
| Disk | 1.7GB | 1.2GB | **960MB** |
| Metal RAM (load) | 2,378MB | 1,878MB | **1,611MB** |
| Metal RAM (peak) | 2,703MB | 2,203MB | **1,936MB** |
| Activity Monitor | ~3GB | ~2.4GB | **~2GB** |
| TTFA median | 101ms | 79ms | **79ms** |
| TTFA range | 80-114ms | 66-81ms | 67-89ms |
| Effective bits/weight | 16 | 11.4 | 8.9 |

**4-bit is the pick.** Sounds more alive than 8-bit despite marginally higher peaks. Smallest disk + RAM footprint. Same TTFA.

**Important:** Use `mx.metal.get_active_memory()` / `mx.metal.get_peak_memory()` for measuring MLX memory — NOT `psutil.Process().memory_info().rss` which gives garbage numbers for MLX workloads.

## Inference Architecture (decided 2026-04-23)

**Holler is a model, not an inference library.** Ship weights on HuggingFace, users run via mlx-audio. One version, not two (no separate Swift implementation).

**For ivi:** mlx-audio Python sidecar process. WebSocket-based, queue-based chunked streaming.
- LLM streams text → detect sentence boundaries → stream TTS for each sentence
- Audio chunks go into a playback queue, skip silence chunks (peak < 0.05)
- First audible audio at 68-126ms after text ready (streaming)
- Generation always faster than real-time (RTF 0.4-0.7), queue never starves

**Sentence timing benchmarks (Katie v6 4-bit, 2026-04-23):**
| Sentence | Words | Non-stream | Stream TTFA | First audible |
|----------|-------|------------|-------------|---------------|
| "Hey!" | 1 | 312ms | 67ms | 319ms (silence) |
| "Exactly." | 1 | 762ms | 72ms | 126ms |
| "Yes, I agree." | 3 | 726ms | 68ms | 68ms |
| "Got it." | 2 | 445ms | — | — |
| 7-word sentence | 7 | 817ms | 65ms | 320ms |
| 14-word sentence | 14 | 1,986ms | 65ms | ~200ms |

**Streaming wins by ~400-650ms** for most sentences. Exception: very short exclamations ("Hey!") where codec warmup silence means non-streaming ties.

## Inference Pipeline

**Stack:** mlx-audio (Python) → MLX → Metal GPU (Apple Silicon unified memory)

1. `mlx_audio.tts.load(checkpoint_path)` loads the model
2. `model.generate(text=..., voice="katie", language="english", temperature=0.6, stream=True, streaming_interval=0.1)` streams codec tokens at 12Hz
3. Each chunk yields ~80ms of 24kHz audio (1920 samples)
4. Play via `sounddevice.OutputStream` or save via `soundfile`

**Live demo:** `inference/live_demo.py` — HTTP server on port 8099, type text in browser, audio plays from Mac speakers. Uses pre-opened audio stream for minimal latency.

**Python venv:** Currently using ivi's venv at `~/Desktop/Files/AI/ivi/audio-test/.venv-tts-bench/` (mlx-audio 0.4.2, mlx-lm 0.31.1, sounddevice 0.5.3). Should create holler's own venv.

## Known: 220ms Leading Silence

The first 2-3 streaming chunks (~220ms) from any Qwen3-TTS generation are near-silence. This is a **known codec LM architecture behavior** — confirmed across Qwen3-TTS, CosyVoice, VALL-E descendants. The codec decoder needs initial context tokens before producing meaningful audio.

Our training data also has 25-212ms of leading silence per clip, which reinforces this behavior. Future training runs should trim leading silence from training clips.

**Mitigations (not yet implemented):**
- Trim leading silence from training data before next training run
- For ivi integration: sentence-level streaming from LLM overlaps codec warmup with text generation
- Study `rekuenkdr/Qwen3-TTS-streaming` — two-phase streaming fork that buffers past the silence before first emit (208ms first audible vs 570ms baseline)

## Key Technical Details

- Qwen3-TTS `codec_embedding.weight` has 3072 slots (1024-dim each). Slots 0-2047 are active codec tokens (DON'T overwrite). 3000-3071 is the 72-slot custom voice region.
- Fine-tuning writes a speaker embedding to a slot and co-trains the full model. At inference, voice name → config lookup → slot → embedding injection at codec position 6.
- Training data: ~385 clips per voice, voice-cloned from a reference through Qwen3-TTS-1.7B-Base. 24kHz mono WAV with 1s trailing silence.
- Multi-voice: same recipe, but JSONL has per-sample `voice_name` field, and training script tracks embeddings per voice. See `training/sft_12hz_multivoice.py`.
- Voice name in v6 config is `katie` at slot 3000 (nested under `talker_config.spk_id`).

## GPU Training

No Vast.ai instances exist. To train, rent a fresh A100 SXM4 40GB (~$0.50/hr on Vast.ai) and follow the runbook in `docs/ivi-session-notes.md` steps 1-11. All training data is local under `voices/`.

**Requirements:** 24GB+ VRAM, CUDA, bf16 support. Training takes ~3-5 min per 2 epochs on A100 with 385 clips per voice.

**Runbook needs standardizing** — the current docs have single-voice steps 1-11, multi-voice is more narrative. Should be consolidated into a proper from-scratch runbook before scaling to 30 voices.

## Community References

- **rekuenkdr** — anonymous hobbyist, posted the winning lr=1e-7 recipe (Issue #39), built the two-phase streaming fork (72 stars). Also works on "OVA" (local voice assistant pipeline).
- **Our GitHub comment** documenting findings: https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999

## HuggingFace Release Target

- `sentium/holler-0.6b` (bf16)
- `sentium/holler-0.6b-8bit`
- `sentium/holler-0.6b-4bit` (the pick for most users)
- Additional precisions as needed

## Hard-Won Lessons

- **LR is the dominant knob.** Not loss, not epochs, not community patches.
- **EOS termination is the diagnostic.** If PyTorch inference hits max_new_tokens on a short sentence, the model is broken.
- **Always verify with PyTorch inference on GPU first.** It's the ground truth. MLX issues are separable from training issues.
- **Loss decreasing ≠ quality.** Loss 1.0 produced noise; loss 12.8 produced clean voice.
- **ref_mel shape is [1, T, 128]** after upstream .transpose(1,2), NOT [1, 128, T]. Pad dim=1 for multi-voice.
- **Use mlx-audio's converter, not mlx_lm** — mlx_lm doesn't support the qwen3_tts model type.
- **psutil RSS is garbage for MLX memory** — use `mx.metal.get_active_memory()`.
- **4-bit quantization works surprisingly well** for voice quality. Sounds more alive than 8-bit.
- **Leading silence is architectural** — all codec LMs do it. Trim at inference or overlap with LLM streaming.

## What's NOT Known / Unresolved

- Whether joint training at 30 voices holds up (only tested at 2)
- Root cause of v7 quality issues (Katie noise, Joe clipping) — see docs/ivi-session-notes.md "Observations" section
- Whether sequential training (one voice at a time, cumulative checkpoints) works better than joint
- Optimal voice count per training run
- Whether alternative quantization modes (mxfp4, mixed_4_6) give better quality
- Whether trimming leading silence from training data reduces the 220ms codec warmup
- There's reportedly a HuggingFace repo that improved Qwen3-TTS size/inference by 3x — not yet identified

## What's Next

1. **Wire Katie v6 4-bit into ivi** — mlx-audio Python sidecar with WebSocket, queue-based chunked streaming. This is the immediate next step.
2. **Solve multi-voice quality** — v7 Katie+Joe has issues (short utterances generate silence). Root cause still unknown.
3. **Design and train more voices** — 28 more needed. Names and characters in `voices/VOICES.md`.
4. **Standardize GPU runbook** — from-scratch instance setup → training → quantization
5. **Create holler's own Python venv** — currently borrowing ivi's
6. **HuggingFace release** under `sentium/` with full docs
7. **Explore optimizations** — two-phase streaming, alternative quant modes, leading silence trimming
