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
│   │   └── training-data/ — 475 clips + train.jsonl + train_curated.jsonl (452)
│   │       ├── audio/           — enhanced clips (new pipeline: DeepFilter→LUFS→deess→presence)
│   │       ├── curation.json    — Tinder decisions (452 keep, 23 reject)
│   │       └── train_curated.jsonl — TRAINING FILE (452 entries)
│   └── joe/               — Male voice (VoiceDesign-sourced, slot 3001)
│       ├── ref.wav              — original VoiceDesign reference
│       ├── ref_cleaned.wav      — DeepFilter + LUFS normalized
│       ├── candidates/    — 28 voice design candidates + index.txt
│       └── training-data/ — 385 clips (needs regeneration at temp 0.85)
│           ├── audio/           — enhanced clips (new pipeline)
│           ├── audio-original/  — raw 1.7B cloner output
│           └── train.jsonl      — full manifest (not yet curated)
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

## Training & Quantization

**Read `docs/training-runbook.md` first.** It is the authoritative, complete reference for all training: recipe, multi-voice, GPU runbook, data pipeline, quantization, quality targets, and hard-won lessons. Everything below is a quick summary.

- **Recipe:** lr=1e-7, 2 epochs, batch_size=2, bf16. Loss stays ~12-15 (correct).
- **Single-voice:** `training/sft_12hz_patched.py` — text_projection patch for 0.6B.
- **Multi-voice:** `training/sft_12hz_multivoice.py` — per-voice JSONL with `voice_name` field, cached embedding injection (bug fixed 2026-04-27).
- **Quantization:** 6-bit affine g64 via `mlx_audio.convert`. **Must manually copy `speech_tokenizer/model.safetensors` after** (converter bug).
- **GPU:** Vast.ai, 3090+ ($0.12-0.50/hr), `remote_setup.sh` + `remote_train_multivoice.sh`.

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

## Voice Data Pipeline & GPU Training

**All in `docs/training-runbook.md`.** Covers voice design, data generation, enhancement, curation, GPU setup, training, quantization, and all environment gotchas. Read it fully before any training work.

Quick reference tools:
```bash
# Enhance clips
.venv-enhance-audio/bin/python tools/enhance_clips.py --voice <name> --gender <male|female>
# Auto-curate (report, then --apply)
.venv-enhance-audio/bin/python tools/auto_curate.py --voice <name>
# Analyze quality
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source <dir> --label <name> --n 80
```

## Community References

- **rekuenkdr** — anonymous hobbyist, posted the winning lr=1e-7 recipe (Issue #39), built the two-phase streaming fork (72 stars). Also works on "OVA" (local voice assistant pipeline).
- **Our GitHub comment** documenting findings: https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999

## HuggingFace Release Target

- `sentium/holler-0.6b` (bf16 — full precision, source for custom quantization)
- `sentium/holler-0.6b-6bit` (affine 6-bit g64 — the pick, best quality-to-size ratio)

## Hard-Won Lessons

Training lessons are in `docs/training-runbook.md`. Inference lessons below (see also `RESEARCH.md` for full 27-experiment log):

- Custom generate loop = 2.3x faster than mlx-audio's streaming mode (no per-chunk `mx.clear_cache()` thrashing)
- Code predictor is 71% of generation time (15 sequential 5-layer transformers per token)
- 12 of 16 codebooks is the sweet spot — skip 13-16 for 18% speed gain, negligible quality loss
- Two-phase streaming: first chunk at 3 tokens (~120ms TTFA), then 40-token chunks
- `psutil RSS` is garbage for MLX memory — use `mx.metal.get_active_memory()`
- Leading 220ms silence is architectural (all codec LMs) — trim at inference or overlap with LLM streaming
- On GPU: `faster-qwen3-tts` (pip) gives ~3x speedup via CUDA graphs (kernel launch overhead is the bottleneck, not compute)

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
