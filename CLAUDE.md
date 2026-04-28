# Holler

Open-source American English voice pack for Qwen3-TTS 0.6B. By Sentium.

**What this is:** A fine-tuned Qwen3-TTS-12Hz-0.6B model with 30 high-quality American English voices, optimized for local inference on Apple Silicon via mlx-audio. ~139ms TTFA streaming, ~1.7GB RAM at 6-bit. The fastest high-quality local TTS available on Mac.

**What this is not:** A new TTS architecture. This is a fine-tune of Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) with better English voices and a fully open training pipeline. All credit for the base model goes to the Qwen team at Alibaba. Our contribution is the voices, the Mac-focused inference setup, and the open training pipeline.

**License:** The base Qwen3-TTS model is Apache 2.0. Our fine-tune scripts and voices will also be Apache 2.0. All releases must include proper attribution to Qwen/Alibaba.

**Focus:** Mac. Local inference on Apple Silicon specifically. The training can happen on any CUDA GPU, but the inference target is mlx-audio on M-series Macs.

**Origin:** Started as the voice component of [ivi](https://ivi.computer), a macOS notch AI assistant by Sentium. We open-sourced it because Qwen3-TTS is SOTA for local inference but ships with only 2 mediocre English voices — and nothing else fills that gap.

**Session logs:** All holler session logs go in the parent ivi repo at `ivi/logs/`, not in `holler/logs/`. Old session logs have been moved there already. `holler/logs/runs/` still holds raw training/inference output logs.

## Current State (2026-04-28)

- **Recipe:** Proven. lr=1e-7, 2 epochs, text_projection patch only. Now with `--save_every_steps` for fractional epoch checkpoints.
- **Katie:** DEV VOICE ONLY. Not shipping. Was used to develop the pipeline. Checkpoint at `checkpoints/katie-v6/` (bf16).
- **Kit + Dakota (current):** 2-voice checkpoint at `checkpoints/holler-kit-dakota-6bit/`. Kit=3000, Dakota=3001. Sounds good.
- **Voices confirmed for Holler v1:** Kit (Prism), Dakota (Trail Guide), plus 8 more TBD from 22 curated candidates.
- **Training data generated on GPU:** Vanilla `qwen-tts` on Vast.ai 3090. 0.7x RTF. Scripts at `training/remote_generate_training_data.py` + `training/corpus.json`. **Do NOT use `faster-qwen3-tts` for voice cloning** — it breaks voice identity.
- **Quantization:** 6-bit affine g64 is the pick.
- **Inference runtime:** Custom fast inference server (`inference/server.py`). RTF 0.38, TTFA 139ms on 6-bit.
- **Auto-curate thresholds need male adjustment:** Current thresholds reject 100% of male voice clips. HNR < 14 and harshness > 2% are female-calibrated. Proposed male: HNR > 8, harshness < 5%, peak > -2.
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
│   ├── kit/               — Androgynous voice "Prism" (VoiceDesign, slot 3000) ← CONFIRMED
│   │   ├── ref.wav        — kit08_prism_clear
│   │   └── training-data/ — 500 clips generated on GPU (vanilla qwen-tts)
│   │       ├── audio/           — 500 enhanced clips
│   │       ├── audio-original/  — 500 raw clips from 3090
│   │       └── train.jsonl
│   ├── dakota/            — Male voice "Trail Guide" (VoiceDesign, slot 3001) ← CONFIRMED
│   │   ├── ref.wav        — dakota03_trail_guide
│   │   └── training-data/ — 400 clips (clips 101-500, first 100 were bad faster-qwen3-tts)
│   │       ├── audio/           — 400 enhanced clips
│   │       ├── audio-original/  — 400 raw clips from 3090
│   │       └── train.jsonl
│   ├── katie/             — Female voice (DEV ONLY, not shipping)
│   │   └── training-data/ — 452 curated clips
│   ├── nora/              — Female voice (VoiceDesign, slot 3002)
│   │   └── training-data/ — 500 clips, 366 curated
│   └── joe/               — Male voice (VoiceDesign, slot 3003)
│       └── training-data/ — 500 clips, 53 curated (89% rejection — needs regen or relaxed thresholds)
├── checkpoints/           — Model checkpoints (not in git — large)
│   ├── katie-v6/          — 1.7GB bf16 (dev voice, not shipping)
│   ├── nora-joe-v1/       — 2.3GB bf16, multi-voice (reference for training quality)
│   ├── nora-joe-v1-6bit/  — 1.7GB 6-bit affine
│   ├── holler-kit-dakota/     — 1.7GB bf16, Kit+Dakota 2-voice
│   └── holler-kit-dakota-6bit/ — 1.1GB 6-bit affine ← CURRENT
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
# Enhance training data (cloned from 1.7B-Base — may have noise from ref)
.venv-enhance-audio/bin/python tools/enhance_clips.py --voice <name> --gender <male|female>
# Enhance VoiceDesign output (already-clean TTS — no DeepFilter, no STFT)
.venv-enhance-audio/bin/python tools/enhance_voicedesign.py --input <dir> --output <dir>
# Auto-curate (report, then --apply)
.venv-enhance-audio/bin/python tools/auto_curate.py --voice <name>
# Analyze quality
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source <dir> --label <name> --n 80
```

**Two enhancement pipelines exist — use the right one:**
- `enhance_clips.py` — for training data cloned from real-world ref audio. Full pipeline: DeepFilter → LUFS → STFT de-ess → presence.
- `enhance_voicedesign.py` — for VoiceDesign candidate output. Lightweight: trim → LUFS → IIR notch. DeepFilter and STFT processing create chirping/musical noise artifacts on already-clean synthetic audio.

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

1. ~~**Wire Katie v6 into ivi**~~ — ✅ DONE.
2. ~~**Try alternative quantization**~~ — ✅ DONE. 6-bit affine g64 wins.
3. ~~**Enhance training audio**~~ — ✅ Pipeline proven.
4. ~~**GPU training data generation**~~ — ✅ DONE. Vanilla `qwen-tts` on Vast.ai 3090. 0.7x RTF (~25 min/voice). Scripts ready.
5. ~~**Kit + Dakota trained**~~ — ✅ DONE. 2-voice checkpoint sounds good. Needs Tinder curation pass.
6. **Add male auto-curate thresholds** — current thresholds reject 100% of male clips. Proposed: HNR > 8, harshness < 5%, peak > -2.
7. **Tinder curation** — Kit (500 clips) and Dakota (400 clips) need manual listening pass.
8. **Pick remaining 8 voices** — from 22 curated candidates. Generate training data, enhance, curate for each.
9. **Full 10-voice train** — once all voices curated, single multi-voice training run.
10. **Explore MLX training** — research shows it's feasible. mlx-audio has the model already; adding `nn.value_and_grad()` could be a 1-day project. Eliminates GPU rental.
11. **HuggingFace release** under `sentium/` with full docs — bf16 + 6-bit affine only.
12. **PR to mlx-audio** — reduce `mx.clear_cache()` frequency in their streaming loop.
