# Holler

Open-source American English voice pack for Qwen3-TTS 0.6B. By Sentium.

**What this is:** A fine-tuned Qwen3-TTS-12Hz-0.6B model with 30 high-quality American English voices, optimized for local inference on Apple Silicon via mlx-audio. ~139ms TTFA streaming, ~1.7GB RAM at 6-bit. The fastest high-quality local TTS available on Mac.

**What this is not:** A new TTS architecture. This is a fine-tune of Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache 2.0) with better English voices and a fully open training pipeline. All credit for the base model goes to the Qwen team at Alibaba. Our contribution is the voices, the Mac-focused inference setup, and the open training pipeline.

**License:** The base Qwen3-TTS model is Apache 2.0. Our fine-tune scripts and voices will also be Apache 2.0. All releases must include proper attribution to Qwen/Alibaba.

**Focus:** Mac. Training data generation runs locally on Apple Silicon via mlx-audio. Training (SFT) requires a CUDA GPU (Vast.ai). Inference target is mlx-audio on M-series Macs.

## HARD RULES

- **NEVER delete checkpoints without explicit confirmation from Chris.** Not during cleanup, not during session wrap, not ever. Checkpoints represent hours of GPU time and are irreplaceable once gone. Ask before deleting. This includes bf16 originals, quantized copies, and anything in `checkpoints/`.

**Origin:** Started as the voice component of [ivi](https://ivi.computer), a macOS notch AI assistant by Sentium. We open-sourced it because Qwen3-TTS is SOTA for local inference but ships with only 2 mediocre English voices — and nothing else fills that gap.

**Session logs:** All holler session logs go in the parent ivi repo at `ivi/logs/`, not in `holler/logs/`. Old session logs have been moved there already. `holler/logs/runs/` still holds raw training/inference output logs.

## Current State (2026-05-11)

- **Recipe:** lr=5e-7, cosine warmup (scheduler fix: `total_steps ÷ grad_accum`), 2 epochs, batch_size=2, grad_accum=4, weight_decay=0.01, **embedding normalization** (L2-norm to 10.0). All training data at uniform -22 LUFS.
- **6-voice v12 (CURRENT):** `checkpoints/holler-6voice-v12-6bit/` and `holler-6voice-v12-bf16/`. Kit=3000, Dakota=3001, Nora=3002, Joe=3003, Oliver=3004, Tessa=3005. Benchmark: 90% clean raw at temp 0.7, every voice 9/10. bf16 sounds significantly better than 6-bit. Also on R2 at `r2:holler/checkpoints/holler-6voice-v12-e1/`.
- **v5 (backup):** `checkpoints/holler-6voice-v5-6bit/` and `v5-bf16/`. Previous best before embedding normalization. On R2 at `r2:holler/checkpoints/holler-6voice-v5-e1/`.
- **Training data on R2:** `r2:holler/training-data/` — all 6 voices at **uniform -22 LUFS** (clips + refs). Original long VoiceDesign refs (7-11s). rclone remote `r2-sentium` configured locally.
- **R2 instance cache:** `r2:holler/instance-cache/` — pip-cache.tar.gz + models-cache.tar.gz. Setup time ~5-8 min.
- **Quantization:** 6-bit affine g64 for shipping. bf16 for quality. 8-bit untested but promising middle ground. **Must copy entire `speech_tokenizer/` directory** (all config files + model.safetensors) after quantization — reuse canonical copy, never re-download.
- **Inference:** Temperature 0.7 (not 0.6) for better prosody. Soft clip knee 0.8.
- **Benchmark tool:** `tools/benchmark_quality.py` — Whisper medium word timestamps, gap analysis, strict WER. `--raw` and `--session` modes, `--temperature` flag, `--compare` for A/B.
- **Python venv:** `.venv` (Python 3.14, mlx-audio, mlx-whisper) and `.venv-enhance-audio` (Python 3.13, torch 2.6, parselmouth, DNSMOS, pyloudnorm).

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
│   │   └── training-data/ — 500 clips, 414 manually curated ✅
│   │       ├── audio/           — 500 enhanced clips
│   │       ├── audio-original/  — 500 raw clips from local generation
│   │       └── train.jsonl
│   ├── dakota/            — Male voice "Trail Guide" (VoiceDesign, slot 3001) ← CONFIRMED
│   │   ├── ref.wav        — dakota03_trail_guide
│   │   └── training-data/ — 500 clips, 374 manually curated ✅
│   │       ├── audio/           — 500 enhanced clips
│   │       ├── audio-original/  — 500 raw clips from local generation
│   │       └── train.jsonl
│   ├── katie/             — Female voice (DEV ONLY, not shipping)
│   │   └── training-data/ — 452 curated clips
│   ├── nora/              — Female voice (VoiceDesign, slot 3002)
│   │   └── training-data/ — 500 clips, manual curation in progress
│   ├── joe/               — Male voice (VoiceDesign, slot 3003)
│   │   └── training-data/ — 500 clips, manual curation in progress
│   ├── oliver/            — Male voice (VoiceDesign, slot 3004)
│   │   └── training-data/ — 500 clips, manually curated ✅
│   └── tessa/             — Female voice (VoiceDesign, slot 3005)
│       └── training-data/ — 520 clips (quotes v2), needs enhance + curate
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
├── Package.swift          — SPM manifest (HollerKit library + holler CLI)
├── Sources/HollerKit/     — Swift library: HollerModel, SpeechSession, StreamingPipeline
├── Sources/HollerCLI/     — Swift CLI: holler --text/--session/--benchmark/--talk
├── Tests/HollerKitTests/  — Unit tests (SentenceBuffer, SilenceAnalyzer, AudioPostProcessor)
├── logs/                  — DEPRECATED: session logs now live in ivi repo at ivi/logs/
│   ├── sessions/          — Old session logs (moved to ivi/logs/)
│   └── runs/              — Raw training/inference logs
└── docs/
    └── handover-lora-failure.md    — Historical: why LoRA doesn't work
```

## Training & Quantization

**Read `docs/training-runbook.md` first.** It is the authoritative, complete reference for all training: recipe, multi-voice, GPU runbook, data pipeline, quantization, quality targets, and hard-won lessons. Everything below is a quick summary.

- **Recipe:** lr=5e-7, cosine warmup, 2 epochs, batch_size=2, grad_accum=4, weight_decay=0.01, **embedding normalization** (L2-norm to 10.0). All data at uniform -22 LUFS. See training-runbook.md for full details.
- **Embedding normalization:** L2-normalize all ECAPA-TDNN speaker embeddings to magnitude 10.0 before injection during training. Strips loudness from embeddings while preserving voice identity (direction). Without this, voices with higher embedding norms produce hotter output, and LUFS changes cascade unpredictably through shared weights. 3 lines in `sft_12hz_multivoice.py`.
- **Cosine scheduler fix:** `total_optimizer_steps = (steps_per_epoch × num_epochs) // grad_accum`. The Accelerator steps the scheduler once per optimizer step (every `grad_accum` batches), so `total_steps` must reflect actual optimizer steps, not dataloader batches. Without this fix, cosine barely decays (completes only 25% of curve).
- **Single-voice:** `training/sft_12hz.py`
- **Multi-voice:** `training/sft_12hz_multivoice.py` — per-voice JSONL with `voice_name` field, cached embedding injection with L2 normalization.
- **Quantization:** 6-bit affine g64 via `mlx_audio.convert`. **Must copy entire `speech_tokenizer/` directory** (config.json, configuration.json, preprocessor_config.json, model.safetensors) from a canonical source — reuse the same copy across all checkpoints, never re-download. The speech tokenizer is identical across all checkpoints.
- **GPU (training only):** Vast.ai, 3090+ ($0.12-0.25/hr). **Requires driver ≥550** for CUDA 12.4 Docker image. `remote_setup.sh` + `remote_train_multivoice.sh`. Always SCP the training script fresh — never pull from R2.

## Inference Architecture (updated 2026-04-24)

**Holler is a model, not an inference library.** Ship weights on HuggingFace, users run via our fast inference server (`inference/server.py`). Uses mlx-audio for model loading and codec decoding, with a custom generate loop for 2.3x faster streaming.

### Quick Start

```bash
# 1. Create venv and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run server (auto-downloads sentium/holler-0.6b-6bit from HuggingFace on first run)
python3 inference/server.py
python3 inference/server.py -c path/to/local/checkpoint --voice kit
# → http://localhost:8100

# 4. Test
curl "http://localhost:8100/tts?text=Hello+world" -o test.wav
curl http://localhost:8100/benchmark
```

### API

```
POST /speak      — streaming float32 PCM (24kHz mono), chunked transfer encoding
  Body: {"text": "...", "voice": "kit", "temperature": 0.6, "n_codebooks": 12}
GET  /tts?text=  — returns complete WAV file
GET  /benchmark  — runs 6-sentence benchmark, returns text report
GET  /health     — {"status": "ok", "model": "...", "voices": [...]}
GET  /           — browser test UI (disable with --no-ui)
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
4. Full details: see `ivi/logs/2026-04-24-holler-rtf-optimization-research.md` (27 experiments logged)

## Inference Pipeline

**Stack:** `inference/server.py` → custom generate loop → mlx-audio model/codec → MLX → Metal GPU

1. `mlx_audio.tts.load(checkpoint_path)` loads model + speech tokenizer
2. Custom generate loop: talker LLM → code predictor → codec tokens at 12Hz
3. Chunked streaming decode: first chunk at 3 tokens (~120ms TTFA), then 40-token chunks
4. HTTP chunked transfer encoding streams float32 PCM to client
5. `mx.clear_cache()` after each complete generation to prevent memory accumulation

**Live demo:** `inference/live_demo.py` — HTTP server on port 8099, type text in browser, audio plays from Mac speakers.

**Python venv:** `.venv` exists (Python 3.13). Has mlx-audio for inference, plus clearvoice/deepfilternet/noisereduce for audio enhancement. torch 2.6 + torchaudio 2.6 (pinned for DeepFilterNet compatibility).

## HollerKit (Swift Package) — Phase 2B Complete

Native Swift TTS library at repo root (`Sources/HollerKit/`). Depends on `sentiuminc/mlx-audio-swift` (git URL, tag `0.31.3-holler.3`).

**Build & Run:**
```bash
./build.sh              # xcodebuild + copies binary to repo root (~3 min first, seconds after)
./build.sh --clean      # nuke .build/.swiftpm first (use when xcodebuild gets confused)
./holler --text 'Hello world' --talk
```
**⚠️ `swift build` compiles but the binary WILL NOT RUN** — mlx-swift's Metal shaders are only compiled by xcodebuild, not SPM. Always use `./build.sh`. This is an mlx-swift architectural limitation (TN3133).

**Architecture:**
```
HollerModel.stream("text", voice:)  →  InferenceActor  →  mlx-audio-swift generateStream()
         ↓                                    ↓                        ↓
   AsyncThrowingStream<Chunk>       StreamingPipeline          Qwen3TTSModel + codec decoder
         ↓                          (silence trim, abort)
   Consumer (app/CLI)
```

**SpeechSession (LLM integration):**
```
session.feed(token)  →  SentenceBuffer  →  per-sentence generation  →  session.audio stream
                        (accumulate,        (carryover KV cache,        (chunks as they're ready)
                         split on . ! ?)     retry on failure)
```

**Key files:**
- `HollerModel.swift` — public API: `load()`, `stream()`, `synthesize()`, `makeSession()`
- `SpeechSession.swift` — LLM integration: `feed()`, `finish()`, `cancel()`, `audio` stream
- `SentenceBuffer.swift` — text accumulation + sentence boundary detection
- `StreamingPipeline.swift` — silence onset trim, abort
- `InferenceActor.swift` — serialized MLX access, streaming decode
- `RetryController.swift` — retry evaluation (too short, no speech, give up)
- `HollerConfiguration.swift` — all tunables + `log` closure for debug
- `HollerCLIApp.swift` — CLI: `--text`, `--session`, `--benchmark`, `--talk`, `--debug`

**Performance (release build, M1 Pro):** TTFA ~360ms (includes silence trim), RTF ~0.49.

**Known issues (2026-05-02):**
1. **Carryover artifacts** — garbled vowels/stuttering ("aauuhha" or "kkk") and rumble on deep carryover. Two causes found: (a) retry cache poisoning — failed attempt's talker cache persisted in `_carry_over_state` while decoder got reset by retry, fixed in Python server.py (Swift already had fix). (b) KV cache degradation beyond ~150 tokens — model wasn't trained on long sequences, quality drops after 4-5 carryover sentences. No fix yet; may need cache size cap or trimming. Speech detection threshold raised 0.007→0.01 to catch rumble via existing silence abort.
2. **Stochastic EOS cutoff** — model hits EOS 1-2 tokens early ~20% of the time on short sentences with heavy carryover. Model-level, not pipeline-fixable.
3. **Long rumble artifact** — partially mitigated by threshold bump (0.007→0.01) and retry cache fix. Remaining cases are from KV cache degradation on deep carryover (see #1b).

## Known: 220ms Leading Silence

The first 2-3 streaming chunks (~220ms) from any Qwen3-TTS generation are near-silence. This is a **known codec LM architecture behavior** — confirmed across Qwen3-TTS, CosyVoice, VALL-E descendants. The codec decoder needs initial context tokens before producing meaningful audio.

Our training data also has 25-212ms of leading silence per clip, which reinforces this behavior. Future training runs should trim leading silence from training clips.

**Mitigations (not yet implemented):**
- Trim leading silence from training data before next training run
- For ivi integration: sentence-level streaming from LLM overlaps codec warmup with text generation
- Study `rekuenkdr/Qwen3-TTS-streaming` — two-phase streaming fork that buffers past the silence before first emit (208ms first audible vs 570ms baseline)

## LUFS, Embeddings & Training Dynamics (2026-05-11)

**The ECAPA-TDNN speaker encoder bakes loudness into voice embeddings.** Voices with naturally louder refs produce embeddings with higher norms, causing the model to generate hotter output. This effect cascades through shared transformer weights: changing one voice's LUFS destabilizes other voices unpredictably ("whack-a-mole").

**Solution: embedding normalization.** L2-normalize all speaker embeddings to magnitude 10.0 before injection during training. This strips loudness information while preserving voice identity (the direction of the embedding vector). Combined with uniform -22 LUFS training data, this produces the most balanced output levels across voices.

**Key findings from 12 training runs (2026-05-11):**
- Uniform LUFS across all voices is essential — per-voice LUFS adjustments cause unpredictable cross-voice interference
- Ref audio length correlates with voice stability: 7-11s refs → consistent voice identity, 4-5s refs → variable
- Ref choice directly affects output levels — different ref from same training data produces different loudness
- Training is non-deterministic — same data, different run, different result (run-to-run variance)
- Larger batch size (bs=8) did NOT fix voice merging — embedding normalization did
- Weight decay 0.05 was too aggressive (killed voice character), 0.01 is the sweet spot
- 3 epochs overfit, 2 epochs is optimal with the cosine scheduler fix
- Temperature 0.7 at inference gives better prosody than 0.6 (training data was generated at 0.85)
- bf16 sounds significantly better than 6-bit — 8-bit quantization untested but promising

**Nora runs ~3 dB hotter than training data** even with embedding normalization. Her voice identity inherently encodes "loud" in the ECAPA-TDNN. Manageable with soft clipping at inference.

**Key numbers:**
- bf16: RTF 0.67-0.81, 2378MB Metal RAM — real-time, best quality
- 6-bit: RTF 0.50-0.57, 1744MB Metal RAM — fast, good quality
- Size: bf16 2.3GB, 6-bit 1.7GB on disk

## Voice Data Pipeline & Training

**All in `docs/training-runbook.md`.** Covers voice design, data generation (local), enhancement, curation, GPU training, quantization, and all environment gotchas. Read it fully before any training work.

Quick reference tools:
```bash
# Enhance training data (current standard for all synthetic TTS output)
.venv-enhance-audio/bin/python tools/enhance_clean.py --voice <name>
# Analyze quality
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source <dir> --label <name> --n 80
# Manual curation tinder
.venv-enhance-audio/bin/python tools/curate_clips.py --voice <name>
```

**Enhancement pipeline:** `enhance_clean.py` is the current standard for all Holler training data. Pipeline: trim → K-weighted LUFS → IIR notch at 5500Hz Q=3.0. **Current LUFS target: -22** (changed from -20 on 2026-05-11). All voices at uniform LUFS — no per-voice adjustments. `enhance_clips.py` and `enhance_voicedesign.py` are deprecated — see their headers for why.

## Community References

- **rekuenkdr** — anonymous hobbyist, posted the winning lr=1e-7 recipe (Issue #39), built the two-phase streaming fork (72 stars). Also works on "OVA" (local voice assistant pipeline).
- **Our GitHub comment** documenting findings: https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999

## HuggingFace Release Target

- `sentium/holler-0.6b` (bf16 — full precision, source for custom quantization)
- `sentium/holler-0.6b-6bit` (affine 6-bit g64 — the pick, best quality-to-size ratio)

## Hard-Won Lessons

Training lessons are in `docs/training-runbook.md`. Inference lessons below (see also `ivi/logs/2026-04-24-holler-rtf-optimization-research.md` for full 27-experiment log):

- Custom generate loop = 2.3x faster than mlx-audio's streaming mode (no per-chunk `mx.clear_cache()` thrashing)
- Code predictor is 71% of generation time (15 sequential 5-layer transformers per token)
- 12 of 16 codebooks is the sweet spot — skip 13-16 for 18% speed gain, negligible quality loss
- Two-phase streaming: first chunk at 3 tokens (~120ms TTFA), then 40-token chunks
- `psutil RSS` is garbage for MLX memory — use `mx.metal.get_active_memory()`
- Leading 220ms silence is architectural (all codec LMs) — trim at inference or overlap with LLM streaming


## What's NOT Known / Unresolved

- Whether 8-bit quantization is a viable middle ground (better quality than 6-bit, smaller than bf16)
- Whether joint training at 30 voices holds up (tested at 2 and 6)
- Whether trimming leading silence from training data reduces the 220ms codec warmup
- EOS failure ~2-4% of the time (model-level) — mitigated by safety cap but not eliminated
- Why Nora consistently runs ~3 dB hot even with embedding normalization — may be inherent to her voice identity

## What's Next

### Immediate

1. ~~**Cosine scheduler fix**~~ — ✅ DONE (v2). `total_steps ÷ grad_accum`.
2. ~~**Embedding normalization**~~ — ✅ DONE (v11). L2-norm to 10.0.
3. ~~**Proper benchmark tool**~~ — ✅ DONE. `tools/benchmark_quality.py` with Whisper word timestamps.
4. ~~**Release-quality 6-voice model**~~ — ✅ DONE. v12 checkpoint.
5. **Test 8-bit quantization** on v12 — could be the sweet spot between 6-bit speed and bf16 quality.
6. **Generate more Tessa training data** — 331 clips is the fewest. 500+ would improve consistency.

### HollerKit (Swift)

7. ~~**Fix decoder stuttering on carryover**~~ — Partially fixed. Remaining: KV cache degradation on deep carryover (5+ sentences).
8. **Investigate long rumble artifact** — occasional generation produces seconds of low rumble instead of speech.
9. **Stochastic EOS cutoff** — model hits EOS 1-2 tokens early.

### Voices & Training

10. **Pick remaining voices** — from roster in `voices/VOICES.md`. Generate training data, enhance, curate for each.
11. **Scale to 10+ voices** — embedding normalization should help maintain voice separation at scale.
12. **Explore MLX training** — eliminates GPU rental. mlx-audio has the model; adding `nn.value_and_grad()` could be a 1-day project.

### Release

13. **HuggingFace release** under `sentium/` — bf16 + 6-bit (+ 8-bit if it tests well).
14. **PR to mlx-audio** — reduce `mx.clear_cache()` frequency in their streaming loop.
