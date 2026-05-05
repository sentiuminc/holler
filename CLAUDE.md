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

## Current State (2026-05-05)

- **Recipe:** lr=5e-7 with cosine warmup, 2 epochs, full-model SFT. See "Training Recipe" section. Previous lr=1e-7 was too low for 6 voices.
- **6-voice v1 (CURRENT):** `checkpoints/holler-6voice-v1-6bit/`. Kit=3000, Dakota=3001, Nora=3002, Joe=3003, Oliver=3004, Tessa=3005. Quality is good with minimal artifacts. Best 6-voice checkpoint so far. Also on R2 at `r2:holler/checkpoints/holler-base-5e7-cosine-2ep-e1/`.
- **Katie:** DEV VOICE ONLY. Not shipping. Checkpoint at `checkpoints/katie-v6/` (bf16).
- **Kit + Dakota (reference):** 2-voice 6-bit at `checkpoints/holler-kit-dakota-6bit/`. Kit=3000, Dakota=3001. Still the quality bar for single-voice fidelity.
- **Training data on R2:** `r2:holler/training-data/` — all 6 voices' audio clips + refs + combined JSONL. Kit/Dakota/Oliver/Tessa at -20 LUFS. Nora/Joe at -22 LUFS (run hotter, need compensation). rclone remote `r2-sentium` configured locally.
- **R2 instance cache:** `r2:holler/instance-cache/` — pip-cache.tar.gz + models-cache.tar.gz + CustomVoice model. Setup time ~5-8 min (was 15-25 min).
- **Training data generated locally:** `tools/generate_training_data.py` (v1, generic texts) and `tools/generate_training_data_quotes.py` (v2, curated quotes). Both use `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` via mlx-audio on Mac. Use quotes version for new voices.
- **Quantization:** 6-bit affine g64 is the target. See "LUFS & Quantization" for current quality issue.
- **Inference runtime (Python):** Custom fast inference server (`inference/server.py`). RTF 0.38, TTFA 139ms on 6-bit (2-voice). 6-voice 6-bit: RTF 0.69, TTFA 265ms. bf16: RTF 1.07 (too slow for real-time).
- **Inference runtime (Swift):** HollerKit library at repo root (`Sources/HollerKit/`). Phase 2B complete. RTF 0.49, TTFA 360ms (release build). See "HollerKit (Swift)" section below.
- **Training data tools:** Pipeline — `tools/enhance_clean.py` (current standard), `tools/analyze_voice_quality.py`, `tools/curate_clips.py` (tinder UI). `enhance_clips.py`, `enhance_voicedesign.py`, and `auto_curate.py` are deprecated (see deprecation headers in each file).
- **Alternative: real speech datasets.** VCTK dataset was tested — speech is slow, boring, and not usable quality for our purposes despite being "studio" recorded. Not a viable source. For real-voice training, podcast/YouTube clips with natural energetic speech are the better approach.
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

- **Recipe:** lr=5e-7 with cosine warmup, 2 epochs, batch_size=2, bf16. Loss ~12-14. Previous lr=1e-7 was too low for multi-voice (flat prosody, airy). See training-runbook.md for full details and 2026-05-05 learnings.
- **Single-voice:** `training/sft_12hz.py`
- **Multi-voice:** `training/sft_12hz_multivoice.py` — per-voice JSONL with `voice_name` field, cached embedding injection (bug fixed 2026-04-27).
- **Quantization:** 6-bit affine g64 via `mlx_audio.convert`. **Must manually copy `speech_tokenizer/model.safetensors` after** (converter bug).
- **GPU (training only):** Vast.ai, 3090+ ($0.12-0.50/hr), `remote_setup.sh` + `remote_train_multivoice.sh`.

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

## LUFS & Quantization Findings (2026-05-04)

**Problem:** 6-voice model sounds great at bf16 but degrades at 6-bit. Old 2-voice 6-bit sounds great.

**Root cause: voice embeddings, not quantization.** The quantized transformer weights are byte-identical between 2v and 6v (882/900 layers). Training at lr=1e-7 moves weights by ~0.00004 — smaller than one 6-bit quantization step, so they round to the same values. The only differences are in the 18 bf16 layers (codec_embedding voice slots, text_embedding, code_predictor embeddings).

**Proof:** Swapped old 2v kit/dakota embeddings into the 6v 6-bit model → output matched 2v quality exactly. The embeddings drive the quality difference, not the quantization.

**Why the embeddings differ:** The 6-voice training used refs normalized to -18 LUFS. The old 2-voice training used natural refs at -21/-23 RMS. The ECAPA-TDNN speaker encoder bakes loudness into the embedding. Hotter refs → embedding encodes "be louder" → the quantized transformer can't reproduce it as cleanly.

**Hypothesis:** The -18 LUFS normalization caused the embedding quality difference. Unproven — could also be joint training dynamics, embedding crowding in adjacent slots, or something else. Testing -20 LUFS next as one variable to eliminate.

**Current state:** Re-normalized all training clips and refs to -20 LUFS. Data is on R2. Needs retrain. Plan to add fixed per-voice gain in inference server to boost output to listening level.

**Key numbers:**
- bf16: RTF 1.07, TTFA 503ms, 2378MB Metal RAM — too slow
- 6-bit: RTF 0.69, TTFA 265ms, 1744MB Metal RAM — fast enough
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

**Enhancement pipeline:** `enhance_clean.py` is the current standard for all Holler training data. Pipeline: trim → K-weighted LUFS → IIR notch at 5500Hz Q=3.0. **Current LUFS target: -20** (changed from -18 on 2026-05-04, see LUFS findings above). `enhance_clips.py` and `enhance_voicedesign.py` are deprecated — see their headers for why.

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

- Whether -20 LUFS training resolves the 6-bit quality degradation (hypothesis — retrain pending, not proven)
- Whether joint training at 30 voices holds up (tested at 2 and 6, 6 has embedding quality issues)
- Whether sequential training (one voice at a time, cumulative checkpoints) works better than joint
- Optimal voice count per training run
- Whether mixed precision (e.g. higher bits for code_predictor/lm_head, lower for talker) could improve quality
- Whether trimming leading silence from training data reduces the 220ms codec warmup
- EOS failure ~2-4% of the time (model-level, both 12cb and 16cb) — mitigated by safety cap but not eliminated
- Server crashes during long idle — needs process supervisor for production

## What's Next

### Immediate

1. ~~**Retrain 6-voice at -20 LUFS**~~ — ✅ DONE. lr=5e-7 + cosine warmup + 2 epochs. Checkpoint: `holler-6voice-v1-6bit`.
2. **Reduce remaining artifacts** — try fixing cosine scheduler (account for grad_accum in total_steps), stratified batching, or weight decay tuning.
3. **Fix Nora's hot output** — try -25 LUFS training data or per-voice inference gain.
4. **Run proper artifact rate measurement** — 20+ samples per voice on the winning checkpoint.
5. **Test longer session mode** — multi-paragraph carryover stability.

### HollerKit (Swift)

4. ~~**Fix decoder stuttering on carryover**~~ — Partially fixed. Root cause: retry cache poisoning. Remaining: KV cache degradation on deep carryover (5+ sentences).
5. ~~**Move Package.swift to repo root**~~ — ✅ DONE.
6. ~~**Push holler to sentiuminc/holler**~~ — ✅ DONE.
7. **Investigate long rumble artifact** — occasional generation produces seconds of low rumble instead of speech.
8. **Stochastic EOS cutoff** — model hits EOS 1-2 tokens early. Needs STT round-trip detection method.

### Voices & Training

9. ~~**All 6 voices curated**~~ — ✅ Kit 414, Dakota 374, Nora 394, Joe 367, Oliver 360, Tessa 331.
10. **Pick remaining voices** — from roster in `voices/VOICES.md`. Generate training data, enhance, curate for each.
11. **Full multi-voice train** — scale to 10+ voices once 6-voice quality is proven at 6-bit.
12. **Explore MLX training** — eliminates GPU rental. mlx-audio has the model; adding `nn.value_and_grad()` could be a 1-day project.

### Release

13. **HuggingFace release** under `sentium/` with full docs — bf16 + 6-bit affine only.
14. **PR to mlx-audio** — reduce `mx.clear_cache()` frequency in their streaming loop.
