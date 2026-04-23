# Session Log: 2026-04-22 — Holler Project Setup + Quantization

**Project:** ivi → holler (new project created this session)
**What happened:** Reflected on yesterday's TTS work, fixed overconfident documentation, created the Holler project (`~/Desktop/Files/AI/holler/`), moved all TTS assets from ivi + Downloads, quantized Katie v6 to 8-bit and 4-bit, benchmarked all three precisions, built a live demo, and discovered the 220ms leading silence behavior.

---

## What I Did

### 1. Reflected on yesterday's v6/v7 TTS sessions
- Read all session logs, README, TODO, memory files from 2026-04-21
- Identified that the previous session stated unverified hypotheses as confirmed conclusions:
  - "Katie noise must come from shared decoder weight shift" → unverified, same-texts comparison was never done
  - "Joe clips because +9% embedding norm" → correlation, not proven causation
  - "Temperature is a huge inference-time knob" → single uncontrolled comparison across different engines
  - "MLX at 0.6 had 1/10 clips clipping" → user reported 3, previous session ignored this
- Fixed language across 5 files: session logs, README, TODO.md, memory file

### 2. Updated Vast instance references
- Both instances (35364783 Kansas, 35377474 Croatia) are deleted
- Updated all references across all docs to reflect this
- Noted: to retrain, rent fresh instance and follow runbook

### 3. Named the open-source project: Holler
- 30 American English voices for Qwen3-TTS 0.6B
- HuggingFace: `sentium/holler-0.6b-4bit` (and other precisions)
- Apache 2.0, Mac-focused, full open training pipeline
- Positioned to fill a real gap: Qwen3-TTS is SOTA local TTS but has only 2 mediocre English voices

### 4. Created and organized `~/Desktop/Files/AI/holler/`
- Moved all scripts from `ivi/audio-test/qwen3-tts-finetune/` into `training/`, `inference/`, `tools/`
- Moved checkpoints from Downloads: katie-v6 (2.3GB) + katie-joe-v7 (2.3GB)
- Moved training data: katie (385 clips, 96MB) + joe (385 clips, 95MB)
- Moved voice candidates, samples, reference audio, logs
- Copied session logs (kept in ivi too since they're ivi history)
- Deleted ~4.7GB of broken old checkpoints from Downloads (v1 runs with wrong LR)
- Git initialized on `main`, wrote `.gitignore` for large files
- Wrote comprehensive `CLAUDE.md`

### 5. Quantized Katie v6
- Used mlx-audio's own converter (`mlx_audio.convert.convert()`) — NOT `mlx_lm.convert` which errors on qwen3_tts model type
- 8-bit: affine, group_size 64, 11.4 effective bits/weight (critical layers kept full precision)
- 4-bit: affine, group_size 64, 8.9 effective bits/weight
- The converter is smart — keeps codec_embedding, speaker embeddings, small layers at full precision

### 6. Benchmarked all three precisions
Ran 10 random conversational texts through each, measuring TTFA (streaming), Metal memory, peaks.

| | bf16 | 8-bit | 4-bit |
|--|------|-------|-------|
| Disk | 1.7GB | 1.2GB | 960MB |
| Metal RAM (peak) | 2,703MB | 2,203MB | 1,936MB |
| Activity Monitor | ~3GB | ~2.4GB | ~2GB |
| TTFA median | 101ms | 79ms | 79ms |

Chris picked **4-bit** — sounds more alive than 8-bit. All precisions sound good, no audible clipping.

### 7. Built live demo (`inference/live_demo.py`)
- HTTP server on port 8099, browser UI, type text and hear it spoken
- Optimized: pre-opened audio stream at boot (not per-request), sounddevice with moderate latency
- Chris tested it live, confirmed ~100-130ms perceived TTFA, quality is great
- Learned: `psutil.rss` gives garbage for MLX memory — must use `mx.metal.get_active_memory()`
- Learned: `latency='low'` on sounddevice causes stuttering — buffer too small for inter-chunk gaps

### 8. Discovered 220ms leading silence
- First 2-3 streaming chunks from any generation are near-silence (peak < 0.02)
- Training data has 25-212ms of leading silence per clip — model learned to reproduce it
- Confirmed via researcher: this is a known codec LM architecture behavior (Qwen3-TTS, CosyVoice, VALL-E descendants)
- `rekuenkdr/Qwen3-TTS-streaming` fork addresses this with two-phase streaming (buffers past silence before first emit)
- Skipping silent chunks in post-processing saves ~80ms of playing dead air but doesn't reduce the 220ms generation time — not a real fix
- Real fixes: trim training data, overlap with LLM streaming, study two-phase approach

### 9. Researched rekuenkdr
- Anonymous GitHub hobbyist in local voice AI space
- No real name/company. Also works on "OVA" (Outrageous Voice Assistant), had a Cardano stake pool
- Posted the winning lr=1e-7 recipe, built the streaming fork (72 stars)

## What I Learned

**mlx-audio has its own converter that understands TTS architectures.** `mlx_lm.convert` is for LLMs only and errors on qwen3_tts. The mlx-audio converter selectively quantizes layers, keeping voice-critical ones at full precision.

**4-bit quantization works surprisingly well for voice.** Subjectively sounds better than 8-bit (more alive), despite marginally higher peaks on paper. The effective bit rate (8.9 bits) is higher than nominal because critical layers stay full.

**psutil is wrong for MLX memory.** Apple Silicon unified memory allocations via Metal don't show up correctly in process RSS. Use `mx.metal.get_active_memory()` for honest numbers.

**Low-latency audio streams cause stuttering.** The TTS model generates chunks every ~83ms. If the audio buffer is smaller than that, it drains between chunks → audible gaps. Need ~100ms buffer minimum.

**Leading silence is architectural AND data-driven.** The codec needs warmup tokens, AND our training data has leading silence. Both contribute. Trimming training data for the next run could help but won't eliminate it entirely.

**The real TTFA users feel is not what we measure.** Our benchmark says 79ms, but button-press-to-ears includes: HTTP overhead (~5ms) + model generation of silent tokens (~220ms) + audio buffer (~100ms) + the silence itself. The felt latency is ~300-500ms, not 79ms.

## For Next Time

**Work happens in `~/Desktop/Files/AI/holler/` from now on.** The CLAUDE.md there has the full state.

**Immediate next steps:**
1. Create holler's own Python venv (currently borrowing ivi's at `~/Desktop/Files/AI/ivi/audio-test/.venv-tts-bench/`)
2. Solve multi-voice quality (the v7 Katie+Joe issues)
3. Standardize GPU runbook for training from scratch
4. Design 28 more American voices

**Later:**
- Wire into ivi sidecar with sentence-chunked streaming
- Explore alternative quantization modes (mxfp4, mixed_4_6)
- Study rekuenkdr's two-phase streaming fork
- Find the HuggingFace repo Chris mentioned that improved model size/inference by 3x
- HuggingFace release under `sentium/`

## Files Changed

New project created at `~/Desktop/Files/AI/holler/` with full structure (see CLAUDE.md there for layout).

In ivi repo:
- `logs/2026-04-21-multivoice-katie-joe.md` — fixed overconfident root-cause language, updated instance refs
- `logs/2026-04-21-tts-finetune-success.md` — updated instance refs
- `audio-test/qwen3-tts-finetune/README.md` — fixed root-cause language, updated instance refs
- `TODO.md` — fixed quality issue descriptions, updated instance refs

In holler (new):
- `CLAUDE.md` — comprehensive project instructions
- `.gitignore`
- `inference/benchmark_katie_v6.py` — bf16 benchmark script
- `inference/benchmark_quant_compare.py` — per-quant benchmark (deprecated psutil measurement)
- `inference/benchmark_quant_proper.py` — proper MLX Metal memory benchmark
- `inference/live_demo.py` — web UI live demo
- All training/inference/tool scripts copied from ivi
- All checkpoints, training data, samples, voice assets moved from Downloads
- All session logs copied from ivi
- Quantized checkpoints created: `katie-v6-8bit/`, `katie-v6-4bit/`
