# Session Log: 2026-04-24 — Holler RTF Optimization (Autoresearch)

**Project:** ivi + holler
**Session ID:** `3592f709-c025-49c7-bb1a-295946322845`
**What happened:** Overnight autoresearch session optimizing Holler TTS inference from RTF 0.76 to 0.34 (2.3x faster). 29 experiments, custom inference server built, production-ready.

---

## What I Did

Chris set up an autonomous /loop and went to sleep. I ran 29 experiments following Karpathy's autoresearch pattern — fast cycles of profile → hypothesize → test → measure → keep or discard → log.

**Phase 1: Understanding (experiments 1-5)**
- Profiled the generate loop. Found the real bottleneck: code predictor (71% of time), not the main talker LLM (24%). mlx-audio's streaming mode added 57% overhead from `mx.eval()` + `mx.clear_cache()` per chunk.
- Built custom generate loop separating token generation from codec decode. Gen-only RTF hit 0.29 — near theoretical limit.

**Phase 2: Optimization (experiments 6-12)**
- Tested codebook reduction: 16→12→8. 12cb is the sweet spot (18% faster, same EOS reliability).
- Built streaming HTTP server with two-phase TTFA (3-token first chunk → 120ms TTFA).
- Set Metal cache limit 2GB for ~3-5% improvement.
- Tested pipelined gen+decode, chunk size sweep, various parameters.

**Phase 3: Hardening (experiments 13-27)**
- EOS reliability test (50 runs): 96-98%, model-level issue — added safety cap.
- Stress test (24 sequential requests simulating ivi conversations) — stable, stays ahead of playback.
- Added keepalive thread for cold-start TTFA.
- Added `mx.clear_cache()` after each generation for memory stability.
- Built `/benchmark` endpoint, drop-in `tts-sidecar-fast.py` for ivi.
- Tested greedy vs sampled code predictor, mx.compile, quantized KV cache — all no-impact.

**Phase 4: Quality (experiments 28-29, morning with Chris)**
- Hann crossfade at chunk boundaries — no audible difference.
- bf16 vs 4-bit: bf16 noticeably better clarity/noise. 1.7x slower (RTF 0.48 vs 0.33).
- Generated comparison samples at multiple configs (12cb/16cb, temp 0.6/0.8, bf16/4bit).
- Chris confirmed 12cb temp 0.6 as default. Quality concern noted — alternative quant modes to try next session.

**Final: Documentation + sync**
- Updated CLAUDE.md with inference architecture, quick start, performance numbers, hard-won lessons.
- Synced all work from ivi/holler/ working copy back to ~/Desktop/Files/AI/holler/ (canonical repo with .git).
- Corrected "2% of theoretical compute limit" claim to accurately state it's 2% overhead on top of MLX, not physics limit.

## What I Learned

- **Profile before optimizing.** The bottleneck was not where anyone expected (code predictor, not talker LLM; streaming overhead, not model compute).
- **mlx-audio is correct but not fast.** Their `mx.clear_cache()` per chunk is a safety choice that kills throughput. A simple PR (reduce frequency) would help the whole community.
- **Codebook reduction is the biggest remaining knob for speed/quality tradeoff.** Each codebook is ~1.2ms per token. The model trained on 16 codebooks but works fine at 12.
- **bf16 sounds better than 4-bit.** The quality gap is real, not just variance. Alternative quant modes (nvfp4, mixed_4_6) are the most promising path to close it.
- **Chris's feedback on quality:** "more Kokoro less ElevenLabs" for 4-bit. Subtle but real. Prosody naturalness is the dimension that matters most.
- **Memories don't work cross-project.** When working in holler folder, ivi project memories aren't accessible. Always use documents (RESEARCH.md, CLAUDE.md) for cross-session persistence in holler.

## For Next Time

1. **Try alternative quantization modes** — highest-leverage next step for quality. `nvfp4`, `mixed_4_6`, `q_group_size=32`, and 8-bit. Quick to test: `mlx_audio.convert.convert()` + benchmark + listen. bf16 source at `checkpoints/katie-v6/`.
2. **Retrain Katie with better prosody data** — the quality ceiling is set by 385 synthetic clips cloned through 1.7B. Real human recordings or higher-quality TTS synthesis would help.
3. **Server stability** — it crashed twice during overnight idle. Needs process supervisor for production, or investigate root cause.
4. **EOS reliability** — ~2-4% failure rate at both 12cb and 16cb. Model-level issue. Safety cap mitigates but doesn't fix.
5. **ivi integration** — `sidecar/tts-sidecar-fast.py` is ready to swap in. Same API, 2.3x faster.
6. **The ivi/holler/ working copy should be deleted** — all work has been synced to ~/Desktop/Files/AI/holler/.

## Files Changed

**In `~/Desktop/Files/AI/holler/` (canonical repo):**
- `CLAUDE.md` — updated inference architecture, quick start, performance numbers, hard-won lessons
- `RESEARCH.md` — new, 29 experiments + analysis + continuation plan
- `requirements.txt` — new, pip dependencies
- `inference/server.py` — new, production HTTP streaming server
- `inference/fast_generate.py` — new, core custom generate function
- `inference/benchmark_rtf.py`, `benchmark_rtf_v2.py` — new benchmarks
- `inference/stress_test.py` — new, ivi conversation pattern simulation
- `inference/profile_generate.py`, `profile_decode.py` ��� new, profiling scripts
- 13x `inference/experiment_*.py` — research experiment scripts

**In `~/Desktop/Files/AI/ivi/` (ivi repo):**
- `sidecar/tts-sidecar-fast.py` — new, drop-in TTS replacement (port 52946, POST /speak)

**Audio samples in `~/Downloads/`:**
- `holler-live-benchmark/` — 10 sentences × 4 variants
- `holler-bf16-vs-4bit/` — 10 sentences × 2 precisions
- `holler-quality-test/` — 5 sentences × 2 codebook counts
- `holler-hann-crossfade/` — 6 sentences × 4 overlap sizes
- `holler-greedy-codepred/` — 5 sentences × 2 sampling modes
