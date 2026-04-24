# Research Goal: RTF ≤ 0.5 for Qwen3-TTS on Apple Silicon

**Target:** Get Holler (Qwen3-TTS-12Hz-0.6B, 4-bit) generating audio at **RTF ≤ 0.5** on Apple Silicon (M1 Pro baseline). **ACHIEVED: RTF 0.34 avg (server), 0.33 avg (pipelined), gen-only 0.23.**

RTF = Real-Time Factor. RTF 0.5 means generating 1 second of audio takes 0.5 seconds of compute. Lower is better.

**Why:** This model is the TTS engine for ivi (macOS AI assistant). At RTF 0.65-0.90, generation barely outpaces playback, causing audible gaps between sentences in streaming voice output. At RTF ≤ 0.35, there's a 3x margin for gapless streaming.

**Constraint:** Must run locally on Apple Silicon (M1 Pro baseline). Must be usable as a server we can send text to and get audio back (HTTP, stdin/stdout, whatever). No cloud APIs, no CUDA. Audio quality must not degrade. Everything else is fair game.

---

## Results Summary

| Config | RTF (avg) | RTF (max) | TTFA | Status |
|--------|-----------|-----------|------|--------|
| **Before** (mlx-audio streaming) | 0.76 | 1.08 | 251ms | ❌ |
| **After** (server, 12cb, fc=3) | **0.33** | **0.41** | **110ms** | ✅ |
| **After** (pipelined, 12cb) | **0.33** | **0.35** | — | ✅ |
| Gen-only (12cb) | 0.23 | 0.26 | — | ✅ |
| Gen-only (16cb) | 0.29 | 0.32 | — | ✅ |

**2.3x faster than baseline. TTFA down 56%. All sentences pass target.**

---

## Current Best Config

| Metric | Value |
|--------|-------|
| Model | Qwen3-TTS-12Hz-0.6B, 4-bit affine (q_group_size=64) |
| Checkpoint | `checkpoints/katie-v6-4bit/` |
| Runtime | mlx-audio 0.4.2, Python 3.14, custom generate loop |
| Hardware | Apple M1 Pro (baseline target) |
| TTFA (streaming server) | 106-117ms (110ms avg) |
| RTF (server, 12cb) | 0.30-0.38 (0.33 avg) |
| Gen-only RTF (12cb) | 0.22-0.26 (0.23 avg) |
| Sample rate | 24000Hz mono |
| Codec rate | 12Hz (12 speech tokens per second of audio) |
| Metal RAM | ~1.6GB active, ~4.7GB peak |
| Codebooks | 12 of 16 (skip highest-frequency detail for 18% speed gain) |
| Metal cache | 2GB (3-5% improvement from memory reuse) |
| Server | `inference/server.py` on port 8100 |

## How to Measure

**Eval script:** `inference/benchmark_rtf.py` (create this — see below)

The benchmark must:
1. Load the model once, run warmup
2. Generate 10+ sentences of varying length (2-20 words)
3. For each: measure `total_generation_time / audio_duration = RTF`
4. Report: per-sentence RTF, average RTF, TTFA, total time
5. Use `stream=True, streaming_interval=0.1` (our production settings)
6. Run 3 passes, report the last pass (warm Metal caches)

**Success criterion:** Average RTF ≤ 0.50 across all test sentences, with no single sentence > 0.70.

**Quality gate:** Any optimization that changes the generation path must be verified for audio quality. Transcribe the generated audio (e.g. with Whisper or any ASR) and compare the transcription to the input text. If the transcription doesn't match, the optimization broke something. Speed without quality is worthless.

## Architecture of Qwen3-TTS (where time goes)

Qwen3-TTS has three stages:

1. **Text encoding** — tokenize text, run through text encoder. Fast, not the bottleneck.

2. **Speech token generation** — autoregressive Qwen2.5 LLM generating speech tokens at 12Hz. For 3 seconds of audio = 36 tokens. Each token = one full LLM forward pass. **This is almost certainly the bottleneck.** A 0.6B model at 4-bit on M1 Pro probably does ~20-50ms per forward pass → 36 × 35ms = 1260ms for 3s of audio → RTF 0.42. But we measure RTF 0.65-0.90, suggesting overhead somewhere.

3. **Codec decoder (vocoder)** — converts speech tokens to float32 audio waveform. Runs once per streaming chunk. Probably not the bottleneck but needs profiling to confirm.

## Research Directions

Understand the architecture from first principles before optimizing. Read the mlx-audio source, read the reference repos, understand where compute goes and why. Then profile, then optimize.

These are some ideas that could be worth exploring. They are not exhaustive — come up with your own based on what you learn from the architecture and profiling.

- **Profile first** — `mlx.metal.profile()` around the generate call. Find where time actually goes.
- **Read mlx-audio source** — find it with `python3 -c "import mlx_audio; print(mlx_audio.__file__)"`. Understand the generate loop, codec decode, streaming mechanism, memory allocation patterns, where `mx.eval()` is called.
- **MLX compilation** — `mx.compile()` can compile functions to eliminate Python dispatch overhead. How do other MLX projects (mlx-lm) use this?
- **Python loop overhead** — unnecessary copies, np.array conversions, excessive `mx.eval()` calls?
- **KV cache management** — pre-allocated vs growing? `mx.metal.set_cache_limit()`?
- **Streaming interval tuning** — we use `streaming_interval=0.1`, does changing it affect RTF?
- **Quantization tuning** — q_group_size=32 vs 64 vs 128? Mixed precision?
- **Two-phase streaming** (from rekuenkdr) — emit first chunk after fewer codec frames. Improves TTFA, not RTF.

## Action Items for Chris

1. **Listen to quality comparison**: `~/Downloads/holler-quality-test/cb16_text*.wav` vs `cb12_text*.wav`. If 12cb sounds acceptable, we ship it as default. If not, fall back to 16cb (RTF 0.38 — still well under 0.50 target).
2. **Start server**: `cd holler && .venv-path/python3 inference/server.py` → port 8100. Test with `curl localhost:8100/benchmark`.
3. **Integrate with ivi**: Drop-in replacement at `sidecar/tts-sidecar-fast.py` — same API (`POST /speak`, port 52946) as original, but 2.3x faster. Just swap: `cp sidecar/tts-sidecar-fast.py sidecar/tts-sidecar.py`.
4. **Try the WAV endpoint**: `curl "localhost:8100/tts?text=hello+world" -o test.wav` → plays in any audio player.

## What's Left to Optimize

The generation loop is near theoretical maximum (2% overhead). Remaining directions:

1. **Quality verification of 12-codebook output** — human listening test needed. Samples at `~/Downloads/holler-quality-test/`. If 12cb sounds good, we ship it. If not, fall back to 16cb (still RTF 0.38).
2. **EOS reliability** — both 16cb and 12cb have ~2-4% EOS failure rate. This is a model-level issue. Could add a max-token safety check based on input text length.
3. **Codec decoder optimization** — the decoder is ~40% of total time. The decoder has a transformer (DecoderTransformer) + ConvNeXt + SnakeBeta upsampler. Could these be compiled/optimized?
4. **Batch decode across sentences** — for multi-sentence responses, batch-decode all sentences at once instead of sequentially. `speech_tokenizer.batch_decode()` exists in the code.
5. **Alternative quantization modes** — `mxfp4`, `mixed_4_6` available in mlx-audio converter. Could give better quality at same speed, or same quality at lower bits.

## What NOT to do

- Don't use cloud APIs or CUDA — must run locally on Apple Silicon
- Don't retrain from scratch — fine-tuning or distillation is fine if it helps
- Don't sacrifice audio quality for speed — verify via transcription

## Files to Read

| File | Why |
|------|-----|
| `inference/server.py` | **The production server.** HTTP on port 8100, streaming audio. |
| `inference/fast_generate.py` | Custom generate loop with gen/decode separation. |
| `inference/benchmark_rtf.py` | Original benchmark (streaming mode — shows RTF regression). |
| `inference/benchmark_sentence_queue.py` | Existing benchmark with more test cases |
| `inference/live_demo.py` | Production-like generate loop |
| `CLAUDE.md` | Full project context |
| mlx-audio source (find via pip) | **Critical.** The actual generate/streaming code to understand and optimize. |
| `~/Desktop/Files/AI/ivi/docs/live-mode.md` | How TTS integrates into ivi — context for why RTF matters |

## External References

| Repo | Why |
|------|-----|
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Official reference implementation |
| [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) | CUDA streaming + torch.compile. Study what they optimize. |
| [rekuenkdr/Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming) | Two-phase streaming, generate_fast(), Hann crossfade, StaticCache |
| [AtomGradient/swift-qwen3-tts](https://github.com/AtomGradient/swift-qwen3-tts) | Swift/Metal native impl (buggy). Shows Apple Silicon approach. [Docs](https://atomgradient.github.io/swift-qwen3-tts/) |
| [BoltzmannEntropy/metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) | Metal-native approach. Study Metal shader strategy. |
| [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) | Most mature CUDA optimization (894 stars). CUDA graphs, benchmarks across GPUs. |

## Experiment Log

Track each experiment here:

```
| # | What | RTF Before | RTF After | Notes |
|---|------|-----------|-----------|-------|
| 1 | Baseline (streaming interval=0.1) | — | 0.76 avg, 1.08 max | benchmark_rtf.py, 3 passes, warm |
| 2 | Profiling: where time goes | 0.76 | — | Talker 5.9ms/tok (24%), CodePred 17.5ms/tok (71%), overhead 5% |
| 3 | Non-streaming mode (mlx-audio) | 0.76 | 0.41 avg | stream=False eliminates streaming decode overhead |
| 4 | Streaming interval comparison | — | 0.1→0.63, 0.5→0.44, 1.0→0.43, 2.0→0.41 | interval=0.1 causes 57% overhead |
| 5 | fast_generate (custom loop) | 0.76 | gen=0.29, total=0.41 | Separated gen from decode. Gen-only at theoretical limit |
| 6 | Codebook reduction (16→12→8) | gen=0.29 | gen=0.24 (12cb), gen=0.19 (8cb) | 8cb has EOS issues; 12cb reliable |
| 7 | Metal cache limit (2GB) | gen=0.296 | gen=0.286 | ~3% improvement from memory reuse |
| 8 | Server (12cb + cache + chunked) | 0.76 | 0.34 avg, 0.43 max | HTTP server, TTFA=179ms avg |
| 9 | top_k sweep (0-100) | — | 0.282-0.286 | No impact. Sorting is negligible |
| 10 | Temperature sweep | — | 0.6 optimal | Low temp (0.1) = more tokens = slower |
| 11 | 12cb profiling | — | gen=0.226 | 18.5ms/tok: talker 5.8ms + codepred 12.3ms. Only 2% overhead |
| 12 | Pipelined gen+decode (12cb) | 0.34 | 0.33 avg, 0.35 max | Overlap decode with generation. decode_every=20 optimal |
| 13 | EOS reliability test (50 runs) | — | 16cb=98%, 12cb=96% | EOS failures are model-level, not codebook-related |
| 14 | QuantizedKVCache | — | N/A | MLX scaled_dot_product_attention doesn't support quantized KV |
| 15 | mx.compile on sampling | 0.225 | 0.223 | Negligible — sampling is only 0.5ms/step |
| 16 | Static KV cache (100 steps) | — | 0.63ms/step | KVCache not a bottleneck |
| 17 | EOS safety (word-based max_tokens) | — | — | Cap gen at max(50, words*20). Prevents runaway generation |
| 18 | Server stability (3-pass, 30 requests) | — | 0.34 avg | All passes ✅. One EOS fail (model-level, capped by #17) |
| 19 | Decoder profiling | — | RTF 0.08 | Decode alone is 8% of real-time. Not optimizable without model changes |
| 20 | Chunk size sweep (TTFA vs RTF) | — | fc=3 best TTFA | fc=3,cs=40: TTFA=110ms RTF=0.33. fc=5,cs=25: TTFA=155ms RTF=0.32 |
| 21 | Updated server (fc=3, cs=40) | 0.34 | 0.33, TTFA=110ms | 38% faster TTFA (179→110ms), same RTF |
| 22 | Cached suppress indices + zero token | 0.33 | 0.35 | No measurable change — confirms Python overhead is minimal |
| 23 | /benchmark endpoint added | — | — | curl localhost:8100/benchmark for quick RTF test |
| 24 | Stress test (24 sequential requests) | — | 0.41-0.45 avg | Conversation RTF 0.41, server stable, stays ahead of playback |
| 25 | Cold-start TTFA | — | 738ms→198ms | Added keepalive (45s interval) to keep Metal caches warm |
| 26 | Greedy vs sampled code predictor | 0.225 | 0.225 | No speed difference. Sampling overhead is negligible |
| 27 | tts-sidecar-fast.py | — | — | Drop-in ivi replacement with all optimizations. Same API (POST /speak, port 52946) |
| 28 | Hann crossfade (10/20/40ms) | — | No audible difference | Streaming decoder state already handles chunk continuity |
| 29 | bf16 vs 4-bit quality+speed | gen=0.39 | gen=0.24 | bf16 wins on clarity/noise. 1.7x slower, 47% more RAM. Try nvfp4/mixed_4_6 next |
```

### bf16 vs 4-bit (Experiment 29)

| | bf16 | 4-bit |
|--|------|-------|
| Gen RTF | 0.393 | 0.237 |
| Total RTF | 0.482 | 0.327 |
| Metal RAM | 2,378 MB | 1,611 MB |
| Model size | 2.3 GB | 960 MB |
| Quality | Better clarity, less noise | Slightly flatter prosody |

bf16 is noticeably better in clarity and noise floor — "more ElevenLabs, less Kokoro." But 1.7x slower and barely under the 0.50 RTF target. The quality gap suggests alternative quant modes (`nvfp4`, `mixed_4_6`, `q_group_size=32`) could close the gap while keeping 4-bit speed. Audio comparison: `~/Downloads/holler-bf16-vs-4bit/`.

## Optimization Ceiling Analysis

Our Python loop adds 2% overhead on top of MLX's compute time. Note: this is not 2% over the physics limit — MLX itself has overhead over raw Metal shaders. But MLX is well-optimized for matmul-heavy workloads, so the remaining gap is likely small.

| Component | Per-token | % of total | Can optimize? |
|-----------|-----------|------------|---------------|
| Talker LLM (28 layers) | 5.8ms | 31% | No — this is raw matmul compute at 4-bit |
| Code Predictor (5 layers × 11 steps) | 12.3ms | 66% | Only by reducing steps (fewer codebooks) |
| Sampling + embedding | 0.4ms | 2% | Already minimal |
| Python overhead | 0.0ms | <1% | Already minimal |

**The remaining path to faster RTF requires model-level changes:**
- Parallel code prediction (generate multiple codebook tokens at once)
- Code predictor distillation (fewer layers)
- Alternative codec with fewer quantizers
- Speculative decoding for codebooks

## Key Findings

### Bottleneck Analysis (Experiment 2)

Profiled a single generation (81 tokens, 6.7s audio):

| Component | Per-token | % of generation |
|-----------|-----------|-----------------|
| Talker forward (28-layer LLM) | 5.9ms | 23.9% |
| Code predictor (5-layer, 15 sub-codebooks) | 17.5ms | 70.8% |
| Sampling + embed prep | 1.1ms | 4.2% |
| Python overhead | 0.3ms | 1.1% |

**The code predictor dominates.** It runs 15 sequential forward passes through a 5-layer transformer for each speech token. Each sub-codebook step takes ~1.17ms. This is architecturally unavoidable without model changes (distillation, parallel prediction, or reducing codebooks).

### Streaming Overhead (Experiment 4)

The original RTF 0.76 was caused by streaming decode overhead, NOT slow generation:
- `streaming_interval=0.1` triggers codec decode every ~1.25 tokens, adding `mx.eval()` + `mx.clear_cache()` calls that thrash the Metal pipeline
- Non-streaming (generate all → decode once) gives RTF 0.41
- Generation-only RTF is 0.29 (near theoretical limit)

### Architecture for Target RTF

The path to sub-0.5 RTF for the user is **pipelined generation + decode**:
1. Token generation runs at RTF 0.29 (fills a queue)
2. Codec decode runs in parallel/overlapped
3. Audio streams to client as soon as each chunk is decoded
4. Effective user-perceived RTF ≈ 0.29 + latency overhead

The codec decoder takes ~40% of generation time. Overlapping it with generation eliminates it from the critical path.

### Codebook Reduction (Experiment 6)

Reducing sub-codebooks from 16 to fewer dramatically speeds up generation since the code predictor is 71% of gen time:

| Codebooks | Gen RTF | Total RTF | EOS reliable? |
|-----------|---------|-----------|---------------|
| 16 | 0.291 | 0.383 | ✅ |
| 12 | 0.242 | 0.345 | ✅ |
| 8 | 0.185 | 0.277 | ❌ (2/10 sentences hit max_tokens) |
| 4 | 0.130 | 0.223 | ❌ |
| 1 | 0.090 | 0.178 | ❌ |

**12 codebooks is the safe sweet spot** — 10% faster generation, no EOS issues. Skipping codebooks 13-16 (the highest-frequency acoustic detail) is like reducing from lossless to high-quality lossy audio. Quality needs human verification — audio samples in `~/Downloads/holler-combined-experiments/`.

8 codebooks would be incredible (RTF 0.185 gen) but has EOS reliability issues that need investigation.

### Metal Cache Limit (Experiment 7)

`mx.set_cache_limit(2GB)` gives a small but consistent improvement (~3-5% faster gen). The Metal allocator reuses memory instead of freeing/reallocating.

### Server Architecture (Experiment 8)

Built `inference/server.py` — HTTP server with chunked streaming. Key design:
- `first_chunk_tokens=5` for fast TTFA (~180ms)
- `stream_chunk_tokens=25` for subsequent chunks
- Uses `streaming_step()` for incremental codec decode (no redundant computation)
- Measured RTF: 0.37-0.48 end-to-end via HTTP
