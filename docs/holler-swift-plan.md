# Holler Swift: Fast On-Device TTS Inference Library

**Status:** Phase 2A complete (streaming + session API). Phase 2B next (KV cache carryover).  
**Date:** 2026-04-30  
**Goal:** Pure Swift TTS inference library for Holler voices. Powers standalone Holler Mac app, and can replace ivi's Python TTS sidecar for in-process inference.

## Architecture

```
holler-swift/                    (open source Swift package, Apache 2.0)
├── Package.swift                (depends on mlx-audio-swift)
├── Sources/HollerKit/
│   ├── HollerModel.swift        (public API: load, synthesize, stream, unload)
│   ├── StreamingOptimizer.swift  (two-phase streaming, codebook skip)
│   └── ...
└── Tests/

Depends on:
  └── Blaizzy/mlx-audio-swift    (Qwen3-TTS inference, same author as Python mlx-audio)
      └── ml-explore/mlx-swift   (Apple's MLX framework, Metal GPU backend)
```

### Products using holler-swift:

1. **Holler.app** — Paid Mac app (~$20 one-time). Text box, voice picker, waveform, export. Bundles 6-bit checkpoint (~1.1GB).
2. **ivi.app** — Replaces Python TTS sidecar entirely. `HollerModel.stream()` called directly from `TTSService.swift`. No HTTP, no process management, no SIGKILL.

### Business model (direction, not locked in):

- Holler voices + training pipeline + inference libraries = **open source** (Apache 2.0)
- Holler Mac app = **paid one-time purchase**
- Open source drives adoption → app monetizes non-technical audience

## Licensing Direction

No LICENSE file exists yet. The intent:

- **Library code** (holler-swift, training scripts, inference server) — **permissive** (Apache 2.0). Attracts developers, builds community. The code isn't the moat.
- **Model weights** (the trained voices) — **restricted**. Free to use in your own projects, but you cannot use Holler voices to build a competing TTS application or service. This is the value we're protecting. Similar to Llama's community license or a Commons Clause addendum.

The split: code is open, voices have a usage restriction. If someone trains their own voices with our open training pipeline, they can do whatever they want with those. But they can't take our 30 voices and ship "Holler Clone" on the App Store.

Exact license text TBD — needs proper drafting.

## Python + Swift: Both Ship

The Holler repo will include both inference backends:

- **Python** (`inference/server.py`) — already proven, RTF 0.38, battle-tested in ivi. Uses mlx-audio. Works today.
- **Swift** (holler-swift, via mlx-audio-swift) — in-process integration for Mac apps, no sidecar overhead. Needs optimization work.

Both are valid for different use cases. Python is great for quick prototyping, server deployments, and anyone already in a Python workflow. Swift is the path to native Mac apps (Holler.app) and in-process integration (ivi without a sidecar). They share the same model checkpoints and produce the same output.

For ivi specifically, Swift integration would eliminate the Python sidecar (process management, SIGKILL, GIL deadlocks, health polling), but the Python path remains as a fallback and for the open source release.

## Why mlx-audio-swift (not speech-swift)

**Tested 2026-04-29. Critical finding.**

| | mlx-audio-swift (Blaizzy) | speech-swift (soniqo) |
|---|---|---|
| Voice identity | **Correct** — Kit sounds like Kit | **Broken** — generic female voice |
| Author | Same person as Python mlx-audio | Different person, reimplemented |
| Custom voice support | Proper speaker token + no instruct injection | Wrong instruct injection, wrong params |
| Quantization | Handles 6-bit from config.json | Needed manual fix (only had 4/8-bit presets) |
| Streaming | Continuous codec decode (`streamingStep`) | Chunked re-decode (causes pops) |
| Popping/artifacts | None in batch output | Present in all outputs |

**Root cause of speech-swift failures:** The embedding construction (`buildPrefillEmbeddings`) and generation loop were reimplemented from scratch by a different developer. Subtle differences in how the prefill sequence is constructed mean the model sees an input distribution it was never trained on, producing wrong voice output.

**mlx-audio-swift works because it's a faithful port by the same author (Prince Canuma) who wrote the Python mlx-audio that our model was trained and tested with.**

## Relationship with mlx-audio-swift

**Not yet determined.** We've confirmed mlx-audio-swift works with our model. The next step is to actually build with it and see how it feels to make changes — then we'll know the right approach.

Options on the table:
- **Depend on it as-is + PR optimizations upstream** — cleanest, benefits community, but we're on Prince's release schedule
- **Fork it** — full control, but we own maintenance of a large codebase
- **Thin wrapper that calls into it** — middle ground, our API on top of his library

We'll figure this out by doing. First: make it work end-to-end in a Mac app. The optimization PRs and structural decisions come after we have something running.

## Current Performance

### Baseline (M1 Pro, Holler Kit 6-bit)

| Metric | Python server (our optimized) | mlx-audio-swift (stock) | **HollerKit (Phase 2A)** |
|---|---|---|---|
| TTFA raw streaming | 128-144ms | 350ms | **125-140ms** |
| TTFA with pipeline | N/A | N/A | **230-350ms** |
| TTFA session (LLM) | N/A | N/A | **265ms** (20ms/tok sim) |
| RTF (12cb, streaming) | 0.45-0.50 | 0.52 | **0.47-0.66** |
| GPU memory | ~1.7GB | ~1.8GB | ~1.78GB |
| Model load | ~1.5s | ~5s | ~1.5s (incl. warmup) |

**TTFA breakdown (with pipeline):** ~130ms model generation + 0-200ms onset detection (discarding codec warmup silence). The onset detector uses 2-of-3 RMS confirmation which adds 1-2 chunks of latency — optimization target for later.

**TTFA breakdown (session):** token feeding delay (LLM-dependent) + generation + onset detection. At 200 tok/s: ~240ms. At 50 tok/s: ~270ms. Feeding is the smallest component.

**RTF note:** Short sentences ("Hey!", "Got it.") have RTF 0.65+ because codec warmup is proportionally larger. Sentences >1s audio hit RTF ~0.47-0.53, matching Python.

### Optimizations Applied (2026-04-29)

1. **Codebook skip** — ✅ DONE. 12 of 16 codebooks by default (configurable per-request). 25% less code predictor work. Zero quality loss.
2. **Memory.clearCache() post-gen** — ✅ DONE. Moved from every 50 steps to after generation. Eliminates GPU stalls.
3. **3-token streaming chunks** — ✅ DONE. Consistent small chunks throughout (not two-phase). Steady audio flow for streaming.
4. **Warmup on model load** — ✅ DONE. Dummy forward pass through talker + code predictor + codec decoder. Front-loads Metal shader JIT (~300ms one-time cost at load).
5. **Codebooks as per-request parameter** — ✅ DONE. `codebooks: 12` (fast) or `codebooks: 16` (full, default). No restart needed.
6. **compile() the Talker step** — ✅ INVESTIGATED, ❌ NOT BENEFICIAL. See section below.

## Optimization Investigation Log

### 1. compile() the Talker step — INVESTIGATED, NOT BENEFICIAL

**Status (2026-04-30): RESOLVED.** compile() works but doesn't help for this model size.

**What we tried (2026-04-29):** Four approaches, all failed:
1. `shapeless: true` with MRoPE → Slice/Split can't infer shapes
2. MRoPE outside compile, pass cos/sin → rotateHalf still uses shape-dependent slicing
3. Compile-safe rotateHalf using roll() → roll uses Split internally
4. `shapeless: false` → recompiles every step (KV cache grows), 30% slower

**What we found (2026-04-30):** speech-swift solved this differently — replace MRoPE with `MLXNN.RoPE` (backed by `MLXFast.rope`, a single fused Metal kernel). For TTS, all 3 MRoPE axes are always identical (T=H=W), so standard 1D RoPE produces mathematically identical output. `MLXFast.rope` has no shape-dependent operations, so `compile(shapeless: true)` works.

**Implementation:** Added `MLXNN.RoPE` to `TalkerAttention`, rewrote `stepWithRawCache` to use `rope(q, offset: MLXArray)` instead of manual cos/sin. `setupCompilation()` now uses `shapeless: true` with offset as `MLXArray` (not baked Int). Compiles and runs correctly — voice output verified.

**Benchmark (M1 Pro, 0.6B 6-bit):**
| | Compiled | Uncompiled |
|---|---|---|
| Per-step avg | 29.5ms | 26.1ms |
| Talker forward (graph build) | 4.8ms | 0.9ms |
| Code predictor | 2.3ms | 2.5ms |
| eval() GPU sync barrier | ~22ms | ~22ms |

**Why it doesn't help:** The talker forward pass is only 0.9ms uncompiled — it just builds a lazy computation graph. The real work happens at the `eval()` barrier (~22ms), where Metal executes the whole graph. compile() can't reduce GPU execution time. The compiled path is slower because it uses concatenation-based KV cache (copies grow each step), while the uncompiled path uses KVCacheSimple (pre-allocated, slice-assign, zero-copy).

**Note:** `mlx-lm` (Python reference) also doesn't use compile() for LLM generation. The MLX team doesn't consider it essential for decode performance — KVCacheSimple's pre-allocation provides the actual benefit.

**Infrastructure preserved:** `setupCompilation()`, `executeTalkerStep()`, and `stepWithRawCache` methods all work. Uncomment `setupCompilation()` in `warmUp()` to enable. May help on larger models or newer hardware where dispatch overhead is proportionally larger relative to compute.

### 2. Lazy eval chain for Code Predictor

**Note (2026-04-29):** The original plan assumed each codebook triggers a separate `eval()`. Examining the actual code, the code predictor tokens feed into each other sequentially via embedding lookups — each codebook depends on the previous one's sampled token. The eval barrier is inherent to the sequential dependency, not an unnecessary sync. The real win is the compiled CP transformer step (same infrastructure as Talker compile).

### 3. Skip codebooks 13-16 — ✅ DONE

Codebook skip implemented and exposed as `codebooks` parameter (default: 16). 10-run benchmark:
- 12cb: 128ms TTFB, 2.33x RTFx, 29.3 tok/s
- 16cb: 143ms TTFB, 2.04x RTFx, 25.6 tok/s
- 14% speed difference, 16cb has slightly fuller audio quality

### 4. Move Memory.clearCache() to post-generation — ✅ DONE

### 5. Streaming chunks — ✅ DONE (3-token throughout)

Changed to fixed 3-token chunks throughout (not two-phase). Two-phase (3 then 40) doesn't work for streaming — after the 240ms first chunk plays, there's a ~1.4s gap waiting for 40 tokens. Consistent small chunks = steady audio flow.

### 6. Warmup on model load — ✅ DONE

Dummy forward pass through talker (prefill + single generation step), code predictor, and codec decoder streaming path. Adds ~300ms to model load, eliminates first-run TTFB penalty (451ms → 144ms).

### 7. (Optional) Test 4-bit quantization

6-bit is not a power of 2, so MLX falls back to general `qmv` kernel instead of optimized `qmv_quad`. 4-bit gets the fast path. Test quality trade-off — may be negligible for 0.6B model.

## Key Files in mlx-audio-swift

The generation loop to optimize:
```
Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift
  - Lines 306-580: generateVoiceDesign() — main generation function
  - Lines 361-375: CustomVoice input preparation (speaker token, no instruct)
  - Lines 411-508: Autoregressive generation loop (optimize here)
  - Lines 439-459: Code predictor inner loop (lazy eval target)
  - Lines 491-504: Streaming chunk decode
  - Lines 506-508: Memory.clearCache() every 50 steps (move to post-gen)
  - Lines 1095-1214: fromPretrained / fromModelDirectory (model loading)

Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSTalker.swift
  - Talker transformer (compile target)

Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSCodePredictor.swift
  - Code predictor (compile + lazy eval target)

Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechTokenizer.swift
  - Codec decoder with streamingStep() for continuous decode
```

## Public API Design

Three tiers: batch (complete audio), streaming (known text), and session (LLM integration).

```swift
import HollerKit

// Load model (~2s: weights + Metal warmup)
let model = try await HollerModel.load(repo: "sentium/holler-0.6b-6bit")

model.voices        // ["kit", "dakota"]
model.isLoaded      // true


// === Tier 1: Batch — complete text, complete audio ===

let audio = try await model.synthesize("Hello world", voice: "kit")
// audio.samples: [Float], audio.sampleRate: 24000, audio.duration: Double


// === Tier 2: Streaming — complete text, chunked audio (~130ms TTFA) ===

for try await chunk in model.stream("Hello world", voice: "kit") {
    // chunk.samples: [Float], chunk.sampleRate: 24000
    playerNode.scheduleBuffer(chunk.asAudioBuffer())
}


// === Tier 3: Session — streaming text in, streaming audio out ===
// For LLM integration. Text arrives token-by-token, audio starts
// as soon as the first sentence is complete. KV cache carries over
// between sentences automatically for prosody continuity.

let session = model.makeSession(voice: "kit")

// Feed text and consume audio concurrently:
async let playback: Void = {
    for try await chunk in session.audio {
        playerNode.scheduleBuffer(chunk.asAudioBuffer())
    }
}()

// Feed tokens as they arrive from an LLM
for await token in llmTokenStream {
    await session.feed(token)  // buffers internally, synthesizes on sentence boundaries
}
await session.finish()  // flush remaining buffered text
try await playback      // audio stream ends after all sentences synthesized

// Cancellation (barge-in: user interrupts while ivi is speaking)
session.cancel()  // stops generation, discards buffer, releases cache


// === Configuration ===

model.configuration.temperature = 0.6
model.configuration.codebooks = 12
model.configuration.maxRetries = 3

// === Memory management ===

await model.unload()  // releases ~1.7GB GPU memory
```

### API Design Principles

- **Library yields `[Float]` + sampleRate.** No AVFoundation dependency. Caller handles playback, file writing, or further processing. This keeps HollerKit focused on generation.
- **Session handles carryover implicitly.** First sentence in a session generates with a fresh KV cache. Subsequent sentences carry over the cache for prosody continuity. The caller never sets a "continue" flag — the session just knows.
- **Sentence buffering is internal.** LLM output arrives as arbitrary text chunks (half a word, 1.5 sentences, etc.). The session's `SentenceBuffer` accumulates text, detects sentence boundaries (`.!?` + space, abbreviation-aware), and dispatches complete sentences to the TTS model. `finish()` flushes whatever remains.
- **Streaming is the primary generation path.** Both `stream()` and `synthesize()` use the same streaming engine internally. `synthesize()` is sugar that collects all chunks into one result. The silence pipeline works on chunks as they arrive (not on complete audio).
- **Force-yield on long sentences.** If no sentence boundary is detected after ~15 words, the buffer yields anyway to prevent long first-sentence delays.

## Model Cache Setup (for testing)

mlx-audio-swift expects models in HuggingFace cache format:

```bash
# Hard-link checkpoint into HF cache (zero disk space, instant)
CACHE=~/.cache/huggingface/hub/mlx-audio/sentium_holler-kit-dakota-6bit
mkdir -p "$CACHE/speech_tokenizer"
for f in holler/checkpoints/holler-kit-dakota-6bit/*; do [ -f "$f" ] && ln -f "$f" "$CACHE/"; done
for f in holler/checkpoints/holler-kit-dakota-6bit/speech_tokenizer/*; do [ -f "$f" ] && ln -f "$f" "$CACHE/speech_tokenizer/"; done
```

CLI test:
```bash
cd mlx-audio-swift
swift build -c release --product mlx-audio-swift-tts --disable-sandbox
# Need metallib — build or copy from speech-swift
.build/arm64-apple-macosx/release/mlx-audio-swift-tts \
  --text "Hello world" \
  --model sentium/holler-kit-dakota-6bit \
  --voice kit \
  --temperature 0.6 \
  --output test.wav
```

## Build Notes

- mlx-audio-swift and speech-swift both pin mlx-swift 0.31.3 (same commit). Metallib is interchangeable.
- speech-swift needs `SpeechCore` binary xcframework (closed source blob) for full build, but `Qwen3TTS` target builds without it: `swift build -c release --target Qwen3TTS --disable-sandbox`
- mlx-audio-swift builds cleanly: `swift build -c release --product mlx-audio-swift-tts --disable-sandbox` (~4 min first build)
- Metallib build: `./scripts/build_mlx_metallib.sh release` (speech-swift has this script, mlx-audio-swift expects Xcode to produce it)

## HollerKit Package Implementation

### Package Location & Dependency

Lives at `holler/swift/HollerKit/`. Depends on sentiuminc/mlx-audio-swift fork via exact SPM tag. Upstream PRs deprioritized — fork works well, all changes go there.

```swift
// Package.swift (key parts)
dependencies: [
    .package(url: "https://github.com/sentiuminc/mlx-audio-swift.git", exact: "0.31.3-holler.1"),
]
```

### Package Structure

```
holler/swift/HollerKit/
├── Package.swift
├── Sources/
│   ├── HollerKit/
│   │   ├── HollerModel.swift          — Public API: load, stream, synthesize, makeSession, unload
│   │   ├── SpeechSession.swift        — Session actor: feed/finish/cancel/audio (LLM integration)
│   │   ├── SentenceBuffer.swift       — Streaming text → sentence boundary detection
│   │   ├── HollerConfiguration.swift  — All tunables with Python-parity defaults
│   │   ├── HollerAudioChunk.swift     — Chunk/Audio types yielded by stream
│   │   ├── HollerError.swift          — Error types
│   │   ├── SilenceAnalyzer.swift      — RMS speech detection, onset finding
│   │   ├── AudioPostProcessor.swift   — Onset trim, 20ms fadeout, silence generation
│   │   ├── RetryController.swift      — Escalating temperature retry (up to 3x)
│   │   ├── InferenceActor.swift       — Actor serializing all MLX calls (batch + streaming)
│   │   └── GenerationSession.swift    — Chunk-by-chunk streaming silence pipeline
│   └── HollerCLI/
│       └── HollerCLIApp.swift         — `holler` CLI binary
└── Tests/
    └── HollerKitTests/
        ├── SilenceAnalyzerTests.swift
        ├── AudioPostProcessorTests.swift
        └── SentenceBufferTests.swift
```

### What Each File Ports from Python (server.py)

| Python (server.py) | Swift (HollerKit) | Notes |
|---|---|---|
| `_has_speech()` L173-183 | `SilenceAnalyzer.hasSpeech()` | 2-of-3 RMS window confirmation |
| `_find_speech_onset()` L186-204 | `SilenceAnalyzer.findSpeechOnset()` | 10ms windows, 150ms pre-roll |
| `_apply_trailing_fadeout()` L207-214 | `AudioPostProcessor.applyFadeOut()` | 20ms linear fade-out |
| `_run_generation()` L217-419 | `GenerationSession` | Chunk-by-chunk streaming silence pipeline |
| `generate_audio()` L424-484 | `RetryController` + `HollerModel.stream()` | 3x retry, escalating temp |
| `_mlx_worker()` L76-146 | `InferenceActor` | Swift actor (not thread+queue) |
| `_carry_over_state` L422 | `SpeechSession` (internal KV cache) | Implicit carryover via session lifecycle |
| Carryover pause L369-374 | `SpeechSession` | 150-250ms pause injection between sentences |
| N/A (HTTP chunked) | `SentenceBuffer` | Streaming text in from LLM, sentence boundary detection |

### 4-Piece Silence Pipeline (GenerationSession)

Port of `_run_generation()` from server.py. Wraps `Qwen3TTSModel.generateStream()` and processes raw audio chunks:

1. **Piece 1:** Skip silent chunks before speech (codec warmup removal)
2. **Piece 2:** Sample-level onset trim in first speech chunk — 10ms RMS windows, 2-of-3 confirmation, 150ms pre-roll
3. **Piece 3:** 20ms linear fade-out on final chunk
4. **Piece 4:** Post-speech silence buffer + abort after `SILENT_ABORT_TOKENS` (16 tokens)

Plus: short audio detection (`minAudioSecondsPerWord: 0.08`) and retry with escalating temperature (+0.1 per attempt, max 3 retries).

### Thread Safety: Swift Actor vs Python Worker Queue

Python uses `_mlx_worker()` — a single persistent thread reading from `inference_queue`. All MLX calls happen on that one thread to avoid Metal thread-safety issues (mlx#2086).

Swift uses an `actor` (`InferenceActor`). The actor guarantees serialized access — only one caller executes at a time. Same safety, idiomatic Swift, no manual queue management.

### Implementation Phases

**Phase 1 — Batch generation + silence pipeline (DONE):**

Uses batch `generate()` API. Silence pipeline processes complete audio post-generation.
TTFA = total generation time (~500ms-2s). Proved the full pipeline works end-to-end.

1. ✅ Package.swift + types (Configuration, AudioChunk, Error)
2. ✅ SilenceAnalyzer + AudioPostProcessor — pure `[Float]` functions, unit testable
3. ✅ RetryController — decision logic, unit testable
4. ✅ InferenceActor + HollerModel (load/unload/voices)
5. ✅ GenerationSession — batch silence pipeline
6. ✅ `HollerModel.stream()` + `synthesize()`
7. ✅ `holler` CLI binary

**Phase 2A — Streaming generation (no fork changes needed):**

Switch from batch `generate()` to streaming `generateStream()`. Proved safe with
6 consecutive calls, zero crashes, ~130ms TTFA (2026-04-30). The Phase 1 segfault
was `String(format: "%s")`, not MLX threading — streaming was never broken.

8. SentenceBuffer — text accumulation, sentence boundary detection, flush
9. GenerationSession rewrite — chunk-by-chunk streaming silence pipeline
10. InferenceActor streaming path — wraps `generateStream()`, yields `[Float]` chunks
11. HollerModel.stream() — yields real streaming chunks (~130ms TTFA)
12. SpeechSession shell — `feed()/finish()/cancel()/audio` API, uses SentenceBuffer

**Phase 2B — KV cache carryover (needs fork changes):**

All changes to sentiuminc/mlx-audio-swift fork, tagged releases. No upstream PRs.

Fork changes to `generateVoiceDesign()`:
- Add `existingCache: [any KVCache]?` param — reuse instead of `talker.makeCache()`
- Return cache after generation for caller to persist
- Extended causal mask when `cache[0].offset > 0` (server.py L271-279)
- Conditional `resetStreamingState()` — skip when carrying over
- Conditional `Memory.clearCache()` — skip when carrying over

```swift
public struct TalkerCacheState: @unchecked Sendable {
    public let cache: [any KVCache]
}
```

HollerKit changes:
13. Wire carryover into SpeechSession — hold TalkerCacheState between sentences
14. Pause injection — 150-250ms silence between carried-over sentences
15. Test carryover audio quality — A/B vs Python server

### `holler` CLI Design

```
holler --text "Hello world" --voice kit [options]
holler --benchmark [options]

Options:
  --text, -t <string>         Text to synthesize (required unless --benchmark)
  --voice, -v <name>          Voice name (default: first available)
  --model, -m <path-or-repo>  Model path or HF repo (default: sentium/holler-0.6b-6bit)
  --output, -o <path>         Output WAV path (default: output.wav)
  --codebooks <int>           Number of codebooks 1-16 (default: 12)
  --temperature <float>       Sampling temperature (default: 0.6)
  --top-k <int>               Top-k sampling (default: 50)
  --max-tokens <int>          Maximum tokens (default: 500)
  --no-retry                  Disable retry logic
  --no-silence-trim           Disable silence trimming
  --benchmark                 Run 6-sentence benchmark
  --help, -h                  Show help
```

Key difference from `mlx-audio-swift-tts`: the `holler` CLI produces production-quality audio (silence-trimmed, faded, retried). Default codebooks = 12 (fast). `mlx-audio-swift-tts` is a raw model exerciser.

### HollerKit Public API

See "Public API Design" section above for the full three-tier API (batch, streaming, session).

### Configuration Defaults (matching Python server.py)

```swift
public struct HollerConfiguration: Sendable {
    public var temperature: Float = 0.6
    public var topK: Int = 50
    public var codebooks: Int = 12               // server.py DEFAULT_CODEBOOKS
    public var maxTokens: Int = 500              // server.py MAX_TOKENS
    public var streamingChunkTokens: Int = 3     // server.py STREAM_CHUNK_TOKENS

    // Silence handling
    public var silentAbortTokens: Int = 16       // server.py SILENT_ABORT_TOKENS
    public var speechOnsetThresholdRMS: Float = 0.007
    public var speechOnsetPreRollMs: Float = 150
    public var fadeOutMs: Float = 20

    // Retry
    public var maxRetries: Int = 3
    public var retryTemperatureStep: Float = 0.1
    public var minAudioSecondsPerWord: Float = 0.08

    // Carryover (Phase 2)
    public var carryoverPauseMinMs: Int = 150    // server.py CARRYOVER_PAUSE_MIN_MS
    public var carryoverPauseMaxMs: Int = 250    // server.py CARRYOVER_PAUSE_MAX_MS
}
```

## Model Distribution (Holler.app)

The model checkpoint (~1.1GB 6-bit) downloads separately on first launch — not bundled in the app.

- Download from HuggingFace (`sentium/holler-0.6b-6bit`) or our own CDN — TBD
- Progress indicator during download, app works offline after that
- Stored in app's caches directory or a shared location

This keeps the app download small and lets us update voices without app updates.

## What This Could Replace in ivi

If Swift integration works well, the entire Python TTS sidecar stack becomes unnecessary:

- `sidecar/tts-sidecar-holler.py` — HTTP wrapper
- `holler/inference/server.py` — Python inference server
- `ivi-app/App.swift: launchTTSSidecar()` — process lifecycle
- `ivi-app/App.swift: killTTSSidecar()` — SIGTERM/SIGKILL dance
- `ivi-app/App.swift: killProcessOnPort(52946)` — port cleanup
- `ivi-app/TTSService.swift: generateAndPlay()` — HTTP streaming client
- `ivi-app/TTSService.swift: waitForSidecar()` — health polling
- `ivi-app/TTSService.swift: TTSStreamDelegate` — URLSession delegate

Would become something like:
```swift
// In TTSService.swift
import HollerKit

private var model: HollerModel?
private var session: SpeechSession?

func ensureModelLoaded() async {
    guard model == nil else { return }
    model = try? await HollerModel.load(repo: "sentium/holler-0.6b-6bit")
}

// Called when ivi starts speaking a response
func startSpeaking(voice: String) async {
    await ensureModelLoaded()
    guard let model else { return }
    session = model.makeSession(voice: voice)

    // Start consuming audio in background
    Task {
        guard let session else { return }
        for try await chunk in session.audio {
            TTSPlayer.shared.scheduleBuffer(chunk.samples, sampleRate: chunk.sampleRate)
        }
    }
}

// Called as LLM tokens arrive
func feedText(_ token: String) async {
    await session?.feed(token)
}

// Called when LLM response is complete
func finishSpeaking() async {
    await session?.finish()
    session = nil
}

// Called when user interrupts (barge-in)
func cancelSpeaking() {
    session?.cancel()
    session = nil
    TTSPlayer.shared.stop()
}
```

No processes. No ports. No HTTP. No SIGKILL. No GIL deadlocks. No health polling. But the Python sidecar stays as a fallback until this is proven solid.

## PR Opportunities

1. **mlx-audio-swift:** compile() + lazy eval + codebook skip + streaming optimizations. High impact, single maintainer (Prince Canuma), builds relationship.
2. **ml-explore/mlx-swift-examples issue #453:** TTS example using mlx-audio-swift. Low effort, high visibility.
3. **mlx-audio (Python):** Reduce `mx.clear_cache()` frequency in streaming loop (already planned in holler CLAUDE.md).

## References

- `mlx-audio-swift` repo: github.com/Blaizzy/mlx-audio-swift
- `speech-swift` repo: github.com/soniqo/speech-swift (reference for compile patterns, NOT for inference)
- `mlx-swift` repo: github.com/ml-explore/mlx-swift
- Holler inference server: `holler/inference/server.py` (Python reference implementation)
- Holler CLAUDE.md: `holler/CLAUDE.md` (training, quantization, known issues)
- WWDC 2025: "Get started with MLX for Apple Silicon"
- speech-swift compile patterns: `Sources/Qwen3TTS/Qwen3TTS.swift` lines 1076-1128 (setupCompilation)
