# Holler

A fast, reliable voice engine for local AI assistants on Apple Silicon.

Holler exists because nothing else fills this gap: there's no local TTS that sounds this good, runs this fast, and handles real-time LLM streaming out of the box. Kokoro and Soprano are fast but sound flat and robotic, especially sentence-by-sentence. Cloud APIs like ElevenLabs and Cartesia sound great but add latency and extreme cost. Holler gives you both — production-quality voices at ~130ms time-to-first-audio, fully on-device, with a streaming pipeline built for the way AI assistants actually work: text arrives token by token, and speech needs to flow out continuously.

Built on [Qwen3-TTS-0.6B](https://github.com/QwenLM/Qwen3-TTS). The model is a fine-tune — you can run it with any Qwen3-TTS-compatible inference engine and it works fine. But the real value is in Holler's inference pipeline: silence detection, retry logic, KV cache carryover for natural prosody across sentences, and all the guardrails that handle the model's quirks so your app doesn't have to.

Holler is built this for [ivi](https://ivi.computer), a local AI assistant for macOS.

> **Samples coming soon.** Voice demos and a video walkthrough are in progress.

## Ways to use it

- **[HollerKit](#hollerkit-swift-package)** (Swift) — native Swift package for macOS apps. Stream text in, get audio out. Best for production integration.
- **[Python server](#python-server)** — HTTP API with streaming audio. Stable, good for any language or quick integration.
- **[CLI](#cli)** — download the binary, run `./holler --text 'Hello' --talk`. Loads the model fresh each run (~1s), but zero setup.
- **Holler.app** — standalone Mac app (coming soon).

## Performance (M1 Pro, 6-bit)

| Metric | Value |
|--------|-------|
| Time to first audio | **~130ms** |
| Real-time factor | **0.38–0.49** (2–2.6x real-time) |
| Model load | **~1s** (cached) |
| Metal RAM | 1.7 GB |
| Download size | 1.7 GB (model + speech tokenizer) |

TTFA is measured from generation start to first audible speech — identical between Swift and Python. RTF varies by runtime: Python's custom generate loop sustains higher throughput (0.38) than mlx-audio-swift's streaming API (0.49).

The codec decoder produces ~220ms of leading silence on some generations (a known Qwen3-TTS architecture behavior). Holler detects and trims this automatically, so the reported TTFA is always time to actual speech. When there's no leading silence (majority of generations), TTFA matches raw model speed at ~130ms.

## HollerKit (Swift Package)

Native Swift library for integrating Holler into macOS apps.

The model works fine with plain inference — generate text, get audio. But real assistant use cases need more: text arrives as a stream of tokens from an LLM, you need continuous speech output without gaps, and the model occasionally produces silence or artifacts that need to be caught and retried. HollerKit handles all of this:

- **Sentence buffering** — feed text in any chunk size (characters, words, LLM tokens), HollerKit accumulates and splits on sentence boundaries automatically
- **Streaming generation** — audio chunks stream out as they're generated, no waiting for the full sentence
- **KV cache carryover** — prosody carries naturally across sentences instead of each sentence starting cold
- **Silence detection** — codec warmup silence is trimmed, post-speech silence triggers early cutoff
- **Retry logic** — failed generations (no speech, too short) are caught and retried with bumped temperature

### Quick Start

Add HollerKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/sentiuminc/holler.git", from: "1.0.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "HollerKit", package: "holler"),
    ]),
]
```

Generate speech:

```swift
import HollerKit

let model = try await HollerModel.load()

// Simple: text in, audio out
let audio = try await model.synthesize("Hello world", voice: "kit")
// audio.samples: [Float], audio.sampleRate: 24000

// Streaming: get audio chunks as they're generated
for try await chunk in model.stream("Hello world", voice: "kit") {
    player.scheduleBuffer(chunk.samples)
}

// LLM integration: feed tokens, get audio
let session = model.makeSession(voice: "kit")

// Consumer side — runs concurrently
Task {
    for try await chunk in session.audio {
        player.scheduleBuffer(chunk.samples)
    }
}

// Producer side — feed text as it arrives from the LLM
for await token in llmStream {
    session.feed(token)
}
await session.finish()
```

If you just need to generate speech from complete text (not streaming from an LLM), `model.synthesize()` or `model.stream()` are all you need — no session required.

### CLI

```bash
# Build once (~3 min first time)
./build.sh

# Speak text through your speakers
./holler --text 'Hello world' --talk

# Save to file
./holler --text 'Hello world' --output hello.wav

# Simulate LLM streaming (token-by-token with sentence buffering)
./holler --session --text 'Sure. Let me check that for you. I think the answer is forty two.'

# Benchmark
./holler --benchmark

# Debug mode — see the full pipeline (sentence splits, chunk RMS, cache state, retries)
./holler --session --debug --text 'Your text here'
```

> `build.sh` uses xcodebuild under the hood because MLX requires compiled Metal shaders (`.metallib`) which only Xcode can produce. `swift build` compiles the Swift code but skips Metal.

### Configuration

```swift
var config = HollerConfiguration()
config.temperature = 0.6          // Sampling temperature
config.codebooks = 12             // Codec books (12 = fast, 16 = max quality)
config.maxRetries = 3             // Retry attempts on failed generation
config.log = { print($0) }       // Enable debug logging

let model = try await HollerModel.load(repo: "sentium/holler-0.6b-6bit", configuration: config)
```

## Python Server

HTTP API with streaming audio. Stable enough for production use — we run it in our own development pipeline.

### Quick Start

```bash
git clone https://github.com/sentiuminc/holler.git
cd holler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python3 inference/server.py

# Open http://localhost:8100 in your browser, or:
curl "http://localhost:8100/tts?text=Hello+world" -o hello.wav
```

On first run, the server downloads `sentium/holler-0.6b-6bit` from HuggingFace (~1.7GB, cached for future runs).

### API

#### `POST /speak` — Streaming audio

Returns audio as it's generated. Float32 PCM at 24kHz, chunked transfer encoding.

```bash
curl -X POST http://localhost:8100/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather looks great today.", "voice": "kit"}'
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to speak |
| `voice` | string | first available | Voice name |
| `temperature` | float | 0.6 | Sampling temperature |
| `top_k` | int | 50 | Top-k sampling |
| `n_codebooks` | int | 12 | Codec books (12 fastest, 16 max quality) |
| `continue` | bool | false | Carry over prosody from previous generation |

#### `GET /tts` — WAV file

```bash
curl "http://localhost:8100/tts?text=Hello+world&voice=kit" -o hello.wav
```

#### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server status, model name, available voices |
| `GET /benchmark` | 6-sentence benchmark with TTFA/RTF results |
| `GET /` | Browser test UI with real-time playback |

## Voices

| Voice | Description | Status |
|-------|-------------|--------|
| Kit | Androgynous, clear, warm | Trained |
| Dakota | Male, grounded, natural | Trained |
| Nora | Female, bright, expressive | In curation |
| Joe | Male, deep, steady | In curation |
| + 6 more | From 22 curated candidates | Coming soon |

Each voice starts as a VoiceDesign text prompt (describing the voice character), which produces a reference clip. That reference is then used with Qwen3-TTS voice clone (1.7B-Base) to generate 500 training clips, which are enhanced, manually curated, and used to fine-tune the 0.6B model. The result is a standard Qwen3-TTS checkpoint — you can run it with any compatible inference tool, or use Holler's pipeline for the full experience.

## Training Your Own Voices

The full training pipeline is documented in `docs/training-runbook.md`:

1. **Voice design** — create voice identity via VoiceDesign text prompts, get reference audio
2. **Data generation** — 500 clips per voice via voice clone (1.7B-Base) using the reference, locally on Mac via mlx-audio
3. **Enhancement** — DeepFilter noise removal, LUFS normalization, de-essing
4. **Curation** — manual listening pass
5. **Training** — lr=1e-7, 2 epochs, text_projection patch, bf16
6. **Quantization** — 6-bit affine g64

Scripts: `training/` for fine-tuning, `tools/` for data generation and curation.

## Requirements

- macOS with Apple Silicon (M1 or later)
- For Swift: [Xcode 16+](https://developer.apple.com/xcode/) (full app, not just Command Line Tools — MLX requires Metal shader compilation which only Xcode provides)
- For Python: Python 3.13+
- ~2GB free RAM for inference

## Attribution

Holler is a fine-tune of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba Cloud (Apache 2.0). All credit for the underlying architecture goes to them.

## License

Apache 2.0
