# Holler

High-quality American English voices for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), optimized for real-time local inference on Apple Silicon.

**360ms to first audio. 2x real-time. 1.7GB RAM. Fully local.**

Holler is a fine-tuned [Qwen3-TTS-12Hz-0.6B](https://github.com/QwenLM/Qwen3-TTS) with curated American English voices. It runs entirely on your Mac via Metal. No cloud, no API keys, no internet required after the initial model download.

Two ways to use it:
- **HollerKit** (Swift) — native Swift package for macOS apps. Stream text in, get audio out.
- **Python server** — HTTP API with streaming audio. Good for prototyping and non-Swift integrations.

## HollerKit (Swift Package)

Native Swift library for integrating Holler into macOS apps. Supports streaming text input (LLM integration), automatic sentence buffering, KV cache carryover for natural prosody, silence detection, and retry logic.

### Quick Start

Add HollerKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/sentium/holler.git", from: "1.0.0"),
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

// Streaming: get audio chunks as they're generated (~360ms to first audio)
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

`SpeechSession` handles everything automatically: sentence boundary detection, streaming generation, silence trimming, KV cache carryover between sentences, and retry on failed generations. Feed it text in any chunk size — single characters, whole sentences, random LLM-sized pieces — it accumulates and splits on sentence boundaries internally.

### CLI

The package includes a command-line tool for testing:

```bash
cd swift/HollerKit

# Speak text through your speakers
swift run holler --text "Hello world" --talk

# Save to file
swift run holler --text "Hello world" --output hello.wav

# Simulate LLM streaming (token-by-token with sentence buffering)
swift run holler --session --text "Sure. Let me check that for you. I think the answer is forty two."

# Benchmark
swift run holler --benchmark

# Debug mode — see the full pipeline (sentence splits, chunk RMS, cache state, retries)
swift run holler --session --debug --text "Your text here"
```

### Configuration

```swift
var config = HollerConfiguration()
config.temperature = 0.6          // Sampling temperature
config.codebooks = 12             // Codec books (12 = fast, 16 = max quality)
config.maxRetries = 3             // Retry attempts on failed generation
config.log = { print($0) }       // Enable debug logging

let model = try await HollerModel.load(repo: "sentium/holler-0.6b-6bit", configuration: config)
```

### Performance (M1 Pro, release build)

| Metric | Value |
|--------|-------|
| Time to first audio | **360ms** avg |
| Real-time factor | **0.49** avg (2x real-time) |
| Metal RAM | 1.7 GB |
| Model on disk | 1.1 GB (6-bit) |

TTFA includes codec warmup silence trimming — the 360ms is time to first *audible* speech, not first raw audio chunk.

## Python Server

HTTP API with streaming audio. Good for prototyping and non-Swift integrations.

### Quick Start

```bash
git clone https://github.com/sentium/holler.git
cd holler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python3 inference/server.py

# Open http://localhost:8100 in your browser, or:
curl "http://localhost:8100/tts?text=Hello+world" -o hello.wav
```

On first run, the server downloads `sentium/holler-0.6b-6bit` from HuggingFace (~1.1GB, cached for future runs).

### API

#### `POST /speak` — Streaming audio

Returns audio as it's generated. Float32 PCM at 24kHz, chunked transfer encoding. First audio arrives in ~139ms.

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

### Python Performance (M1 Pro)

| Metric | Value |
|--------|-------|
| Time to first audio | **139ms** avg |
| Real-time factor | **0.38** avg (2.6x real-time) |
| Metal RAM | 1.7 GB |

## Voices

Holler v1 ships with curated American English voices (2 trained, 8 more coming):

| Voice | Description | Status |
|-------|-------------|--------|
| Kit | Androgynous, clear, warm | Trained |
| Dakota | Male, grounded, natural | Trained |
| + 8 more | From 22 curated candidates | Coming soon |

All voices are created using Qwen3-TTS VoiceDesign, then fine-tuned with 400-500 curated clips per voice.

## Training Your Own Voices

The full training pipeline is documented in `docs/training-runbook.md`:

1. **Voice design** — create voice identity via VoiceDesign text prompts
2. **Data generation** — 500 clips per voice on a cloud GPU (~$0.15/voice)
3. **Enhancement** — DeepFilter noise removal, LUFS normalization, de-essing
4. **Curation** — manual listening pass
5. **Training** — lr=1e-7, 2 epochs, text_projection patch, bf16
6. **Quantization** — 6-bit affine g64

Scripts: `training/` for fine-tuning, `tools/` for data generation and curation.

## Requirements

- macOS with Apple Silicon (M1 or later)
- For Swift: Xcode 16+ / Swift 6.2+
- For Python: Python 3.13+
- ~2GB free RAM for inference

## Attribution

Holler is a fine-tune of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba Cloud (Apache 2.0). All credit for the underlying architecture goes to them.

## License

Apache 2.0
