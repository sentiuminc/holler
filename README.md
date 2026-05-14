# Holler

An open-source text to speech model with 6 American voices, and a highly performant inference engine for Apple Silicon. Finetuned from [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), and optimized for local AI assistant usecases. The inference server supports streaming text in and audio out, and has 140ms TTFA latency with an RTF of up to 2.1x depending on the configuration.

Built for [ivi](https://ivi.computer), an ambient AI assistant for macOS. We open-sourced it because Qwen3-TTS is the best local TTS model available but ships with only 2 mediocre English voices.

<video src="holler-intro.mp4" autoplay muted loop playsinline width="100%"></video>

> **[Listen to all voice samples on HuggingFace](https://huggingface.co/sentiuminc/holler-0.6b#voices)** | **[12 vs 16 codebook comparison](https://huggingface.co/sentiuminc/holler-0.6b-6bit#codebook-comparison)**

## Model Variants

| | **bf16** | **6-bit** |
|--|----------|-----------|
| **Repo** | [`sentiuminc/holler-0.6b`](https://huggingface.co/sentiuminc/holler-0.6b) | [`sentiuminc/holler-0.6b-6bit`](https://huggingface.co/sentiuminc/holler-0.6b-6bit) |
| **Size** | 2.3 GB | 1.7 GB |
| **Use case** | Best quality, offline generation | Streaming, real-time assistants |

**Which should I use?**

- **Start with bf16 + 16 codebooks.** This is the full-precision model with all 16 codec books. Best voice fidelity, most natural prosody. Use this for anything where quality matters more than speed — pre-generated audio, content creation, or any use case where you're not streaming token-by-token from an LLM.

- **Use 6-bit + 12 codebooks for streaming.** The quantized model with 12 of 16 codec books skips the highest-frequency acoustic detail for an 18% speed gain with a small quality loss. Lower RAM footprint. This is what [ivi](https://ivi.computer) uses for real-time voice responses where time-to-first-audio and RTF matter most.

The weights are standard Qwen3-TTS checkpoints — you can load them with any compatible inference engine and they'll work. But Holler's own inference pipeline (both the Swift HollerKit/CLI and the Python server) exists for a reason: it's significantly faster than stock mlx-audio (~2x), and it handles the model's quirks — codec warmup silence, occasional empty generations, stochastic EOS cutoffs — so your app doesn't have to. If you just want to drop in the weights and call `generate()`, go for it. If you want it fast and reliable, use Holler's pipeline.

Codebooks are configurable at inference time (not baked into the model), so you can use also 16 codebooks with the 6-bit model or 12 with bf16 if you want.

## Performance (M1 Pro, 16GB)

Measured with HollerKit (Swift), `--session` mode, 10 multi-sentence paragraphs. Medians reported — individual generations vary depending on text length and codec warmup.

| Metric | bf16 / 16cb | 6-bit / 16cb | 6-bit / 12cb |
|--------|-------------|--------------|--------------|
| TTFA (median) | ~200ms | ~170ms | ~147ms |
| Real-time factor | 0.68 | 0.54 | 0.47 |
| Speed | 1.5x real-time | 1.8x real-time | 2.1x real-time |
| Metal RAM | ~2.4 GB | ~1.7 GB | ~1.7 GB |
| Download size | 2.3 GB | 1.7 GB | 1.7 GB |

**About TTFA:** This is time to first *audible speech*, not time to first audio chunk. Qwen3-TTS (like most codec language models) produces 80-800ms of near-silence at the start of each generation. Holler detects and trims this automatically, so the TTFA reported here is when you actually hear the voice start speaking. Most TTS benchmarks and providers report time to first byte or first audio frame, which includes this silence. These numbers are what you actually experience.

## Voices

| Voice | Gender | Style | Description |
|-------|--------|-------|-------------|
| **Nora** | Female | Polished, articulate, warm | Bright, expressive |
| **Tessa** | Female | Bright, sharp, articulate | Dry humor, quick |
| **Kit** | Neutral | Steady, analytical, professional | Clear, warm |
| **Dakota** | Male | Crisp, outdoorsy, steady | Grounded, natural |
| **Joe** | Male | Bright, punchy, high energy | Casual, enthusiastic |
| **Oliver** | Male | Deep, confident, clear | Measured, deliberate |

Each voice was designed with Qwen3-TTS VoiceDesign, cloned into 300-500 training clips, enhanced, manually curated, and fine-tuned into the 0.6B model. See [How This Model Was Made](#how-this-model-was-made) for the full pipeline.

## Ways to use it

- **[CLI](#cli)** — run `./holler --text 'Hello' --talk`. Zero setup, uses HollerKit under the hood. **Recommended** — produces the most stable, natural-sounding audio.
- **[HollerKit](#hollerkit-swift-package)** (Swift) — native Swift package for macOS apps. Stream text in, get audio out. Best for production integration.
- **[Python server](#python-server)** — HTTP API with streaming audio. Good for quick integration or non-Swift projects.

## HollerKit (Swift Package)

Native Swift library for integrating Holler into macOS apps.

The model works fine with plain inference — generate text, get audio. But real assistant use cases need more: text arrives as a stream of tokens from an LLM, you need continuous speech output without gaps, and the model occasionally produces silence or artifacts that need to be caught and retried. HollerKit handles all of this:

- **Sentence buffering** — feed text in any chunk size (characters, words, LLM tokens), HollerKit accumulates and splits on sentence boundaries automatically
- **Streaming generation** — audio chunks stream out as they're generated, no waiting for the full sentence
- **KV cache carryover** — prosody carries naturally across sentences instead of each sentence starting cold
- **Silence detection** — codec warmup silence is trimmed, post-speech silence triggers early cutoff
- **Retry logic** — failed generations (no speech, too short) are caught and retried with bumped temperature
- **Streaming AGC** — automatic gain control targeting -20 LUFS, equalizes loudness across all voices in real-time

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
config.temperature = 0.7          // Sampling temperature
config.codebooks = 16             // Codec books (16 = best quality, 12 = fast streaming)
config.maxRetries = 3             // Retry attempts on failed generation
config.log = { print($0) }       // Enable debug logging

// Default model is sentiuminc/holler-0.6b (bf16)
// For faster streaming, use the 6-bit variant:
let model = try await HollerModel.load(repo: "sentiuminc/holler-0.6b-6bit", configuration: config)
```

## Python Server

HTTP API with streaming audio. Stable enough for production use — we run it in our own development pipeline.

### Quick Start

```bash
git clone https://github.com/sentiuminc/holler.git
cd holler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the server (downloads sentiuminc/holler-0.6b from HuggingFace on first run, ~2.3GB)
python3 inference/server.py

# Or use the 6-bit model for lower RAM:
python3 inference/server.py -c sentiuminc/holler-0.6b-6bit

# Open http://localhost:8100 in your browser, or:
curl "http://localhost:8100/tts?text=Hello+world" -o hello.wav
```

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
| `temperature` | float | 0.7 | Sampling temperature |
| `top_k` | int | 50 | Top-k sampling |
| `n_codebooks` | int | 16 | Codec books (16 best quality, 12 fastest) |
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

## How This Model Was Made

Holler is a supervised fine-tune of [Qwen3-TTS-12Hz-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) with custom voice data. The full pipeline and all tooling are in this repo — here's what we did and what we learned.

### Pipeline

1. **Voice design** — describe a voice character in text, generate reference audio with [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (1.7B model). Multiple rounds of candidates per voice, pick the best 7-11s clip.
2. **Data generation** — clone 500 training clips per voice using the reference audio with Qwen3-TTS 1.7B-Base. Runs locally on Apple Silicon via mlx-audio.
3. **Enhancement** — DeepFilterNet3 noise removal (single pass only — never cascade enhancers), K-weighted LUFS normalization to -22, IIR notch de-esser at 5500Hz Q=3.0.
4. **Curation** — manual listening pass with `tools/curate_clips.py` (clip tinder). Typically keeps 300-400 of 500 clips.
5. **Training** — SFT on a CUDA GPU (Vast.ai, 3090+, ~$0.15/hr). See recipe below.
6. **Quantization** — 6-bit affine g64 via `mlx_audio.convert` for the fast variant.

Full details in [`docs/training-runbook.md`](docs/training-runbook.md).

### Recipe

<!-- TODO: Read session logs and expand with more detail on learnings -->

```
lr=5e-7, cosine warmup scheduler
2 epochs, batch_size=2, grad_accum=4, weight_decay=0.01
Embedding normalization: L2-norm all ECAPA-TDNN speaker embeddings to magnitude 10.0
All training data at uniform -22 LUFS
```

### What We Learned

- **lr=5e-7 with cosine warmup** is the sweet spot for multi-voice. lr=1e-7 (the community standard for single-voice) is too conservative — cosine barely decays. The scheduler `total_steps` must be `(steps_per_epoch × num_epochs) / grad_accum` to get a proper curve.
- **Embedding normalization is critical** for multi-voice. The ECAPA-TDNN speaker encoder bakes loudness into embeddings — voices with louder refs get higher embedding norms, causing the model to generate hotter output. L2-normalizing to magnitude 10.0 strips loudness while preserving voice identity (direction).
- **Uniform LUFS across all voices.** Per-voice LUFS adjustments cause unpredictable cross-voice interference through shared transformer weights.
- **2 epochs optimal.** 3 epochs overfits. Weight decay 0.01 (0.05 is too aggressive, kills voice character).
- **Ref audio length matters.** 7-11s refs produce consistent voice identity. 4-5s refs are variable.
- **Temperature 0.7 at inference** (the default) gives the best prosody. Training data was generated at 0.85 — going lower than 0.7 makes speech sound rushed.

Scripts: `training/sft_12hz.py` (single-voice), `training/sft_12hz_multivoice.py` (multi-voice with per-voice JSONL and cached embedding injection).

### Tools

```bash
# Enhance training data
.venv-enhance-audio/bin/python tools/enhance_clean.py --voice <name>

# Analyze voice quality (spectral, LUFS, peak stats)
.venv-enhance-audio/bin/python tools/analyze_voice_quality.py --source <dir> --label <name> --n 80

# Manual curation (clip tinder — listen, keep/skip)
.venv-enhance-audio/bin/python tools/curate_clips.py --voice <name>

# Quality benchmark (Whisper-based WER + gap analysis)
.venv/bin/python tools/benchmark_quality.py --checkpoint <path> --mode raw --temperature 0.7
```

## Requirements

- macOS with Apple Silicon (M1 or later)
- For Swift: [Xcode 16+](https://developer.apple.com/xcode/) (full app, not just Command Line Tools — MLX requires Metal shader compilation which only Xcode provides)
- For Python: Python 3.13+
- ~2GB free RAM for inference (6-bit), ~2.5GB for bf16

## Attribution

Holler is a fine-tune of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba Cloud (Apache 2.0). All credit for the underlying architecture goes to them.

## License

Apache 2.0
