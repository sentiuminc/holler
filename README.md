# Holler

High-quality American English voices for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), optimized for real-time local inference on Apple Silicon.

**139ms to first audio. 2.6x real-time. 1.7GB RAM. Fully local.**

Holler is a fine-tuned [Qwen3-TTS-12Hz-0.6B](https://github.com/QwenLM/Qwen3-TTS) with curated American English voices and a fast streaming inference server. It runs entirely on your Mac via [mlx-audio](https://github.com/ml-explore/mlx-audio) and Metal. No cloud, no API keys, no internet required.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/sentium/holler.git
cd holler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the server
python3 inference/server.py

# Open http://localhost:8100 in your browser, or:
curl "http://localhost:8100/tts?text=Hello+world" -o hello.wav
```

On first run with no local checkpoint, the server downloads `sentium/holler-0.6b-6bit` from HuggingFace (~1.1GB, cached for future runs).

## API

### `POST /speak` -- Streaming audio

Returns audio as it's generated. Float32 PCM at 24kHz, chunked transfer encoding. First audio arrives in ~139ms.

```bash
curl -X POST http://localhost:8100/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "The weather looks great today.", "voice": "kit"}'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to speak |
| `voice` | string | first available | Voice name |
| `temperature` | float | 0.6 | Sampling temperature |
| `top_k` | int | 50 | Top-k sampling |
| `n_codebooks` | int | 12 | Codec books (max 16, 12 is fastest with negligible quality loss) |
| `continue` | bool | false | Carry over prosody from previous generation |

**Response headers:** `X-Sample-Rate: 24000`, `X-Format: float32-pcm`

### `GET /tts` -- WAV file download

Generates the full audio and returns a complete WAV file. Simple and easy to use, but you wait for the entire generation to finish.

```bash
curl "http://localhost:8100/tts?text=Hello+world&voice=kit" -o hello.wav
```

**Query parameters:** `text` (required), `voice`, `temperature`

### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server status, model name, available voices |
| `GET /benchmark` | Runs a 6-sentence benchmark and returns TTFA/RTF results |
| `GET /` | Browser test UI with real-time streaming playback |

## CLI Options

```
python3 inference/server.py [options]

  --checkpoint, -c PATH   Model checkpoint directory (default: auto-detect)
  --port, -p PORT         Server port (default: 8100)
  --voice, -v NAME        Default voice (default: first voice in checkpoint)
  --no-ui                 Disable browser test UI at /
```

## Performance

Measured on M1 Pro, 6-bit affine quantization, 12 codebooks:

| Metric | Value |
|--------|-------|
| Time to first audio | **139ms** avg |
| Real-time factor | **0.38** avg (2.6x real-time) |
| Metal RAM | 1.7 GB |
| CPU during generation | ~8% |
| Model on disk | 1.1 GB (6-bit) |

### Why not use mlx-audio directly?

mlx-audio's built-in streaming gives RTF ~0.76. Holler's server gives RTF ~0.38. The difference:

1. mlx-audio calls `mx.eval()` + `mx.clear_cache()` after every chunk, thrashing the Metal pipeline
2. Our custom generate loop does one `mx.eval()` per token with chunked decode
3. We use 12 of 16 codebooks by default, skipping the highest-frequency acoustic detail for 18% speedup with negligible quality loss

## Voices

Holler v1 ships with 10 curated American English voices (work in progress -- currently 2 are trained):

| Voice | Description | Status |
|-------|-------------|--------|
| Kit | Androgynous, clear, warm | Trained |
| Dakota | Male, grounded, natural | Trained |
| + 8 more | From 22 curated candidates | Coming soon |

All voices are created using Qwen3-TTS VoiceDesign (text-described voice synthesis), then fine-tuned with 400-500 curated training clips per voice.

## Architecture

```
inference/server.py
  -> MLX worker thread (single persistent thread for all inference)
  -> Custom generate loop (talker LLM -> code predictor -> codec tokens at 12Hz)
  -> Streaming codec decode (3-token first chunk, then 3-token chunks)
  -> HTTP chunked transfer encoding -> client
```

All MLX operations run on a single dedicated thread via a work queue. This avoids a fatal crash caused by MLX thread-local cache destructors on short-lived HTTP threads ([mlx#2086](https://github.com/ml-explore/mlx/issues/2086)).

The server includes retry logic (up to 3 attempts with increasing temperature), silence detection and trimming, prosody carry-over between sentences, and a 30-second per-chunk timeout.

## Training

Holler uses supervised fine-tuning on the base Qwen3-TTS-12Hz-0.6B model. The full training pipeline is documented in `docs/training-runbook.md` and includes:

1. **Voice design** -- create voice identity via VoiceDesign text prompts
2. **Data generation** -- 500 clips per voice on a cloud GPU (Vast.ai, ~$0.15/voice)
3. **Enhancement** -- DeepFilter noise removal, LUFS normalization, spectral de-essing
4. **Curation** -- manual listening pass via browser-based Tinder-style tool
5. **Training** -- lr=1e-7, 2 epochs, text_projection patch, bf16
6. **Quantization** -- 6-bit affine g64 via mlx_audio.convert

Training scripts are in `training/`. Data generation and curation tools are in `tools/`.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.13+
- ~2GB free RAM for inference

## Attribution

Holler is a fine-tune of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team at Alibaba Cloud. The base model is Apache 2.0 licensed. All credit for the underlying architecture and pre-training goes to the Qwen team.

## License

Apache 2.0
