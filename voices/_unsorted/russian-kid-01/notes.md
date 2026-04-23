## Russian Kid 01

Source: Galkin comedy sketch — a Russian kid learning English in school
YouTube: https://www.youtube.com/watch?v=-2LaVuvCj-I

### How it was made

1. Downloaded the YouTube video audio
2. Extracted the clip at 00:24-00:32.5 (8.5s) — Galkin speaking English with a heavy Russian accent: "My name is Anton. Every morning I usually get up at 7 o'clock and brush my teeth."
3. Cloned via `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` with `language='english'`, `temperature=0.6`

### Ref text (must match ref.wav)

"My name is Anton. Every morning I usually get up at 7 o clock and brush my teeth."

### Gotcha: reference bleed

The model leaks the tail end of the reference audio into the start of generated speech. Original 13.5s clip ending with "go to school by bus" caused generations to start with "bus" or "and go to school". Fix: trim the ref to end cleanly mid-sentence.

### Files

- `ref.wav` — 8.5s trimmed reference (24kHz mono)
- `cloned-samples/v3_01.wav` — "Good morning. It is currently forty two degrees outside..."
- `cloned-samples/v3_02.wav` — "I ordered you an Uber. It will be here in six minutes..."
