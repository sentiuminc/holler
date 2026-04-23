# Session Log: 2026-04-23 — Inference Architecture Decision

**Project:** Holler
**Session ID:** `c989f897-2faf-4fae-8df2-439540fef25a`
**What happened:** Spent the day figuring out how to run Holler — evaluated Swift libraries, tested speech-swift extensively, ultimately decided on Python mlx-audio as the inference runtime. Built comprehensive sentence-queue benchmarks.

---

## What I Did

### 1. Researched the Swift TTS ecosystem

Surveyed four options for running Qwen3-TTS on Apple Silicon in Swift:
- **soniqo/speech-swift** (651 stars) — full toolkit, real chunked streaming, actively maintained
- **Blaizzy/mlx-audio-swift** (588 stars) — Swift port of mlx-audio
- **AtomGradient/swift-qwen3-tts** (8 stars) — interesting compression research, fake streaming
- **rekuenkdr/Qwen3-TTS-streaming** (72 stars) — CUDA-only, not applicable

Also researched rekuenkdr's work in detail — their two-phase streaming concept (208ms first audible vs 570ms baseline) and OVA voice assistant.

### 2. Explored ivi's current architecture

ivi has no TTS integrated. The sidecar is a Bun HTTP+SSE server bridging Swift ↔ Claude CLI. The TTS architecture was designed (SentenceBuffer → serial queue → AVAudioEngine) but never implemented — stalled at "we need a Python sidecar."

### 3. Tested speech-swift extensively

Built a minimal Swift test harness (`swift-test/`) that imports Qwen3TTS from speech-swift. Discovered:

- **Weight formats are identical** between mlx-audio's Python converter and speech-swift's models (verified all 900 model keys + 496 tokenizer keys — same dtypes, same shapes)
- **Non-streaming works:** 10/10 generations, 1.4s model load, 38ms/step, RTF 0.59-0.67
- **Streaming works initially:** 90-142ms TTFA, but crashes after ~13 total inference calls
- **Audio quality issues:** Pops in both streaming AND non-streaming, end cutoff. These artifacts don't exist in Python mlx-audio with the same checkpoint.

Stock `aufklarer/` model ran 20 calls without issues — the streaming crash is specific to our checkpoint, the audio artifacts are general.

**Decision: Not using speech-swift as a dependency.** Audio quality is disqualifying. Reference code kept at `speech-swift/` for future use.

### 4. Realized Holler is a model, not an inference library

Chris pushed back on maintaining two versions (Swift + Python). The right framing: Holler ships trained weights on HuggingFace. Users run via mlx-audio (Python), which already works perfectly. One version, one path.

### 5. Built comprehensive mlx-audio benchmarks

Created `inference/benchmark_sentence_queue.py` — simulates ivi's real pattern with realistic LLM response sentences. Tested all three voices (Katie v6 4-bit, Katie v7 bf16, Joe v7 bf16).

Key results for Katie v6 4-bit:
- Streaming TTFA: 64-80ms consistently
- "Hey!" non-streaming: 312ms, "Exactly." non-streaming: 762ms
- Generation always faster than real-time, queue never starves
- 57 non-streaming + 24 streaming generations, zero crashes, zero artifacts

### 6. Streaming vs non-streaming comparison

Measured actual time-to-ears for both approaches:
- "Exactly." — streaming: 126ms to ears, non-streaming: 762ms (streaming wins by 636ms)
- "Yes, I agree." — streaming: 68ms to ears, non-streaming: 726ms (streaming wins by 658ms)
- "Hey!" — streaming: 319ms (silence chunks), non-streaming: 312ms (tie)

Streaming wins dramatically for most sentences. Exception: very short exclamations where codec warmup silence means non-streaming ties.

### 7. Confirmed v7 quality issues

Katie v7 "Okay." generates 6 seconds of near-silence (peak 0.376). Joe v7 "Okay." generates pure silence (peak 0.000). Longer sentences sound okay but are slower than v6 4-bit. The v7 multi-voice quality regression on short utterances is confirmed and reproducible.

### 8. Cleanup

Deleted 6.9GB: katie-v6-8bit (redundant), katie-joe-v7-4bit (re-quantizable), speech-swift/.build, swift-test/.build. Quantized v7 to 4-bit during session but deleted to save space.

## What I Learned

**Don't chase "better" when "working" is right there.** The entire speech-swift detour was trying to find something better than Python mlx-audio — which already produced clean, fast audio with zero issues. The Swift ecosystem isn't mature enough for custom fine-tuned TTS models yet.

**Chris doesn't want two versions of anything.** One implementation, one path. If it's Python, it's Python. Don't propose maintaining Swift + Python in parallel.

**Chris listens to the audio.** I can't hear artifacts — he can. Always generate audio and have him listen before declaring something "works." The speech-swift audio had pops I couldn't detect through data analysis alone.

**The "Python in Activity Monitor" concern is real but not blocking.** It's a UX issue for shipping, not a technical blocker. The architecture decision is: mlx-audio Python sidecar with WebSocket, queue-based chunked streaming.

## For Next Time

**Immediate next step:** Wire Katie v6 4-bit into ivi via mlx-audio Python sidecar. WebSocket-based, queue-based chunked streaming. Stream everything, skip silence chunks (peak < 0.05). First audible audio at 68-126ms.

**The ivi integration architecture:**
- mlx-audio Python process (sidecar) with WebSocket server
- LLM streams text → sentence boundary detection → stream TTS per sentence
- Audio chunks queue up → skip silence → play continuously
- Generation faster than real-time, so queue never starves

**v7 is parked.** Don't try to fix v7 quality issues next session. Focus on getting v6 Katie working in ivi first.

**speech-swift reference code** is at `speech-swift/` — well-documented 4-stage pipeline (Qwen3TTS.swift), weight loading patterns, transformer implementations. Useful if we ever build our own Swift implementation.

## Files Changed

- `CLAUDE.md` — Updated current state to 2026-04-23, added Inference Architecture section with streaming benchmarks, updated What's Next priorities, reflected 8-bit deletion
- `inference/benchmark_sentence_queue.py` — NEW: comprehensive sentence-queue benchmark simulating ivi's pattern
- `speech-swift/` — NEW: cloned soniqo/speech-swift repo (reference code only, .build deleted)
- `swift-test/` — NEW: minimal Swift test harness (Package.swift + main.swift, .build deleted)
- `checkpoints/katie-v6-8bit/` — DELETED (redundant, 4-bit is the pick)
- `checkpoints/katie-joe-v7-4bit/` — CREATED then DELETED (can re-quantize anytime)
- Memory files created: project_speech_swift_integration.md, user_context.md, feedback_audio_to_downloads.md, feedback_keep_instances.md, project_origin.md, reference_vastai.md, reference_ivi_venv.md
