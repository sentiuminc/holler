# Session Log: 2026-04-23 — Live Mode Planning & Handoff to ivi

**Project:** holler
**Session ID:** `8803abfd-25d4-4b26-8307-ef3e7f223ff0`
**What happened:** Context-loading session to prepare a TTS integration plan for ivi. No code written — deliverable was a handoff document.

---

## What I Did

Read all key reference files on both sides:
- Holler: `inference/live_demo.py`, `inference/benchmark_sentence_queue.py`, `CLAUDE.md`
- ivi: `sidecar/server.ts`, `ivi-app/VoiceManager.swift`, `ivi-app/App.swift`, `docs/voice-tts-research.md`, `CLAUDE.md`, `TODO.md`

Synthesized everything into `~/Desktop/Files/AI/ivi/docs/live-mode.md` — a two-part implementation plan:
1. **Part 1 (Python TTS sidecar):** All proven patterns, exact code, timing data, protocol spec
2. **Part 2 (Swift side):** Recommendations for audio playback, sentence detection, interrupt flow, call mode UX — all marked as unverified

The ivi coding session picked up the doc immediately and made substantive modifications:
- Changed protocol from WebSocket to HTTP chunked transfer (URLSessionWebSocketTask deadlocks in NSPanel context)
- Corrected model size to 1.6GB (960MB model + 651MB speech tokenizer)
- Changed silence threshold from 0.05 to 0.02 (matching live_demo.py's actual value)
- Added the `speech_started` flag pattern from live_demo.py
- Specified separate AVAudioEngine for playback (not shared with VoiceManager's recording engine)

## What I Learned

- ivi's `URLSessionWebSocketTask` has a known deadlock issue inside `NSPanel` context — HTTP is the proven transport in this codebase
- The model is actually 1.6GB total (I'd been citing 960MB which is just the weights, not the speech tokenizer)
- The silence threshold in live_demo.py is 0.02, not 0.05 — I used the wrong value from benchmark code in my first drafts
- ivi already has a sophisticated push-to-talk system (Control key hold with 150ms commit threshold, cancellation on key combos, AVAudioEngine lifecycle management)

## For Next Time

- The ivi session is actively building the TTS sidecar. If they need Holler-side help (model questions, inference issues), this session has full context.
- The Python TTS sidecar script hasn't been written yet — it needs to be created (either in holler or ivi's sidecar directory).
- The `live-mode.md` doc in ivi is the source of truth now — it was modified by the ivi session with corrections. Don't reference the original version I wrote.
- Chris wants "call mode" (double-click → always-listening) but the ivi session scoped step 1 as voice-triggered spoken responses first. Full call mode with VAD comes later.

## Files Changed

- `~/Desktop/Files/AI/ivi/docs/live-mode.md`: Created (then modified by ivi session). TTS integration implementation plan.
