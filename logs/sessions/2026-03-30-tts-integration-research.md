# Session Log: 2026-03-30 — TTS Integration Research & Voice Prompt Optimization Prep

**Project:** ivi (macOS notch AI assistant)
**What happened:** Continued from TTS runtime research session. Investigated GPU contention, threading, and AVAudioEngine. Replaced animated glow with static drop shadow. Shelved Kokoro, discovered Inworld TTS as viable cloud option. A/B tested both engines. Prepared voice prompt optimization methodology.

---

## What I Did

### GPU Contention Discovery
Researched from primary sources (WWDC sessions, Swift source code, Apple Silicon architecture docs) whether SwiftUI rendering and MLX Metal compute contend for GPU resources. **Confirmed they share the same GPU ALUs.** The animated glow effect (triple `.shadow()` at 60fps via `TimelineView(.animation)`) was causing 4-5 GPU render passes per frame, competing with MLX inference. Replaced with a static black drop shadow — zero GPU cost. Old glow preserved as comments.

### Swift Threading Verification
Verified from Swift runtime source code that `Task.detached(priority: .userInitiated)` maps directly to `QOS_CLASS_USER_INITIATED` (0x19), which prefers P-cores on Apple Silicon. `@MainActor` runs at `QOS_CLASS_USER_INTERACTIVE` (0x21) — higher priority. TTS generation must be on detached tasks, never main actor.

### AVAudioEngine Research
Confirmed from Apple docs: `scheduleBuffer()` is non-blocking, completion handlers fire on a private internal queue, safe to call from any thread. Never call `stop()` from inside a completion handler (deadlocks). Gapless streaming: pre-schedule 2-3 buffers, refill in completion handlers.

### Shelved Kokoro TTS
Chris decided Kokoro-82M quality wasn't sufficient for a conversational assistant. Removed kokoro-ios and MisakiSwift dependencies from project.yml. Updated docs/voice-tts-research.md with all new findings (GPU contention, threading, AVAudioEngine, sentence detection, text preprocessing, 10 untested items).

### Discovered Inworld TTS
Ran A/B comparison between Kokoro (Jessica voice, on-device) and Inworld TTS 1.5 (Ashley voice, cloud). Both sound good. Generated 8 audio samples saved to `~/Downloads/inworld-tts-samples/`. Key finding: Kokoro sounds fantastic as full text (no sentence chunking), competitive with Inworld. Best Inworld settings: temp 1.5, speed 1.5, Mini model.

### Inworld API Deep Research
Researcher agent documented all Inworld TTS knobs: 65+ voices, temperature (0-2), speaking_rate (0.5-1.5), emotion tags, SSML break tags, voice cloning from 5-15s audio, voice design from text description. No pitch control.

### Voice Prompt Optimization Methodology
Researched best practices for voice agent prompting (LiveKit, LMNT, Inworld, Vapi, Murf, Sierra AI, PolyAI). Key insight: the bottleneck is how the LLM writes, not the TTS engine. Created `docs/voice-prompt-optimization.md` with full autoresearch-style methodology for the next session.

### Key Insight: Full Text vs Sentence Chunking
Kokoro sounds dramatically better generating full paragraphs vs sentence-by-sentence. The prosody flows naturally across sentences. This changes the streaming strategy — batch more text before generating, accept slightly higher TTFA for much better quality.

## What I Learned

- Chris shelves decisively. When quality doesn't meet the bar, no amount of engineering saves it. The right move is to preserve research and move on.
- "The prompt is the product" — for voice, the TTS engine matters less than what words the LLM generates. Voice-native prompting is the actual unlock.
- Chris wants one output mode that works for both voice and text, not two separate modes. If it sounds good spoken, it reads well in the notch too.
- Inworld Mini at $5/1M chars is cheap enough to not worry about (~$4-9/month for heavy use).
- The autoresearch methodology (fixed eval set, clear metric, iterate) applies well to prompt optimization.

## For Next Time

- **Next session:** Optimize ivi's system prompt for voice-native output using `docs/voice-prompt-optimization.md` as the brief
- Pickup prompt is in the session summary above
- The test harness: pull questions from metrics DB, send via curl, evaluate responses
- The system prompt is at `sidecar/server.ts:131` in `buildSystemPrompt()`
- Don't read TODO.md

## Files Changed
- `ivi-app/NotchKit/NotchGlow.swift`: Replaced animated glow with static black drop shadow
- `project.yml`: Removed kokoro-ios and MisakiSwift dependencies
- `docs/voice-tts-research.md`: Major update — GPU contention, threading, AVAudioEngine, sentence detection, untested items, revised build order
- `docs/voice-prompt-optimization.md`: NEW — full brief for prompt optimization session
- `audio-test/inworld-tts-test.js`: NEW — Inworld TTS WebSocket test script
- `audio-test/kokoro-test/Sources/KokoroTest/main.swift`: Modified for voice preview testing (full text gen, Jessica voice)
- Memory files updated: project_voice_tts.md, project_voice_architecture.md, project_notch_glow.md, project_inworld_tts.md (new)
