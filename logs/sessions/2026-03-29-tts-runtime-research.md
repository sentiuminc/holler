# Session Log: 2026-03-29 — TTS Runtime Research & Benchmarking

**Project:** ivi (macOS notch AI assistant)
**What happened:** Investigated TTS runtime options for Kokoro-82M, ruled out FluidAudio CoreML, chose kokoro-ios (MLX Swift), benchmarked performance, finalized stitching config, wrote full implementation spec.

---

## What I Did

### FluidAudio CoreML Investigation
- Generated A/B test samples: FluidAudio CoreML vs Python Kokoro (gold standard)
- Chris identified quality difference: "squeezed", "shaky" sound from FluidAudio
- Investigated systematically, ruling out each hypothesis:
  - G2P phonemes: **identical** (FluidAudio's 178k lexicon matches espeak output)
  - Float16 precision: **not the cause** (5s model is Float32, confirmed via model metadata)
  - De-esser: **no difference** (tested with `--no-deess`)
  - 15s model: **worse** (Mixed Float16/Float32, confirmed more wobbly)
- Root cause found: **fixed-shape CoreML conversion**. Model pads 18 real tokens to 124 positions. BERT processes padding alongside real tokens, producing subtly different audio. Fundamental CoreML limitation.
- Also discovered why FluidAudio removed espeak-ng: GPL-3.0 licensing incompatible with Apache 2.0. They replaced with CoreML BART G2P (PR #350), documented its inferiority (PR #414).

### kokoro-ios Evaluation
- Cloned and built kokoro-ios (MLX Swift, MisakiSwift G2P)
- Overcame build issues: SPM can't compile Metal shaders, needed xcodebuild + Metal Toolchain download (700MB one-time). Also needed static linking patches for KokoroSwift and MisakiSwift to avoid duplicate MLX symbols.
- Benchmarked on M1 Pro: 95-400ms per sentence, matching Python quality
- Confirmed dynamic-length input (no padding) → identical audio quality to Python

### Stitching Optimization
- Iterated through 8 versions of silence trimming and crossfade config
- Started with RMS-based windowed approach (too complex, cut into speech)
- Found last night's session code via subagent — simple amplitude threshold was the winner
- Final config: amplitude 0.01, 200ms tail, 10ms lead-in, 20ms crossfade
- A/B tested against Kokoro's full-text generation (strategy C) — virtually identical pacing

### Resource Profiling
- Measured CPU, memory, generation time across varied workloads
- CPU: ~60% avg during gen (~8% of machine), GPU does real work via Metal
- Memory: ~350MB loaded, ~200MB after clearCache (Metal framework overhead)
- Tested load/unload lifecycle 15 cycles — stable, no growth
- Found `Memory.clearCache()` is required to free MLX buffer pool
- Remaining ~200MB is Metal heap + compiled shaders (process-lifetime, confirmed via Apple Dev Forums)

### Edge Case Testing
- Tested short first sentence safety: "Done." + 30-word sentence = 510ms gap
- Solution: batch short first sentence with second if already available
- Generated 10 varied response scenarios (weather, debug, casual, music, etc.)

## What I Learned

- Chris has a good ear for audio quality differences — identified the CoreML degradation immediately as "squeezed/shaky"
- Chris pushes back to verify reasoning, not to disagree — held firm on memory leak investigation, was right to demand proper analysis
- Don't dismiss resource issues as "doesn't affect us" — Chris correctly called out the 200MB retention as something that needs understanding, even if it turns out to be unfixable
- Speed settings differ per voice: sky's "normal" is 1.0, heart/jessica "normal" is 1.1
- The research agent hallucinated that FluidAudio had re-added espeak-ng — always verify agent claims against actual code/git history
- `swift build` cannot compile Metal shaders (documented MLX Swift limitation) — only `xcodebuild` works. Non-issue for ivi since it already uses Xcode.

## For Next Time

- Full implementation spec is at `docs/voice-tts-research.md` — read this before starting implementation
- Memory files updated: `project_voice_tts.md` and `project_voice_architecture.md`
- Test artifacts in `audio-test/kokoro-test/` (benchmark CLI) and `~/Downloads/kokoro-comparison/` (all audio samples)
- kokoro-ios cloned at `audio-test/kokoro-ios/` with Package.swift patched to `.static`
- Build order starts with adding SPM dep to project.yml, then TTSManager, AudioPlayer, SentenceBuffer
- Don't read TODO.md (Chris's instruction)

## Files Changed
- `docs/voice-tts-research.md`: NEW — complete TTS implementation spec (engine decision, benchmarks, stitching config, memory management, integration plan)
- `memory/project_voice_tts.md`: Updated — kokoro-ios chosen, full details
- `memory/project_voice_architecture.md`: Updated — STT + TTS pipeline architecture
- `memory/MEMORY.md`: Updated index entries for voice files
- `audio-test/kokoro-ios/`: Cloned repo with static linking patch
- `audio-test/kokoro-test/`: Test CLI package for benchmarking
