# Session Log: 2026-04-20/21 — TTS Benchmark: Qwen3-TTS Wins

**Project:** ivi
**Session ID:** `bc8fca8c-358e-4dc5-a0a8-f6aa9b6bcf5f`
**What happened:** Benchmarked 7 local TTS models head-to-head on real ivi outputs. Qwen3-TTS 0.6B CustomVoice at 8-bit quantization won decisively — 77ms TTFA with excellent audio quality. Discovered streaming_interval=0.1 trick that made it faster than everything else.

---

## What I Did

### Phase 1: Kokoro Playground
- Started by catching up on existing Kokoro setup (af_jessica, sentence-chunking, crossfade stitching)
- Pulled latest ivi outputs from metrics DB and synthesized them through Kokoro
- Built an HTML playground with audio players + free text input (Bun server on port 3456)
- Added voice selector with all 50+ Kokoro voices

### Phase 2: Research & Planning
- Researched current local TTS landscape: Soprano, Qwen3-TTS, Chatterbox (not Microsoft — it's Resemble AI), F5-TTS, Dia, Orpheus
- User provided 3 Cartesia reference audio clips (different voices/speeds) for voice cloning tests
- Created formal benchmark plan, entered plan mode, got approval

### Phase 3: Full Benchmark
- Set up Python venv with mlx-audio (`audio-test/.venv-tts-bench/`)
- Ran each model on same 6 real ivi output texts
- Measured TTFA (streaming) and total generation time
- Results: Soprano fast but one voice, Qwen3-TTS good but initially appeared slow, Chatterbox 1s+ overhead, F5-TTS broken, Dia unusable

### Phase 4: Qwen3-TTS Deep Dive
- Discovered the "Base" model has NO built-in speakers (spk_id: {}) — voice was random every run
- Switched to CustomVoice model which has fixed speaker embeddings (token IDs)
- Found only 2 English speakers: Aiden (good), Ryan (too surfer-chill)
- **Key discovery:** streaming_interval=0.1 drops TTFA from 350ms to 100ms with zero quality loss
- Tested all 9 speakers, instruct variations (doesn't actually work on 0.6B), temperature sweep

### Phase 5: Quantization
- Tested 8-bit: 77ms TTFA, sounds identical to bf16 — **confirmed winner**
- Tested 4-bit: 78ms TTFA, slightly "wonky/wobbly" quality — rejected
- User confirmed 8-bit sounds great

### Phase 6: Fine-Tuning Research
- Researched LoRA fine-tuning for custom voices extensively
- Found community experiences: fine-tuning locks voice identity but flattens expressiveness
- Found existing LoRAs (Yuna Korean female on M4, Polish, Hindi) but no English female
- Identified training recipe: LoRA r=16, LR 2e-6, 10-30 min audio, 3-5 epochs, scale 0.3-0.35
- Trainable on Apple Silicon (proven) or $2-4 cloud GPU

## What I Learned

- **Don't declare limitations prematurely.** Multiple times I said something was impossible and Chris pushed back to reveal a solution (streaming_interval, wrong model variant, etc.)
- Chris processes by rapid iteration — "just run it" not "explain the theory first"
- When Chris says "pause" or asks "what are you planning," STOP and explain before executing
- The Qwen3-TTS architecture is genuinely impressive: full Qwen2.5 LLM reasoning about prosody at 12Hz token rate, streaming decoder that works with arbitrary chunk sizes
- "Chatterbox" not "Cheddarbox" — by Resemble AI, not Microsoft. User had tested it before and knows their quality claims are overblown.

## For Next Time

1. **Fine-tune female voice** — user wants to do this "tomorrow." Need:
   - 10-30 min of consistent female voice audio at 24kHz
   - Source: likely Cartesia (generate varied text) or find existing dataset
   - Use instavar/qwen3-tts-lora-finetuning or ARahim3/mlx-tune
   - User has Vast.ai credits for cloud GPU

2. **Investigate prosody control** — user concerned about:
   - Big pauses between sentences (empty audio in streamed output)
   - Random emotional swings (happy then sad)
   - Whether we can trim silence from chunks or post-process
   - Don't accept "that's just how it is" — dig deeper

3. **0.6B instruct** — officially doesn't work, but user wants us to verify. Maybe there's a way to influence delivery beyond just temperature.

4. **Integration path** — Python MLX sidecar process (like Kokoro but Python instead of Swift). Need to design this.

## Files Changed

- `docs/voice-tts-research.md`: Updated from SHELVED to ACTIVE, added full Qwen3-TTS section with settings, benchmark results, fine-tuning path
- `audio-test/bench_soprano.py`: Soprano 80M benchmark script
- `audio-test/bench_qwen3tts.py`: Qwen3-TTS Base benchmark
- `audio-test/bench_chatterbox.py`: Chatterbox Turbo single-voice benchmark
- `audio-test/bench_chatterbox_multi.py`: Chatterbox multi-voice benchmark
- `audio-test/bench_dia.py`: Dia 1.6B benchmark (killed — too slow)
- `audio-test/bench_f5tts.py`: F5-TTS benchmark (broken)
- `audio-test/bench_qwen3_clone.py`: Qwen3-TTS voice clone benchmark
- `audio-test/kokoro-test/Sources/KokoroTest/main.swift`: Added CLI mode (--text, --voice, --output)
- `audio-test/kokoro-test/server.ts`: Bun server for Kokoro HTML playground
- `audio-test/tts-benchmark.py`: Original comprehensive benchmark script (not used in final approach)
- Audio outputs: `~/Downloads/ivi-tts-benchmark/` (organized by model, not committed)
