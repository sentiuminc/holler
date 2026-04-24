# Session Log: 2026-04-24 — Quantization Method Experiment

**Project:** Holler
**Session ID:** `f22bdb1d-7165-4c8f-941d-f0b8df8cb5cb`
**What happened:** Tested all MLX quantization methods on Katie v6. 6-bit affine g64 wins on voice quality — becomes the new production pick, replacing 4-bit.

---

## What I Did

### Research phase
Spawned two agents in parallel: a researcher to survey all MLX quantization methods (affine, mxfp4, mxfp8, nvfp4, mixed precision recipes, newer tools like mlx-optiq/JANG), and an explorer to dig through mlx-audio's converter source code for exact parameter support. Key finding: nobody has ever benchmarked quantization methods on a TTS/audio model. All prior work is LLM-focused.

### Quantization
Built `inference/experiment_quant_methods.py` — a two-phase script that quantizes all variants from bf16, then benchmarks each. Quantized 8 variants from Katie v6 bf16:
- affine 4-bit g64 (baseline), g32 (finer groups), g128 (coarser groups)
- affine 3-bit g64, 6-bit g64, 8-bit g64
- mxfp4, nvfp4, mxfp8

### First benchmark (mlx-audio's generate)
Initial benchmarks used mlx-audio's built-in `model.generate(stream=True)`. Chris immediately caught that RTF showed 0.67x instead of the expected 0.34x — because mlx-audio's loop thrashes Metal with per-chunk `mx.eval()` + `mx.clear_cache()`. This was a mistake — should have used our server from the start. Saved as a feedback memory for future sessions.

### Server improvements
Added `--checkpoint` / `-c` and `--port` / `-p` CLI flags to `server.py` (was hardcoded before). Updated health endpoint to report actual model name dynamically. This enabled easy A/B testing across checkpoints.

### Re-benchmark through our server
Re-benchmarked 6-bit, 8-bit, and 3-bit through `server.py`:
- 6-bit: RTF 0.38, TTFA 139ms, clean EOS — **the pick**
- 8-bit: RTF 0.40, TTFA 131ms, clean but bigger, no quality advantage
- 4-bit: RTF 0.38, TTFA 127ms, clean but less vocal presence
- 3-bit: BROKEN — EOS destroyed, generates 4-25s for short sentences, all clipped to 1.0

### Chris listened to samples
Generated ~30 samples across variants for A/B listening. Chris immediately noticed 6-bit has significantly better vocal presence and dynamics than 4-bit. Generated 15 more 6-bit samples to confirm — all clean.

### Release decision
Chris decided: bf16 + 6-bit affine only for HuggingFace. No 4-bit (worse quality for marginal size savings). No 8-bit (bigger than 6-bit with no benefit).

## What I Learned

- Chris wants samples in flat folders with descriptive names, not nested directories — makes A/B listening much easier.
- Always use our inference server for benchmarks, never mlx-audio's built-in generate. The RTF difference is 2x and gives misleading numbers. Chris caught this immediately.
- Chris doesn't hesitate to cut options. When 6-bit is clearly better, there's no reason to ship 4-bit "just in case." Ship the best thing.
- Chris pushed back on my hasty dismissal of 8-bit — was right that we should test it properly before concluding. The old "8-bit sounds worse than 4-bit" was from a different quant method (and was never re-tested through our server).

## For Next Time

- The quantization question is settled: 6-bit affine g64 is the pick. Don't revisit unless the model architecture changes.
- Server now accepts `--checkpoint` flag — use this for all future testing instead of editing source.
- Next priorities: retrain Katie with better prosody data, solve multi-voice quality, design more voices, HuggingFace release.
- Novel finding worth highlighting in the HF release: 6-bit affine is the sweet spot for codec-based TTS quantization. No prior benchmarks exist for this.

## Files Changed

- `inference/server.py`: Added `--checkpoint`/`-c` and `--port`/`-p` CLI flags, updated default to 6-bit checkpoint, dynamic health endpoint model name
- `inference/experiment_quant_methods.py`: New file — quantize + benchmark all quant variants from bf16
- `CLAUDE.md`: Updated current state (6-bit pick), structure (new files/dirs), quantization section (full experiment results table), performance numbers, hard-won lessons, what's next, HuggingFace release target
- Memory: `feedback_use_our_inference.md` — always use server.py for benchmarks
