# Session Log: 2026-04-24 — Quantization Experiment + Audio Enhancement Pipeline

**Project:** Holler
**Session ID:** `f22bdb1d-7165-4c8f-941d-f0b8df8cb5cb`
**What happened:** Tested all MLX quantization methods (6-bit affine wins), then built and deployed a 3-stage audio enhancement pipeline (ClearVoice→DeepFilterNet3→Recipe E) across all 770 training clips.

---

## What I Did

### Part 1: Quantization (first half of session)

Researched all MLX quant methods via two parallel agents. Quantized 8 variants from Katie v6 bf16. Benchmarked first through mlx-audio's generate (Chris caught the misleading RTF), then properly through our server. Added `--checkpoint`/`--port` CLI flags to server.py.

Results: 6-bit affine g64 wins on voice quality (noticeably more vocal presence than 4-bit). 3-bit broken (EOS destroyed). mxfp8 surprisingly bad. Release plan: bf16 + 6-bit only.

### Part 2: Training Data Quality (middle)

Built automated quality filter for Katie's 385 clips — iterated thresholds, settled on 600ms+ silence gaps as the flag. 16 clips flagged (4%), but Chris listened and noted they were all okay — prosody issues can't be caught programmatically, only by ear. Estimated ~30% of clips need manual review.

### Part 3: Audio Enhancement Pipeline (second half)

Researched enhancement options (DeepFilterNet, AudioSR, AP-BWE, Resemble Enhance, VoiceFixer, ClearerVoice-Studio). Most failed to install (Python version issues, corrupt checkpoints, missing deps).

Built iterative A/B comparisons:
1. DSP-only (high-pass, normalize, trim, de-essing) — de-essing smoothed the top too much
2. Recipe E (high-pass + 30% spectral denoise) — Chris noticed less ear pain on AirPods. Spectral analysis confirmed 2-6kHz noise removal.
3. Created holler's own .venv (Python 3.13) to get DeepFilterNet3 working (torchaudio 2.6 needed)
4. DeepFilter→E won the 5-way shootout
5. Got ClearVoice/MossFormer2 working — Alibaba's 50M-param speech enhancer using spectral masking (safe, no vocoder)
6. ClearVoice→DeepFilter→E won the 7-way shootout. Chris: "light years better than Cartesia"
7. Fixed ClearVoice 48kHz→24kHz resampling bug (clips were playing at half speed)
8. Fixed volume matching (peak-matching to prevent quieter output)
9. Ran pipeline on all 770 clips (Katie 385 + Joe 385) in ~8 minutes. Originals backed up.

### Key technical insight: MossFormer2 uses spectral masking, not vocoding

This is why it's safe for our use case. It predicts a frequency-domain mask and multiplies it with the STFT — the original voice signal passes through. Unlike Resemble Enhance or VoiceFixer which run audio through a neural vocoder and change the voice identity. Confirmed by reading the ClearVoice source code.

## What I Learned

- Chris has ear pain from AirPods listening to TTS — the 2-6kHz noise floor is the cause. This is a real quality signal, not subjective preference.
- Chris wants flat folders with descriptive filenames for A/B listening, not nested directories.
- Chris pushes back on premature conclusions — was right that I dismissed 8-bit too hastily without proper testing.
- "Enhancement" for training data means two things: (1) signal processing pipeline, and (2) manual curation by ear. Only the first is automatable. Chris explicitly called this out.
- Always use our inference server for benchmarks — Chris caught the misleading RTF numbers immediately.
- Chris prefers to hear the actual training data (what the model will see), not idealized versions.

## For Next Time

- Manual curation is the blocker: Chris needs to listen to all 770 clips and remove ~30% with bad prosody. Can't be automated.
- After curation: retrain Katie and Joe on enhanced+curated data, quantize to 6-bit, compare with current model.
- The enhancement pipeline is fully documented in memory and CLAUDE.md. Code is in the memory file. Dependencies are in holler's .venv.
- holler's .venv doesn't have mlx-audio yet — currently inference still uses ivi's old venv. Should add mlx-audio to holler's venv.
- The earlier session log (2026-04-24-quantization-experiment.md) covers only the quant half. This log covers the full session. Both committed.

## Files Changed

- `CLAUDE.md`: Updated intro (6-bit specs), current state, structure (audio-original dirs, venv), quantization section (full experiment table), training data enhancement section (new), hard-won lessons (quant + enhancement findings), venv docs, todos
- `inference/server.py`: `--checkpoint`/`-c` and `--port`/`-p` flags, default to 6-bit, dynamic health endpoint
- `inference/experiment_quant_methods.py`: New — quantize + benchmark all variants
- `voices/katie/training-data/audio/`: All 385 clips enhanced (ClearVoice→DeepFilter→E)
- `voices/katie/training-data/audio-original/`: Backup of pre-enhancement clips
- `voices/joe/training-data/audio/`: All 385 clips enhanced
- `voices/joe/training-data/audio-original/`: Backup of pre-enhancement clips
- `.venv/`: Created holler's own venv (Python 3.13, torch 2.6, clearvoice, deepfilternet)
- Memory: `project_audio_enhancement.md` (pipeline recipe), `feedback_use_our_inference.md`, `reference_ivi_venv.md` (updated to reflect own venv)
