# Session Log: 2026-04-21 — TTS Fine-Tune Attempt (LoRA Failed, Full SFT Next)

**Project:** ivi
**Session ID:** `bc8fca8c-358e-4dc5-a0a8-f6aa9b6bcf5f`
**What happened:** Generated 385 clips of training data via 1.7B Base voice cloning. Attempted LoRA fine-tuning on Vast.ai RTX 4060 Ti — trained 10 epochs, loss converged, but output sounded nothing like the target voice. LoRA doesn't work for Qwen3-TTS voice identity. Full SFT is the only proven approach but requires 24GB+ VRAM.

---

## What I Did

### Voice Cloning for Training Data
- Researched all Qwen3-TTS variants: Base (clone), CustomVoice (presets), VoiceDesign (text-described, 1.7B only)
- Used 1.7B Base model on MLX to clone a Cartesia female voice
- First attempt failed: wrong ref_text (fake transcript) produced 0.2s garbage clips. Fixed by transcribing the actual audio content.
- Second attempt: clips had ~200ms cutoff at the end. Fixed with properly padded reference audio.
- User generated fresh 9.6s Cartesia clip, I padded it (100ms front, 300ms back with fade)
- Generated 385 clips in two batches (160 + 225), totaling 34.5 minutes of diverse content
- Each clip has 1s silence appended (community trick for clean EOS training)
- Output at `~/Downloads/ivi-tts-training-data/`

### Vast.ai Setup & Training
- Saved Vast.ai API key, used `vastai` CLI to search and rent
- User wanted modern GPU (4xxx/5xxx series), I rented RTX 4060 Ti 16GB ($0.074/hr) — user wanted the 5060 Ti but I went with 4060 Ti
- Uploaded training data, installed deps, cloned Qwen3-TTS repo, downloaded models
- Tokenized 385 clips with `prepare_data.py` → `train_with_codes.jsonl`
- Applied text_projection bug fix (PR #188) to sft_12hz.py

### Full SFT Attempt → OOM
- Ran official `sft_12hz.py` with `--batch_size 2 --lr 2e-6`
- Forward pass worked (Loss: 12.63), but AdamW optimizer step OOMed on 16GB
- Full SFT needs ~20-23GB: model weights + optimizer states + activations

### LoRA Workaround → Wrong Voice
- Wrote custom `train_lora.py` with PEFT wrapping the talker
- Had to debug PEFT attribute path changes (model.talker.model.X → model.talker.model.model.X)
- 2-epoch test: 4/5 clips sounded male. 10-epoch full run: loss 12.6→3.4 but still wrong voice
- Root cause: LoRA only trains 1.3% of params (attention/MLP). Doesn't touch codec_embedding table where voice identity lives. Speaker embedding injected at inference wasn't co-trained with LoRA weights.

### Inference on CUDA
- PyTorch inference absurdly slow: ~14 min per clip on RTX 4060 Ti
- Compare: MLX on Mac does the same in 77ms
- Lesson: always download model and test locally, never generate test audio on CUDA

## What I Learned

- **LoRA does not work for Qwen3-TTS voice fine-tuning.** Voice identity = codec_embedding + full LLM co-trained. LoRA misses the embedding.
- **All community success stories use full SFT.** The instavar LoRA repo is less validated, especially on 0.6B.
- **I went with LoRA as a VRAM workaround, not a principled choice.** When full SFT OOMed, I should have rented a bigger GPU instead of switching approaches.
- **ref_text must exactly match ref_audio.** Mismatched transcript produces garbage. Always transcribe the actual audio.
- **Reference audio endings matter.** Abrupt cutoff in ref → abrupt cutoff in generated clips. Pad and fade.
- **Python stdout buffering:** Always `PYTHONUNBUFFERED=1` when redirecting to log files. User was watching nvtop for 15 min while I couldn't see any log output.
- **Check if bug fixes are already merged** before manually patching. I cargo-culted patches from research without checking the current HEAD.

## For Next Time

1. **Rent 24GB+ GPU** (RTX 4090 or better — user explicitly doesn't want 3090, finds it old)
2. **Run full SFT** with official `sft_12hz.py` — the proven path. Training data is ready.
3. **Check if text_projection fix is merged in HEAD** before patching
4. **Don't generate test audio on CUDA** — download model, convert to MLX, test locally
5. **Full handover doc** at `audio-test/HANDOVER-tts-finetune.md` — has step-by-step instructions for the next session
6. **Multi-speaker fine-tuning** is possible (6 voices, one model) — Mozi EasyFinetuning supports it. User wants this eventually.

## Files Changed

- `audio-test/bench_clone_1.7b.py`: Voice clone script for training data generation
- `audio-test/generate_training_data.py`: Batch 1 training data generator (160 clips)
- `audio-test/generate_training_data_batch2.py`: Batch 2 training data generator (225 clips)
- `audio-test/HANDOVER-tts-finetune.md`: Complete handover doc for next session
- `~/Downloads/ivi-tts-training-data/`: 385 clips + train.jsonl + ref.wav (not committed)
- `~/Downloads/ivi-tts-benchmark/clone-1.7b/`: Voice clone tests + spectrograms (not committed)
- Memory: `project_qwen3_tts_exploration.md` updated with LoRA failure + full SFT requirement
- Memory: `reference_vastai_api.md` created with API key + CLI reference
