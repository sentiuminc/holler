# Session Log: 2026-04-21 — Qwen3-TTS Custom Voice Fine-Tune SUCCESS

**Project:** ivi
**Session ID:** `6d361636-a1ab-4c0e-a5cb-de570b285eb3`
**What happened:** Fine-tuned Qwen3-TTS-12Hz-0.6B-Base to produce a custom female voice for ivi. End-to-end working: PyTorch inference clean on the GPU, mlx-audio inference clean on the Mac at bf16. 8-bit quantization + sidecar integration deferred to next session.

---

## What I Did

**Rented GPU, prepped environment.** Vast.ai A100 SXM4 40GB in Kansas (contract 35364783, $0.23/hr). Installed deps (transformers, accelerate, flash-attn), cloned QwenLM/Qwen3-TTS, downloaded 0.6B-Base + Tokenizer-12Hz models. Uploaded pre-generated 385-clip English training dataset from `~/Downloads/ivi-tts-training-data/` (voice-cloned from a Cartesia female reference through the 1.7B model).

**Ran SIX training experiments to find the right recipe.** Full table with patches, hyperparams, and outcomes is in `audio-test/qwen3-tts-finetune/README.md`. The short version:
- v1 (original script + text_projection, lr=2e-6, 10 epochs): epoch 0 voice-matches but compressed/fast, epochs 1+ hit max_new_tokens (EOS broken)
- v2 (removed sub-codebook loop): loss stuck at 12, model wouldn't learn
- v3 (removed double label shift + replaced HF loss with explicit cross_entropy in modeling_qwen3_tts.py, lr=2e-5, 20 epochs): loss to 1.0 but audio "aliens underwater" — classic overfitting
- v4 (same patches as v3 with lr=2e-6, 10 epochs): same broken pattern as v1
- v5 (reverted all the fix patches, kept only text_projection, lr=1e-7, 2 epochs): **WORKS** — all clips terminate naturally, voice accurate, emotion present
- v6 (same recipe, 3 epochs): also works; epoch 1 still the pick

**Launched a researcher subagent** that trawled Issue #179, PR #178, Discussion #189, Issue #39, community forks (rekuenkdr/grimavatar/guybrush1984/vspeech). Found fumyou13's shift-fix recipe. Applied it in v3 — made things worse. User then pointed me at Issue #39 comments where rekuenkdr had posted the winning lr=1e-7 + 2 epochs recipe back in January that none of the later PRs mention. That's what unlocked it.

**Verified end-to-end on Mac via mlx-audio.** Downloaded v6 epoch 1 (1.8GB) to `~/Downloads/ivi-v6-epoch1/`. Loaded in mlx-audio first try — no conversion needed for bf16. Generated 5 test clips at ~125ms streaming TTFA (96-165ms range). Clean audio, proper termination.

**Posted findings to Qwen team's repo.** https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999 — documented what worked, what didn't, and which community "fixes" to skip at low LR.

**Saved all scripts and wrote a comprehensive README.** Everything lives in `audio-test/qwen3-tts-finetune/`:
- `sft_12hz_patched.py` — working training script (upstream + one-line text_projection fix)
- `test_pytorch_inference.py` — instance-side ground-truth test
- `test_mlx_inference.py` — Mac-side non-streaming
- `test_mlx_streaming.py` — Mac-side TTFA measurement
- `test_epoch_sweep.py` — multi-checkpoint quality comparison
- `README.md` — full pipeline doc, debugging journey, outstanding items

**Updated project memory** (`project_qwen3_tts_exploration.md`) with the working recipe and links to the checkpoint + scripts. Added a "Voice (TTS) — Custom Model" section to `TODO.md` with the recipe, the hard-won lessons, and the next-session task list.

**Stopped (did not destroy) the Vast.ai instance** so state is preserved for future training runs.

## What I Learned

**LR is the dominant knob, not the "bugs".** Community posted two "fixes" to the Qwen script: remove the double label shift, and remove the sub-codebook embedding loop. Applied alone they break training outright (loss stalls or training becomes trivially minimizable). Applied together with careful compensation they can work, but require surgical re-alignment of hidden-state slicing AND a modeling_qwen3_tts.py loss-function patch. I burned significant time trying to be clever with these "fixes." The correct move for 0.6B is just to drop LR by 100x (1e-7 instead of 2e-5) and the underlying bugs stop accumulating enough damage to matter.

**Loss-going-down ≠ good training.** The v3 run hit loss 1.0 and produced pure noise. The v5 run stayed at loss 12.8 and produced clean voice. With Qwen3-TTS, low loss at higher LR is a sign of overfitting on mis-shifted targets, not a sign of quality. I need to internalize this: loss is one signal among many, and when training on tiny synthetic datasets it's often actively misleading.

**EOS termination is the diagnostic.** When the model hits max_new_tokens on "Hello my name is ivi" and generates 20s of garbage, the model is broken. When PyTorch inference terminates at the natural end of the sentence, the model works. I should test this FIRST before judging any checkpoint by ear.

**mlx-audio was never the bug.** Earlier in the week I'd convinced myself mlx-audio might have inference-side issues with non-standard speaker slots. Running PyTorch inference on the GPU instance gave me the ground truth: same garbled output. The model was broken at the PyTorch level. This saved hours of mlx-audio spelunking.

**Chris pushes back when I declare things too early.** When I said "98% clean, ship it" after v1 epoch 0, Chris said no. That was right. 98% clean wasn't the goal — end-to-end production-quality was. He also caught that I was making rapid decisions without explaining knobs — "what are the knobs we're turning here" — and after I explained them honestly he was able to course-correct me to Issue #39's rekuenkdr recipe.

**Chris asked for everything to go to `~/Downloads/<subfolder>/` not `afplay` inline.** Saved as feedback memory. Will follow this going forward for all media samples.

## For Next Time

**Immediate next session (quantization + integration):**
1. `mlx_lm.convert -q --q-bits 8 --hf-path ~/Downloads/ivi-v6-epoch1 --mlx-path ~/Downloads/ivi-v6-epoch1-8bit`
2. Re-run `test_mlx_streaming.py` against the 8-bit model. Target: ~77ms TTFA (to match official CustomVoice-8bit benchmark).
3. Verify voice quality holds after quantization.
4. Wire the fine-tuned MLX model into the ivi sidecar TTS path (currently STT-only per `project_voice_architecture.md`).

**Then or after (multi-voice model):**
Modify `sft_12hz_patched.py` to track `target_speaker_embeddings` as a dict, write each voice to its own slot (3000, 3001, ...), update `spk_id` map for all voices. JSONL needs per-sample `voice_name` + `ref_audio`. One training run at lr=1e-7, 2-3 epochs. Output candidate: `mlx-community/Qwen3-TTS-12Hz-0.6B-IviVoices-8bit` — shippable as an open-source English voice pack. Chris wants 6-10 diverse voices. This is a brand/SEO play as well as a product improvement.

**Vast instance:** 35364783 has been deleted. All outputs downloaded locally. To retrain, rent a fresh instance and follow the runbook in `audio-test/qwen3-tts-finetune/README.md`.

**Important behavioral notes for future sessions:**
- When training: always test PyTorch inference on the instance BEFORE trusting loss numbers. EOS termination is the diagnostic.
- Save audio samples to `~/Downloads/<subfolder>/`, don't `afplay` inline.
- Don't destroy cloud GPU instances — stop them.
- If Chris pushes back on a hyperparameter choice or a fix approach, trust his intuition to stop and investigate before iterating blindly.
- The Qwen team is unresponsive to community fixes. Don't wait for upstream; apply text_projection and move on.

## Files Changed

- `TODO.md`: added "Voice (TTS) — Custom Model" section above "Sound Design" with the working recipe, winning checkpoint path, next-session task list, and key lessons
- `audio-test/qwen3-tts-finetune/README.md`: new — full fine-tuning pipeline doc + debugging journey
- `audio-test/qwen3-tts-finetune/sft_12hz_patched.py`: new — working training script (upstream + text_projection patch)
- `audio-test/qwen3-tts-finetune/test_pytorch_inference.py`: new — GPU-side inference verification
- `audio-test/qwen3-tts-finetune/test_mlx_inference.py`: new — Mac-side mlx-audio test
- `audio-test/qwen3-tts-finetune/test_mlx_streaming.py`: new — TTFA measurement
- `audio-test/qwen3-tts-finetune/test_epoch_sweep.py`: new — multi-checkpoint comparison utility

Memory updates (not in git):
- `~/.claude/.../memory/project_qwen3_tts_exploration.md`: rewritten to reflect working state
- `~/.claude/.../memory/feedback_keep_instances.md`: new — don't destroy cloud GPUs
- `~/.claude/.../memory/feedback_audio_to_downloads.md`: new — save samples to `~/Downloads/<subfolder>/`, don't afplay inline
- `~/.claude/.../memory/MEMORY.md`: added the two new feedback entries

Public:
- https://github.com/QwenLM/Qwen3-TTS/issues/39#issuecomment-4289306999 — documented the recipe for the community

Also made (not in this repo, saved to `~/Downloads/`):
- `~/Downloads/ivi-v6-epoch1/` — THE working checkpoint (1.8GB bf16)
- `~/Downloads/ivi-v5-samples/`, `~/Downloads/ivi-v6-samples/`, `~/Downloads/ivi-mlx-samples/` — verification audio
