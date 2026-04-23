# Session Log: 2026-04-21 (evening) — Multi-Voice Training: Katie + Joe (v7)

**Project:** ivi
**Session ID:** `6d361636-a1ab-4c0e-a5cb-de570b285eb3` (continuation of morning v6 session; same session ID, separate log for clarity since the work is a distinct milestone)
**What happened:** Extended single-voice fine-tuning to joint multi-voice. Trained Katie + Joe into one checkpoint with distinct slot assignments (Katie=3000, Joe=3001). Pipeline works end-to-end. Quality regressions discovered (Katie noise, Joe edge-clipping) that need next-session investigation.

---

## What I Did

**Audited slot architecture in v6 checkpoint.** Inspected `codec_embedding.weight` (3072 × 1024). Mapped: slots 0–2047 are active speech codec tokens (can't reuse), 2048–2199 are special tokens (BOS/EOS/lang), 2500–2999 near-zero reserved, 3000–3071 is the 72-slot custom-voice region (upstream convention). Katie's slot 3000 had norm 10.07, confirming it's intentionally high as an injected speaker embedding. This killed the "why not use slot 100?" idea — slot 100 is an active codec token; overwriting it would corrupt audio generation.

**Renamed ivi_female → katie** in `~/Downloads/ivi-v6-epoch1/config.json` (spk_id + spk_is_dialect maps). Updated `sft_12hz_patched.py` default `--speaker_name`. No retraining. Added "fantasy name convention" header comment.

**Designed Joe locally using VoiceDesign-8bit.** Chris wanted Joe as "American male Jarvis-like assistant". 4 rounds of candidate generation via `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` with different instruct strings. Round 1 (joe01–joe08, tone exploration) picked joe08 "modern assistant". Round 2 (joe08b–joe08g) tried "faster/more animated" — too much. Round 3 (joe08h–joe08k) went subtle — picked **joe08h_subtle_nudge**. Round 4 (reruns + siblings) confirmed 08h as the pick.

**Cloned Joe's 385-clip training dataset locally.** Ran Katie's proven v1 (160 texts) + v2 (225 texts) recipes via `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` locally on Mac with `joe08h_subtle_nudge.wav` as reference. 34.5 min of audio, zero failures. Dataset structure matches Katie's exactly. Output: `~/Downloads/joe-training-data/`.

**GPU odyssey (3 instances).** Original Kansas instance 35364783 ($0.24/hr) stayed in "state change queued" for 15+ min — host was full. Rented Cali A100 PCIe 35377045 ($0.47/hr) — took 8+ min to even start SSH, killed it. Rented **Croatia A100 SXM4 35377474** ($0.60/hr, 99.7% reliability) — SSH up in 20s. That's the winner.

**Set up Croatia step-by-step (runbook, per Chris's preference).** Abandoned my initial bundled `remote_setup.sh` and instead did each action as a separate SSH call: install qwen-tts + flash-attn 2.8.3, clone Qwen3-TTS, download 0.6B-Base (2.4GB) + Tokenizer-12Hz (651MB), rsync Katie + Joe datasets (770 clips, ~240MB).

**Wrote `sft_12hz_multivoice.py` — the multi-voice SFT script.** Three surgical changes vs `sft_12hz_patched.py`:
1. `VoiceAwareDataset` subclass injects `voice_name` from JSONL into each sample and carries it through collate as a `voice_names` list.
2. `target_speaker_embeddings` is now a dict keyed by voice_name. At checkpoint save, each voice's cached embedding is written to its own slot, and `spk_id`/`spk_is_dialect` config maps are populated for all voices.
3. **Ref-mel padding fix** — critical bug. Upstream `TTSDataset.collate_fn` does `torch.cat(ref_mels, dim=0)` assuming identical shape across samples. Breaks with multi-voice: Katie's 10s ref vs Joe's 7s ref → different T → RuntimeError. Override pads each ref_mel to batch-max T. First attempt padded dim=-1 (n_mels=128, always identical across all refs, so nothing happened). Correct dim is 1 — ref_mel shape is `[1, T, 128]` after upstream `.transpose(1,2)`.

**Wrote `build_combined_jsonl.py`** to merge per-voice JSONLs into one, rewriting paths to absolute on-instance paths and adding `voice_name` field per sample.

**Tokenized + trained.** Ran upstream `prepare_data.py` (which preserves unknown fields like `voice_name`), then kicked off `sft_12hz_multivoice.py` with `--voice_slot_map_json '{"katie": 3000, "joe": 3001}'`. Completed both epochs in **~3 minutes** on A100 SXM4. Loss hovered 12.5–15 throughout (correct for this LR per v6 lessons — low loss would mean broken EOS). Both voices cached from the first batch.

**Ran PyTorch inference on GPU** (10 test clips per voice across 10 new English texts). 20 clips total, 0 EOS breakages. Downloaded via scp (slow — eventually switched to rsync).

**Ran mlx-audio local inference** on Mac with the downloaded v7 checkpoint. Confirmed both voices load and generate fluent speech. Discovered that PyTorch-default-temperature (0.9) produces significantly more clipping than mlx at temp=0.6 — temperature is a major knob at inference time.

**Diagnosed audio quality issues.** Compared peak distributions and hard-clip counts:
- mlx-local: Katie 0/10 hard clips, Joe 1/10 (test_05, 10 clipped samples, peak 1.000)
- PyTorch remote: Katie 1/10, Joe 4/10 (higher temperature)

Chris heard additional noise on Katie that wasn't present in v6 standalone. Katie's slot 3000 embedding norm is bit-identical between v6 and v7 (10.0720), so the embedding itself didn't change. **Root cause is unknown.** One possibility is shared decoder weight shift from joint training; another is that these specific test texts just surface artifacts that were always there. The v6-on-same-texts comparison test was never run, so this is unresolved.

Joe's slot 3001 embedding has norm 10.9685 (+9% vs Katie). Whether this norm difference contributes to clipping is unknown — it could also just be what a deeper male voice sounds like at these codec resolutions. Chris reported audible clipping on multiple Joe clips, though the >= 0.999 peak threshold only flagged 1/10. The hard-clip metric understates the perceptual issue.

**Paused Vast instance 35377474** at session end. Instance has since been deleted. All checkpoints, samples, and logs were downloaded to `~/Downloads/` before deletion. To retrain, rent a new instance and follow the runbook in `audio-test/qwen3-tts-finetune/README.md` (steps 1-11).

**Updated docs/memory**: README multi-voice addendum, TODO section rewrite, memory file updated with v7 state.

## What I Learned

**"Trivial" is a fix, not an explanation — don't confuse the two.** Chris called me on it when I said the clipping fix was trivial and he asked "really trivial, why is this happening now?" The FIX is trivial (clip guard), but the CAUSE required actual investigation. Separate those two things when communicating.

**Joint training may have affected Katie's output, but this is unverified.** Katie's embedding is bit-identical between v6 and v7, yet v7 Katie sounded noisier. One possible explanation is shared decoder weight shift from Joe's gradients, but the same test texts were never run through v6 Katie, so it could simply be text-dependent. Future sessions should evaluate every voice after joint training on the same test inputs, not just the new one.

**The v6 "EOS is the diagnostic" lesson generalized.** At v7 I added a duration heuristic (clip longer than 3× expected → suspect EOS broken) as a sanity check. All 20 clips passed, giving me confidence the model was actually working before diving into quality analysis.

**Temperature may matter, but the comparison was uncontrolled.** PyTorch remote at temp=0.9 clipped more than MLX local at temp=0.6, but these are different inference engines, different hardware, and a single run each — not a clean A/B. Lower temperature likely helps, but we don't have real evidence for how much.

**Don't assume tensor shapes.** My first pad fix operated on dim=-1 because I assumed ref_mel was `[1, n_mels, T]`. It's actually `[1, T, n_mels]` after an upstream `.transpose(1,2)`. Ten seconds of diagnostic printing would have saved one debug cycle. When seeing tensor shape errors, print first, assume second.

**Runbook > one-shot script for observability.** I started with a bundled `remote_setup.sh` that did install + clone + download in one go. Chris pushed back ("I would rather have a runbook than a script executing this to be honest"). He was right — when the setup failed partway, I had no idea how far it got. Per-step SSH calls would have given clear checkpointing. I pivoted to runbook style and it worked much better for debugging.

**Vast proxies can be very slow.** Our first scp attempt ran at ~150 KB/s through the ssh7.vast.ai proxy. Rsync with -z compression flag helped (6 MB/s). For future large-data moves, consider HuggingFace Hub as a transit layer (faster CDN) — Chris said don't for this session but it's worth remembering.

**Monitor tool became counterproductive.** I was using Monitor tasks for long-running polls. Chris told me to stop — they were noisy and redundant with regular Bash run_in_background. Switched to plain background Bash tasks for everything from that point on. The notifications from Bash completions are enough; don't layer Monitor watchers on top.

## For Next Time

**Immediate investigation (one-shot test):**
Run v6 single-voice Katie on the same 10 test texts used in v7 inference. Compare peak distributions and Chris's subjective noise perception. If v6 Katie is also noisy on these texts, the "joint training regression" hypothesis is wrong — it's just these specific texts surfacing artifacts that were always there. If v6 Katie is clean, joint training is the cause and we need to mitigate it.

**Mitigation strategies to try:**
1. **Inference-time clip guard** in `remote_inference_multivoice.py` and `test_mlx_multivoice.py`: `audio = audio / max(1.0, np.abs(audio).max() / 0.9)` before save. Also set `temperature=0.6` as default everywhere.
2. **Renormalize slot 3001 to 10.07** in v7 safetensors — evens loudness without retraining. Unknown if voice identity preserves (direction preserved, magnitude scaled) — test first.
3. **Speaker-embedding normalization at training time** — force unit norm (or fixed target norm) before position-6 injection in the training loop. Retrain v7 → v7.1.

**Deferred (previously queued):**
- 8-bit quantization: `mlx_lm.convert -q --q-bits 8 --hf-path ~/Downloads/ivi-v7-katie-joe/checkpoint-epoch-1`
- Streaming TTFA verification target: ~77ms
- Sidecar integration: wire fine-tuned MLX multi-voice into ivi sidecar TTS path

**Future: expand voice pack.** Pipeline is now proven. Next time we want a new voice: design via VoiceDesign model → clone 385 clips locally → add to training JSONL with new `voice_name` → retrain. Target `mlx-community/Qwen3-TTS-12Hz-0.6B-IviVoices-8bit` once quality is dialed in.

**Instance management:**
- Both Vast instances (35377474 Croatia, 35364783 Kansas) have been deleted. All outputs were downloaded locally first.
- To retrain: rent a fresh instance and follow the runbook in `audio-test/qwen3-tts-finetune/README.md`. Training data is local at `~/Downloads/ivi-tts-training-data/` (Katie) and `~/Downloads/joe-training-data/` (Joe).

**Behavioral notes:**
- When Chris asks "why", give the actual diagnosis, not just the fix. "Trivial fix" is not an answer to "why".
- Don't use Monitor tool for polling — plain background Bash is enough.
- Runbook style > bundled script for anything where partial failure is meaningful.
- Save everything to `~/Downloads/<subfolder>/`, never afplay inline.
- Stop instances, don't destroy.

## Files Changed

- `audio-test/qwen3-tts-finetune/README.md` — added v7 Multi-Voice Addendum section with code changes, training run details, known issues, hypotheses, and asset paths
- `TODO.md` — rewrote Voice (TTS) section to cover v6 + v7 state, next-session investigation items, slot architecture, instance tracking
- `audio-test/qwen3-tts-finetune/sft_12hz_multivoice.py` — NEW, multi-voice SFT script with VoiceAwareDataset + dict-of-embeddings + ref_mel pad fix
- `audio-test/qwen3-tts-finetune/build_combined_jsonl.py` — NEW, merges per-voice JSONLs for multi-voice training
- `audio-test/qwen3-tts-finetune/remote_inference_multivoice.py` — NEW, PyTorch inference for multi-voice with EOS-broken heuristic flagging
- `audio-test/qwen3-tts-finetune/test_mlx_multivoice.py` — NEW, local mlx-audio inference for multi-voice (peak + clip logging)
- `audio-test/qwen3-tts-finetune/transcribe_and_verify.py` — NEW, local whisper transcription + WER vs expected (not yet used; queued next-session)
- `audio-test/qwen3-tts-finetune/design_joe_candidates.py` — NEW, initial Joe voice design (8 candidates)
- `audio-test/qwen3-tts-finetune/design_joe_variants.py` — NEW, faster/more-animated variants (rejected)
- `audio-test/qwen3-tts-finetune/design_joe_subtle.py` — NEW, subtle-nudge variants (winner: joe08h)
- `audio-test/qwen3-tts-finetune/design_joe_from_8h.py` — NEW, expansions around joe08h (reruns + siblings)
- `audio-test/generate_joe_training_data.py` — NEW, Joe v1 training data generator (160 clips)
- `audio-test/generate_joe_training_data_batch2.py` — NEW, Joe v2 training data generator (225 clips)
- `audio-test/qwen3-tts-finetune/remote_setup.sh` — NEW, instance setup script (used partially before pivoting to runbook style)
- `audio-test/qwen3-tts-finetune/remote_train.sh` — NEW, instance training orchestration
- `audio-test/qwen3-tts-finetune/orchestrate_to_sleep.sh` — NEW, autonomous loop orchestrator (inference → download → transcribe → stop)
- `audio-test/qwen3-tts-finetune/sft_12hz_patched.py` — minor: default `--speaker_name` changed from `speaker_test` to `katie`, added fantasy-name convention header comment

Memory updates (not in git):
- `~/.claude/.../memory/project_qwen3_tts_exploration.md` — added v7 multi-voice section with checkpoint paths, embedding norms, slot architecture map, known quality issues, and code-fix notes
- `~/.claude/.../memory/MEMORY.md` — description updated to reflect multi-voice state

Also made (not in this repo, saved to `~/Downloads/`):
- `~/Downloads/joe-candidates/` — 26 voice-design candidates + index.txt with all prompts
- `~/Downloads/joe-training-data/` — 385 cloned clips (reusable)
- `~/Downloads/ivi-v7-katie-joe/checkpoint-epoch-1/` — the multi-voice checkpoint (2.3GB bf16)
- `~/Downloads/ivi-v7-katie-joe/samples/{katie,joe}/` — 20 PyTorch remote inference samples
- `~/Downloads/ivi-v7-katie-joe/samples-mlx/{katie,joe}/` — 20 mlx-audio local inference samples
- `~/Downloads/ivi-v7-katie-joe/{train.log,inference.log,mlx-inference.log,orchestrator.log}` — run logs

Rename of existing:
- `~/Downloads/ivi-v6-epoch1/config.json` — `ivi_female` → `katie` in spk_id and spk_is_dialect
