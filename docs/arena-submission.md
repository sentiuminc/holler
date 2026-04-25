# Arena & Leaderboard Submission Plan

**Status:** Blocked on HuggingFace release (v1 first). Once weights are up, deploy Space + submit.

**Models to submit:** bf16 (full) and affine-6bit-g64 (quantized). Both produce identical quality for arena purposes.

---

## TTS Arena V2 (Primary Target)

**URL:** https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2
**Leaderboard:** https://tts-agi-tts-arena-v2.hf.space/leaderboard
**Format:** Blind A/B voting, Elo rankings. Users hear two anonymous models, pick the better one.

**How to submit:**
1. Open a Discussion: https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2/discussions/new
2. Or email maintainer **mrfakename** — `me@mrfake.name` / X: `@realmrfakename`
3. Discord: https://discord.gg/HB8fMR6GTr

**What they need:** A HuggingFace Space with a `/synthesize` endpoint that accepts `{"text": "..."}` and returns WAV bytes. The Arena router (https://github.com/TTS-AGI/tts-router-v2) calls your Space during comparisons.

**Fine-tunes welcome.** Kokoro, MaskGCT, MeloTTS variants already compete. Lead with "Holler Katie" not "Qwen3-TTS fine-tune."

**Note:** Qwen3-TTS base isn't on the Arena yet (request #113 sitting unanswered). We'd be first.

**TTS Arena V2 is buggy** — login issues reported. May need to coordinate via Discord/email instead.

---

## Artificial Analysis Speech Arena

**Arena:** https://artificialanalysis.ai/text-to-speech/arena
**Leaderboard:** https://artificialanalysis.ai/text-to-speech/leaderboard
**Format:** 71 models, blind A/B, Elo. More commercially focused. Tracks pricing.

**How to submit:** Email `hello@artificialanalysis.ai`

**Draft email:**
> Subject: Holler — open-source TTS model for Speech Arena
>
> Hi, I'd like to submit Holler Katie for the Speech Arena. It's a fine-tuned Qwen3-TTS-0.6B with a high-quality American English female voice, optimized for conversational/assistant use cases. Fully open source (Apache 2.0).
>
> HuggingFace: sentium/holler-0.6b
> Demo: [link to Gradio Space]
>
> Happy to set up whatever endpoint format you need.

---

## Pendrokar's TTS Spaces Arena (Easy Win)

**URL:** https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena
**Format:** Community-run, directly calls HF Spaces via Gradio API.

**How to submit:** Open a Discussion on the Space. Maintainer (Pendrokar/Yanis L) is responsive.

**Easiest path** — if your Space has a Gradio interface, it can be added directly. No router integration needed.

---

## HuggingFace Space Setup

**Free options:**
- **CPU** — free, slow (~RTF 3-5). Fine for arena (not real-time, just returns WAV).
- **ZeroGPU** — free A100 via `@spaces.GPU` decorator. ~60s per request. More than enough.

Speed doesn't matter for arenas — it's quality voting, not latency.

**What to build:** Gradio app, ~40 lines. Load model with PyTorch/transformers (not mlx-audio — HF Spaces are Linux/NVIDIA). Expose `synthesize(text) -> wav`. The bf16 weights load directly in standard PyTorch.

---

## Checklist

- [ ] Push weights to HuggingFace: `sentium/holler-0.6b` (bf16) + `sentium/holler-0.6b-6bit`
- [ ] Write model card with audio samples
- [ ] Deploy Gradio Space with PyTorch inference + ZeroGPU
- [ ] Submit to Pendrokar's TTS Spaces Arena (easiest, do first)
- [ ] Submit to TTS Arena V2 (discussion + email mrfakename)
- [ ] Email Artificial Analysis
- [ ] Join TTS-AGI Discord for visibility
