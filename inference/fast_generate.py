"""Fast Holler inference — custom generate loop optimized for RTF.

Key optimizations over mlx-audio's generate():
1. Separate generation from codec decoding (no streaming overhead during generation)
2. Single mx.eval() per token (not per-component)
3. No mx.clear_cache() during generation loop
4. Minimal Python overhead in hot path
5. Pre-computed suppress token mask instead of per-token array creation
"""
import time
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

SAMPLE_RATE = 24000


def fast_generate(model, text, voice="katie", language="english", temperature=0.6,
                  top_k=50, max_tokens=500):
    """Generate speech tokens as fast as possible, decode after."""
    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups

    # Prep inputs (one-time)
    input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
        text, language=language, speaker=voice
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

    # Pre-compute suppress mask (once, reused every token)
    suppress_start = config.vocab_size - 1024
    suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                                  if i != eos_token_id], dtype=mx.int32)

    cache = model.talker.make_cache()
    code_cache = model.talker.code_predictor.make_cache()

    generated_codes = []
    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]

    get_input_emb = model.talker.get_input_embeddings()
    code_pred = model.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    t_gen_start = time.time()

    for step in range(max_tokens):
        # Talker forward
        logits, hidden = model.talker(input_embeds, cache=cache)

        # Sample first codebook (suppress special tokens, apply top_k)
        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(
            first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1
        )
        if top_k > 0:
            top_k_vals = mx.sort(first_logits, axis=-1)[:, -top_k:]
            threshold = top_k_vals[:, 0:1]
            first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
        next_token = categorical_sampling(first_logits, temperature)[:, None]  # [1, 1]

        # EOS check (lazy — will be evaluated with the batch below)
        is_eos = next_token[0, 0] == eos_token_id

        # Code predictor for remaining codebooks
        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]

        # Reset code cache inline (no allocation)
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0

        # First code predictor step: concat hidden + code_0_embed
        code_0_embed = get_input_emb(next_token)
        code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
        code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=0)
        ct = categorical_sampling(code_logits[:, -1, :], temperature)[:, None]
        code_tokens.append(ct)

        # Remaining code predictor steps
        for code_idx in range(1, num_code_groups - 1):
            code_input = code_embeddings[code_idx - 1](code_tokens[-1])
            code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
            ct = categorical_sampling(code_logits[:, -1, :], temperature)[:, None]
            code_tokens.append(ct)

        all_codes = mx.concatenate(code_tokens, axis=1)  # [1, 16]

        # Prepare next input embedding
        if trailing_idx < trailing_len:
            text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed

        codec_embed = get_input_emb(next_token)
        for i in range(num_code_groups - 1):
            codec_embed = codec_embed + code_embeddings[i](code_tokens[i + 1])
        input_embeds = text_embed + codec_embed

        # Single eval point per token
        mx.eval(input_embeds, is_eos)

        if is_eos.item():
            break

        generated_codes.append(all_codes)

    gen_ms = (time.time() - t_gen_start) * 1000

    if not generated_codes:
        return np.array([], dtype=np.float32), 0, gen_ms, 0

    # Decode all at once
    t_decode = time.time()
    codes = mx.stack(generated_codes, axis=1)  # [1, seq, 16]
    audio, audio_lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if 0 < valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
    decode_ms = (time.time() - t_decode) * 1000

    audio_np = np.array(audio).flatten().astype(np.float32)
    return audio_np, len(generated_codes), gen_ms, decode_ms


def benchmark():
    checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "checkpoints", "katie-v6-4bit")

    sentences = [
        "Hey!",
        "Got it.",
        "What's on your mind?",
        "Yeah, that's pretty common with voice input.",
        "Something about the umami thing appeals to me.",
        "I'd say mushroom, even though it's technically not one.",
        "The server was returning a five hundred error because the database connection pool was exhausted.",
        "Probably better to just catch stuff as it comes up than do a whole tally.",
        "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
        "Like, does the speech to text screw up like that a lot, or was that just this time around?",
    ]

    print(f"Loading model from {checkpoint}...")
    model = load(checkpoint)

    # Warmup
    fast_generate(model, "Hello.", voice="katie")
    fast_generate(model, "Testing warmup.", voice="katie")
    print("Warmup done.\n")

    for pass_num in range(1, 4):
        print(f"{'='*85}")
        print(f"Pass {pass_num}/3 — fast_generate (gen-only + decode separate)")
        print(f"{'='*85}")
        print(f"{'Words':>5} {'Tokens':>6} {'GenMs':>7} {'DecMs':>7} {'Audio':>6} {'GenRTF':>7} {'TotRTF':>7}  Text")
        print("-" * 85)

        gen_rtfs = []
        total_rtfs = []

        for text in sentences:
            audio, n_tokens, gen_ms, decode_ms = fast_generate(model, text, voice="katie")
            audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
            gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
            total_rtf = ((gen_ms + decode_ms) / 1000) / audio_s if audio_s > 0 else 999
            words = len(text.split())

            gen_rtfs.append(gen_rtf)
            total_rtfs.append(total_rtf)

            print(f"{words:>5} {n_tokens:>6} {gen_ms:>6.0f}ms {decode_ms:>6.0f}ms {audio_s:>5.1f}s {gen_rtf:>6.3f} {total_rtf:>6.3f}  {text[:40]}")

        avg_gen = sum(gen_rtfs) / len(gen_rtfs)
        avg_total = sum(total_rtfs) / len(total_rtfs)
        max_gen = max(gen_rtfs)
        max_total = max(total_rtfs)

        print(f"\n  Gen-only:  avg={avg_gen:.3f} max={max_gen:.3f}")
        print(f"  Gen+Dec:   avg={avg_total:.3f} max={max_total:.3f}")

        target_gen = "✅" if avg_gen <= 0.50 and max_gen <= 0.70 else "❌"
        target_total = "✅" if avg_total <= 0.50 and max_total <= 0.70 else "❌"
        print(f"  Target gen-only (avg ≤ 0.50, max ≤ 0.70): {target_gen}")
        print(f"  Target total   (avg ≤ 0.50, max ≤ 0.70): {target_total}")
        print()


if __name__ == "__main__":
    benchmark()
