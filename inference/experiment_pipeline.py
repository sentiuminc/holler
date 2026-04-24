"""Experiment: Pipelined generation + decode.
Instead of generate-all-then-decode, generate N tokens, kick off decode (lazy),
then continue generating while decode executes on GPU.

On a single Metal GPU, true parallelism isn't possible, but MLX's lazy evaluation
means we can submit decode work without waiting for it, then do more generation
while the GPU works through the queue."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

from fast_generate import fast_generate
fast_generate(model, "Hello.", voice="katie")
print("Warmup done.\n")

text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."


def pipelined_generate(mdl, text, voice="katie", language="english", temperature=0.6,
                       top_k=50, max_tokens=500, n_codebooks=12, decode_every=15):
    """Generate tokens in chunks, overlap decode with next generation chunk.

    Returns list of (audio_np, gen_batch_ms, decode_ms) tuples.
    """
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language=language, speaker=voice
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

    suppress_start = config.vocab_size - 1024
    suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                                  if i != eos_token_id], dtype=mx.int32)

    cache = mdl.talker.make_cache()
    code_cache = mdl.talker.code_predictor.make_cache()
    get_input_emb = mdl.talker.get_input_embeddings()
    code_pred = mdl.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]
    generated_codes = []
    decoded_up_to = 0
    results = []

    mdl.speech_tokenizer.decoder.reset_streaming_state()

    # Pending decode (submitted but not waited on)
    pending_decode = None
    pending_decode_start = None

    t_total = time.time()

    for step in range(max_tokens):
        logits, hidden = mdl.talker(input_embeds, cache=cache)

        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1)
        if top_k > 0:
            top_k_vals = mx.sort(first_logits, axis=-1)[:, -top_k:]
            threshold = top_k_vals[:, 0:1]
            first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
        next_token = categorical_sampling(first_logits, temperature)[:, None]
        is_eos = next_token[0, 0] == eos_token_id

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None; c.values = None; c.offset = 0

        if actual_extra > 0:
            code_0_embed = get_input_emb(next_token)
            code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=0)
            ct = categorical_sampling(code_logits[:, -1, :], temperature)[:, None]
            code_tokens.append(ct)

            for code_idx in range(1, actual_extra):
                code_input = code_embeddings[code_idx - 1](code_tokens[-1])
                code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
                ct = categorical_sampling(code_logits[:, -1, :], temperature)[:, None]
                code_tokens.append(ct)

        while len(code_tokens) < num_code_groups:
            code_tokens.append(mx.zeros_like(next_token))

        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_len:
            text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed

        codec_embed = get_input_emb(next_token)
        for i in range(min(actual_extra, num_code_groups - 1)):
            codec_embed = codec_embed + code_embeddings[i](code_tokens[i + 1])
        input_embeds = text_embed + codec_embed

        mx.eval(input_embeds, is_eos)
        if is_eos.item():
            break
        generated_codes.append(all_codes)

        # Check if we should decode
        n_new = len(generated_codes) - decoded_up_to
        if n_new >= decode_every:
            # If there's a pending decode, wait for it now
            if pending_decode is not None:
                mx.eval(pending_decode)
                audio_np = np.array(pending_decode.squeeze(1)[0]).flatten().astype(np.float32)
                results.append(audio_np)
                pending_decode = None

            # Submit new decode (lazy — won't block)
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)  # Need to eval the codes before streaming_step

            pending_decode = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            decoded_up_to = len(generated_codes)

    # Wait for any pending decode
    if pending_decode is not None:
        mx.eval(pending_decode)
        audio_np = np.array(pending_decode.squeeze(1)[0]).flatten().astype(np.float32)
        results.append(audio_np)

    # Decode remaining tokens
    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)
        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        mx.eval(wav)
        audio_np = np.array(wav.squeeze(1)[0]).flatten().astype(np.float32)
        results.append(audio_np)

    total_ms = (time.time() - t_total) * 1000

    if results:
        audio = np.concatenate(results)
    else:
        audio = np.array([], dtype=np.float32)

    return audio, len(generated_codes), total_ms


# Compare pipelined vs sequential
print("=== Pipelined vs Sequential ===")

for label, func_args in [
    ("Sequential (16cb)", {"n_codebooks": 16}),
    ("Sequential (12cb)", {"n_codebooks": 12}),
]:
    from experiment_combined import fast_generate_ncb
    audio, n, gen_ms, dec_ms = fast_generate_ncb(model, text, **func_args)
    audio_s = len(audio) / SAMPLE_RATE
    total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s
    print(f"  {label}: {gen_ms+dec_ms:.0f}ms total, {audio_s:.1f}s audio, RTF={total_rtf:.3f}")

for decode_every in [10, 15, 20, 30, 50]:
    audio, n, total_ms = pipelined_generate(model, text, decode_every=decode_every)
    audio_s = len(audio) / SAMPLE_RATE
    rtf = (total_ms / 1000) / audio_s
    print(f"  Pipeline (12cb, dec_every={decode_every:>2}): {total_ms:.0f}ms total, {audio_s:.1f}s audio, RTF={rtf:.3f}")

# Full benchmark with pipelined approach
print("\n=== Full benchmark — Pipelined 12cb ===")
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

rtfs = []
print(f"{'Words':>5} {'Tokens':>6} {'TotalMs':>8} {'Audio':>6} {'RTF':>6}  Text")
print("-" * 80)

for text in sentences:
    audio, n, total_ms = pipelined_generate(model, text, decode_every=20)
    audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
    rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
    words = len(text.split())
    rtfs.append(rtf)
    print(f"{words:>5} {n:>6} {total_ms:>7.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}  {text[:40]}")

print(f"\n  Average RTF: {sum(rtfs)/len(rtfs):.3f}")
print(f"  Max RTF: {max(rtfs):.3f}")
target = "✅" if sum(rtfs)/len(rtfs) <= 0.50 and max(rtfs) <= 0.70 else "❌"
print(f"  Target (avg ≤ 0.50, max ≤ 0.70): {target}")
