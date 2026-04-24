"""Experiment: mx.compile on code predictor forward pass.
The code predictor's __call__ is where 66% of time goes.
Can we compile it?"""
import time
import os
import numpy as np
import mlx.core as mx
from functools import partial
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

from fast_generate import fast_generate
fast_generate(model, "Hello.", voice="katie")
print("Warmup done.\n")


def generate_with_compiled_codepred(mdl, text, voice="katie", language="english",
                                     temperature=0.6, top_k=50, max_tokens=500,
                                     n_codebooks=12):
    """Generate with mx.compile on the code predictor model forward pass."""
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

    # Try to compile the code predictor's inner model forward
    # This is tricky because of dynamic cache shapes
    # Let's try compiling just the lm_head + sampling as one op
    @mx.compile
    def sample_code_logits(logits, temp_val):
        l = logits[:, -1, :]
        return categorical_sampling(l, temp_val)

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]
    generated_codes = []

    t_gen = time.time()

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
            ct = sample_code_logits(code_logits, temperature)[:, None]
            code_tokens.append(ct)

            for code_idx in range(1, actual_extra):
                code_input = code_embeddings[code_idx - 1](code_tokens[-1])
                code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
                ct = sample_code_logits(code_logits, temperature)[:, None]
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

    gen_ms = (time.time() - t_gen) * 1000

    if not generated_codes:
        return np.array([], dtype=np.float32), 0, gen_ms, 0

    t_decode = time.time()
    codes = mx.stack(generated_codes, axis=1)
    audio, audio_lengths = mdl.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if 0 < valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
    decode_ms = (time.time() - t_decode) * 1000

    return np.array(audio).flatten().astype(np.float32), len(generated_codes), gen_ms, decode_ms


# Compare
texts = [
    "Something about the umami thing appeals to me.",
    "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
]

print("=== Compiled code_logits sampling ===")
for text in texts:
    # Warmup
    generate_with_compiled_codepred(model, text)

    times = []
    for _ in range(3):
        audio, n, gen_ms, dec_ms = generate_with_compiled_codepred(model, text)
        audio_s = len(audio) / 24000
        gen_rtf = (gen_ms / 1000) / audio_s
        times.append(gen_rtf)
    print(f"  compiled: avg gen RTF={sum(times)/len(times):.3f} ({times})")

    times = []
    for _ in range(3):
        from experiment_combined import fast_generate_ncb
        audio, n, gen_ms, dec_ms = fast_generate_ncb(model, text, n_codebooks=12)
        audio_s = len(audio) / 24000
        gen_rtf = (gen_ms / 1000) / audio_s
        times.append(gen_rtf)
    print(f"  baseline: avg gen RTF={sum(times)/len(times):.3f} ({times})")
