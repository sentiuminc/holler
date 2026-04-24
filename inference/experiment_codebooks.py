"""Experiment: What happens if we use fewer codebooks?
The code predictor generates 15 sub-codebook tokens per speech token, taking 70% of gen time.
If we can use fewer codebooks with acceptable quality, we can dramatically speed up generation.

Also tests: Metal cache limit, greedy decoding for code predictor."""
import time
import os
import sys
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.expanduser("~/Downloads/holler-codebook-experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = load(CHECKPOINT)

# Warmup
from fast_generate import fast_generate
fast_generate(model, "Hello.", voice="katie")
print("Warmup done.\n")

test_text = "Something about the umami thing appeals to me."


def generate_with_n_codebooks(mdl, text, voice="katie", language="english",
                               temperature=0.6, top_k=50, max_tokens=500,
                               n_codebooks=16):
    """Generate with only first N codebooks. Fill remaining with zeros."""
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups

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

        # Only generate up to n_codebooks-1 additional codebooks
        actual_extra = min(n_codebooks - 1, num_code_groups - 1)

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

        # Pad remaining codebooks with zeros
        while len(code_tokens) < num_code_groups:
            code_tokens.append(mx.zeros_like(next_token))

        all_codes = mx.concatenate(code_tokens, axis=1)

        # Prepare next input (always uses all codebooks for the embedding)
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
    audio, audio_lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if 0 < valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
    decode_ms = (time.time() - t_decode) * 1000

    return np.array(audio).flatten().astype(np.float32), len(generated_codes), gen_ms, decode_ms


# Experiment 1: Metal cache limit
print("=== Experiment: Metal cache limit ===")
for cache_gb in [0, 2, 4, 8]:
    if cache_gb > 0:
        mx.metal.set_cache_limit(cache_gb * 1024 * 1024 * 1024)
    else:
        mx.metal.set_cache_limit(0)

    audio, n, gen_ms, dec_ms = fast_generate(model, test_text, voice="katie")
    audio_s = len(audio) / SAMPLE_RATE
    gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
    total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999
    print(f"  cache_limit={cache_gb}GB: gen={gen_ms:.0f}ms dec={dec_ms:.0f}ms genRTF={gen_rtf:.3f} totalRTF={total_rtf:.3f}")

# Reset
mx.metal.set_cache_limit(0)
print()

# Experiment 2: Fewer codebooks
print("=== Experiment: Fewer codebooks ===")
print(f"{'CB':>3} {'Tokens':>6} {'GenMs':>7} {'DecMs':>7} {'Audio':>6} {'GenRTF':>7} {'TotRTF':>7}")
print("-" * 55)

for n_cb in [16, 12, 8, 4, 2, 1]:
    audio, n_tok, gen_ms, dec_ms = generate_with_n_codebooks(
        model, test_text, n_codebooks=n_cb
    )
    audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
    gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
    total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999

    fname = os.path.join(OUTPUT_DIR, f"codebooks_{n_cb}.wav")
    if len(audio) > 0:
        sf.write(fname, audio, SAMPLE_RATE)

    print(f"{n_cb:>3} {n_tok:>6} {gen_ms:>6.0f}ms {dec_ms:>6.0f}ms {audio_s:>5.1f}s {gen_rtf:>6.3f} {total_rtf:>6.3f}")

print(f"\nAudio samples saved to {OUTPUT_DIR}")
print("Listen to compare quality: codebooks_16.wav (baseline) vs lower counts")

# Experiment 3: Greedy (temperature=0) for code predictor tokens only
print("\n=== Experiment: Temperature 0 (greedy) for code predictor ===")
# This avoids the categorical_sampling overhead for sub-codebooks
# We still sample the main token with temperature for diversity

def generate_greedy_codes(mdl, text, voice="katie", language="english",
                           temperature=0.6, top_k=50, max_tokens=500):
    """Main token sampled normally, code predictor tokens greedy (argmax)."""
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups

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

        code_0_embed = get_input_emb(next_token)
        code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
        code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=0)
        # GREEDY for code predictor
        ct = mx.argmax(code_logits[:, -1, :], axis=-1, keepdims=True)[:, None]
        code_tokens.append(ct)

        for code_idx in range(1, num_code_groups - 1):
            code_input = code_embeddings[code_idx - 1](code_tokens[-1])
            code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
            ct = mx.argmax(code_logits[:, -1, :], axis=-1, keepdims=True)[:, None]
            code_tokens.append(ct)

        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_len:
            text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed

        codec_embed = get_input_emb(next_token)
        for i in range(num_code_groups - 1):
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
    audio, audio_lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if 0 < valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
    decode_ms = (time.time() - t_decode) * 1000
    return np.array(audio).flatten().astype(np.float32), len(generated_codes), gen_ms, decode_ms

audio, n, gen_ms, dec_ms = generate_greedy_codes(model, test_text)
audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999
if len(audio) > 0:
    sf.write(os.path.join(OUTPUT_DIR, "greedy_codes.wav"), audio, SAMPLE_RATE)
print(f"  Greedy codes: gen={gen_ms:.0f}ms dec={dec_ms:.0f}ms genRTF={gen_rtf:.3f} totalRTF={total_rtf:.3f}")

# Compare with normal sampling
audio, n, gen_ms, dec_ms = fast_generate(model, test_text, voice="katie")
audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999
print(f"  Normal:       gen={gen_ms:.0f}ms dec={dec_ms:.0f}ms genRTF={gen_rtf:.3f} totalRTF={total_rtf:.3f}")
