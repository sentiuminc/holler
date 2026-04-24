"""Profile 12-codebook generation to find remaining optimization targets."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

# Warmup
from fast_generate import fast_generate
fast_generate(model, "Hello.", voice="katie")
print("Warmup done.\n")

text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."
N_CODEBOOKS = 12

config = model.config.talker_config
eos_token_id = config.codec_eos_token_id
num_code_groups = config.num_code_groups
actual_extra = N_CODEBOOKS - 1

suppress_start = config.vocab_size - 1024
suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                              if i != eos_token_id], dtype=mx.int32)

input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
    text, language="english", speaker="katie"
)
mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

cache = model.talker.make_cache()
code_cache = model.talker.code_predictor.make_cache()
get_input_emb = model.talker.get_input_embeddings()
code_pred = model.talker.code_predictor
code_embeddings = code_pred.codec_embedding

trailing_idx = 0
trailing_len = trailing_text_hidden.shape[1]
generated_codes = []

talker_times = []
code_pred_times = []
other_times = []

t_total = time.time()

for step in range(500):
    # Talker
    t0 = time.time()
    logits, hidden = model.talker(input_embeds, cache=cache)
    first_logits = logits[:, -1, :]
    first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
        mx.array(float("-inf"), first_logits.dtype), axis=-1)
    top_k_vals = mx.sort(first_logits, axis=-1)[:, -50:]
    threshold = top_k_vals[:, 0:1]
    first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
    next_token = categorical_sampling(first_logits, 0.6)[:, None]
    is_eos = next_token[0, 0] == eos_token_id
    mx.eval(next_token, is_eos)
    talker_times.append((time.time() - t0) * 1000)

    if is_eos.item():
        break

    # Code predictor
    t0 = time.time()
    code_tokens = [next_token]
    code_hidden = hidden[:, -1:, :]
    for c in code_cache:
        c.keys = None; c.values = None; c.offset = 0

    code_0_embed = get_input_emb(next_token)
    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
    code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=0)
    ct = categorical_sampling(code_logits[:, -1, :], 0.6)[:, None]
    code_tokens.append(ct)

    for code_idx in range(1, actual_extra):
        code_input = code_embeddings[code_idx - 1](code_tokens[-1])
        code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
        ct = categorical_sampling(code_logits[:, -1, :], 0.6)[:, None]
        code_tokens.append(ct)

    # Pad
    while len(code_tokens) < num_code_groups:
        code_tokens.append(mx.zeros_like(next_token))
    all_codes = mx.concatenate(code_tokens, axis=1)
    mx.eval(all_codes)
    code_pred_times.append((time.time() - t0) * 1000)

    # Embed prep
    t0 = time.time()
    if trailing_idx < trailing_len:
        text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
        trailing_idx += 1
    else:
        text_embed = tts_pad_embed

    codec_embed = get_input_emb(next_token)
    for i in range(actual_extra):
        codec_embed = codec_embed + code_embeddings[i](code_tokens[i + 1])
    input_embeds = text_embed + codec_embed
    mx.eval(input_embeds)
    other_times.append((time.time() - t0) * 1000)

    generated_codes.append(all_codes)

total_ms = (time.time() - t_total) * 1000
n = len(talker_times) - 1  # Last one was EOS check

audio_s = len(generated_codes) / 12.0
rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999

print(f"12 codebooks, {n} tokens, {audio_s:.1f}s audio, {total_ms:.0f}ms, RTF={rtf:.3f}")
print()

talker_avg = sum(talker_times[1:]) / (len(talker_times)-1)  # Skip first (prefill)
code_avg = sum(code_pred_times) / len(code_pred_times)
other_avg = sum(other_times) / len(other_times)
total_per_tok = talker_avg + code_avg + other_avg

print(f"Per-token breakdown (avg, excluding prefill):")
print(f"  Talker:     {talker_avg:.1f}ms ({talker_avg/total_per_tok*100:.0f}%)")
print(f"  CodePred:   {code_avg:.1f}ms ({code_avg/total_per_tok*100:.0f}%)")
print(f"  EmbedPrep:  {other_avg:.1f}ms ({other_avg/total_per_tok*100:.0f}%)")
print(f"  Total:      {total_per_tok:.1f}ms")
print(f"  Prefill:    {talker_times[0]:.1f}ms")
print()

# What's the theoretical minimum?
print(f"At {total_per_tok:.1f}ms/tok × 12Hz = {total_per_tok*12:.0f}ms per 1s audio")
print(f"Theoretical RTF: {total_per_tok*12/1000:.3f}")
print(f"Measured RTF: {rtf:.3f}")
print(f"Overhead: {((rtf - total_per_tok*12/1000)/(total_per_tok*12/1000))*100:.0f}%")
