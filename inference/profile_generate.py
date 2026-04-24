"""Profile where time goes in the Qwen3-TTS generate loop.
Measures: prefill, per-token talker forward, code predictor, sampling, codec decode, overhead."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000

print(f"Loading model from {CHECKPOINT}...")
model = load(CHECKPOINT)

# Warmup
for r in model.generate(text="Hello.", voice="katie", language="english",
                        temperature=0.6, stream=True, streaming_interval=0.1):
    pass
print("Warmup done.\n")

# Now instrument a single generation manually
text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."

config = model.config.talker_config
eos_token_id = config.codec_eos_token_id
suppress_tokens = [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos_token_id]

t_prep = time.time()
input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
    text, language="english", speaker="katie"
)
mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)
prep_ms = (time.time() - t_prep) * 1000

cache = model.talker.make_cache()
code_cache = model.talker.code_predictor.make_cache()

generated_codes = []
generated_token_ids = []
trailing_idx = 0

# Timing accumulators
talker_times = []
sample_times = []
code_pred_times = []
embed_times = []
eval_times = []

t_total = time.time()

for step in range(500):
    # --- Talker forward ---
    t0 = time.time()
    logits, hidden = model.talker(input_embeds, cache=cache)
    mx.eval(logits, hidden)
    talker_times.append((time.time() - t0) * 1000)

    # --- Sample first codebook token ---
    t0 = time.time()
    next_token = model._sample_token(
        logits, temperature=0.6, top_k=50, top_p=1.0,
        repetition_penalty=1.05,
        generated_tokens=generated_token_ids if generated_token_ids else None,
        suppress_tokens=suppress_tokens, eos_token_id=eos_token_id,
    )
    mx.eval(next_token)
    sample_times.append((time.time() - t0) * 1000)

    is_eos = next_token[0, 0] == eos_token_id
    if is_eos.item():
        break

    # --- Code predictor (remaining codebooks) ---
    t0 = time.time()
    code_tokens = [next_token]
    code_hidden = hidden[:, -1:, :]
    for c in code_cache:
        c.keys = None
        c.values = None
        c.offset = 0

    for code_idx in range(config.num_code_groups - 1):
        if code_idx == 0:
            code_0_embed = model.talker.get_input_embeddings()(next_token)
            code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
        else:
            code_embed = model.talker.code_predictor.codec_embedding[code_idx - 1](code_tokens[-1])
            code_input = code_embed
        code_logits, code_cache, _ = model.talker.code_predictor(
            code_input, cache=code_cache, generation_step=code_idx,
        )
        next_code = model._sample_token(code_logits, temperature=0.6, top_k=50, top_p=1.0)
        code_tokens.append(next_code)

    all_codes = mx.concatenate(code_tokens, axis=1)
    mx.eval(all_codes)
    code_pred_times.append((time.time() - t0) * 1000)

    # --- Prepare next input ---
    t0 = time.time()
    if trailing_idx < trailing_text_hidden.shape[1]:
        text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
        trailing_idx += 1
    else:
        text_embed = tts_pad_embed

    codec_embed = model.talker.get_input_embeddings()(next_token)
    for i, code in enumerate(code_tokens[1:]):
        codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[i](code)
    input_embeds = text_embed + codec_embed
    mx.eval(input_embeds)
    embed_times.append((time.time() - t0) * 1000)

    generated_token_ids.append(int(next_token[0, 0]))
    generated_codes.append(all_codes)

    if step > 0 and step % 50 == 0:
        mx.clear_cache()

total_ms = (time.time() - t_total) * 1000
n = len(talker_times)

# Codec decode timing
t_decode = time.time()
if generated_codes:
    codes = mx.stack(generated_codes, axis=1)
    audio, audio_lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if valid_len > 0 and valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
decode_ms = (time.time() - t_decode) * 1000

audio_dur = len(generated_codes) / 12.0  # 12Hz codec rate
rtf = (total_ms / 1000) / audio_dur if audio_dur > 0 else 999

print(f"Text: \"{text[:60]}...\"")
print(f"Tokens generated: {n}")
print(f"Audio duration: {audio_dur:.1f}s")
print(f"Total generation: {total_ms:.0f}ms (RTF {rtf:.3f})")
print(f"Codec decode: {decode_ms:.0f}ms")
print(f"Prep: {prep_ms:.0f}ms")
print()
print(f"{'Component':>20} {'Total ms':>10} {'Per-token':>10} {'% of gen':>10}")
print("-" * 55)
talker_total = sum(talker_times)
sample_total = sum(sample_times)
code_total = sum(code_pred_times)
embed_total = sum(embed_times)
accounted = talker_total + sample_total + code_total + embed_total
overhead = total_ms - accounted

for name, total in [("Talker forward", talker_total), ("Sampling (1st CB)", sample_total),
                     ("Code predictor", code_total), ("Embed prep", embed_total),
                     ("Unaccounted", overhead)]:
    per_tok = total / n if n > 0 else 0
    pct = total / total_ms * 100 if total_ms > 0 else 0
    print(f"{name:>20} {total:>9.0f}ms {per_tok:>9.1f}ms {pct:>9.1f}%")

print()
print(f"Token rate: {n / (total_ms/1000):.1f} tok/s (need {12/0.5:.0f} for RTF 0.5)")
print(f"Per-token budget at RTF 0.5: {1000/24:.1f}ms (currently {total_ms/n:.1f}ms)")

# Per-token breakdown (first 5 and last 5)
print(f"\nPer-token detail (first 5):")
for i in range(min(5, n)):
    print(f"  step {i}: talker={talker_times[i]:.1f}ms sample={sample_times[i]:.1f}ms code={code_pred_times[i]:.1f}ms embed={embed_times[i]:.1f}ms")
if n > 10:
    print(f"Per-token detail (last 5):")
    for i in range(max(5, n-5), n):
        print(f"  step {i}: talker={talker_times[i]:.1f}ms sample={sample_times[i]:.1f}ms code={code_pred_times[i]:.1f}ms embed={embed_times[i]:.1f}ms")
