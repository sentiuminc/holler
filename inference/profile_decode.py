"""Profile where time goes in codec decode.
The decoder is ~40% of total time. Can we make it faster?"""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)

# Generate some tokens first
from fast_generate import fast_generate
audio, n_tokens, gen_ms, decode_ms = fast_generate(
    model, "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
    voice="katie"
)
print(f"Generated {n_tokens} tokens, gen={gen_ms:.0f}ms, decode={decode_ms:.0f}ms\n")

# Now profile decode approaches
# Regenerate tokens so we have the codes
config = model.config.talker_config
eos_token_id = config.codec_eos_token_id
num_code_groups = config.num_code_groups
text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."

input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
    text, language="english", speaker="katie"
)
mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

cache = model.talker.make_cache()
code_cache = model.talker.code_predictor.make_cache()
from mlx_lm.sample_utils import categorical_sampling

suppress_start = config.vocab_size - 1024
suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                              if i != eos_token_id], dtype=mx.int32)

generated_codes = []
trailing_idx = 0
get_input_emb = model.talker.get_input_embeddings()
code_pred = model.talker.code_predictor
code_embeddings = code_pred.codec_embedding

for step in range(500):
    logits, hidden = model.talker(input_embeds, cache=cache)
    first_logits = logits[:, -1, :]
    first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
        mx.array(float("-inf"), first_logits.dtype), axis=-1)
    next_token = categorical_sampling(first_logits, 0.6)[:, None]
    is_eos = next_token[0, 0] == eos_token_id

    code_tokens = [next_token]
    code_hidden = hidden[:, -1:, :]
    for c in code_cache:
        c.keys = None; c.values = None; c.offset = 0

    code_0_embed = get_input_emb(next_token)
    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
    code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=0)
    ct = categorical_sampling(code_logits[:, -1, :], 0.6)[:, None]
    code_tokens.append(ct)

    for code_idx in range(1, num_code_groups - 1):
        code_input = code_embeddings[code_idx - 1](code_tokens[-1])
        code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
        ct = categorical_sampling(code_logits[:, -1, :], 0.6)[:, None]
        code_tokens.append(ct)

    all_codes = mx.concatenate(code_tokens, axis=1)

    if trailing_idx < trailing_text_hidden.shape[1]:
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

codes = mx.stack(generated_codes, axis=1)  # [1, seq, 16]
n = codes.shape[1]
print(f"\nHave {n} tokens to decode\n")

# Test 1: Full decode (chunked_decode)
print("=== Decode Methods ===")

for trial in range(3):
    t0 = time.time()
    audio, lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(lengths[0])
    if 0 < valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)
    ms = (time.time() - t0) * 1000
    audio_s = audio.shape[0] / 24000
    decode_rtf = (ms / 1000) / audio_s
    print(f"  chunked_decode (trial {trial+1}): {ms:.0f}ms for {audio_s:.1f}s audio, decode RTF={decode_rtf:.3f}")

# Test 2: Direct decoder call (no chunking)
print()
codes_t = mx.transpose(codes, (0, 2, 1))  # [1, 16, seq]
for trial in range(3):
    t0 = time.time()
    wav = model.speech_tokenizer.decoder(codes_t)
    wav_out = wav.squeeze(1)[0]
    mx.eval(wav_out)
    ms = (time.time() - t0) * 1000
    audio_s = wav_out.shape[0] / 24000
    decode_rtf = (ms / 1000) / audio_s
    print(f"  direct decoder (trial {trial+1}): {ms:.0f}ms for {audio_s:.1f}s audio, decode RTF={decode_rtf:.3f}")

# Test 3: Streaming decode with various chunk sizes
print()
for chunk_size in [10, 25, 50, 100, n]:
    model.speech_tokenizer.decoder.reset_streaming_state()
    t0 = time.time()
    total_samples = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_codes = codes_t[:, :, start:end]
        wav = model.speech_tokenizer.decoder.streaming_step(chunk_codes)
        total_samples += wav.shape[-1]
        mx.eval(wav)
    ms = (time.time() - t0) * 1000
    audio_s = total_samples / 24000
    decode_rtf = (ms / 1000) / audio_s if audio_s > 0 else 999
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"  streaming_step chunk={chunk_size:>3} ({n_chunks:>2} chunks): {ms:.0f}ms, RTF={decode_rtf:.3f}")

# Test 4: Direct decoder call, profiled by component
print("\n=== Decoder Component Profiling ===")
codes_t2 = mx.transpose(codes, (0, 2, 1))
decoder = model.speech_tokenizer.decoder

# Dequantize
t0 = time.time()
hidden = decoder.quantizer.decode(codes_t2)
hidden = mx.transpose(hidden, (0, 2, 1))
mx.eval(hidden)
print(f"  Dequantize: {(time.time()-t0)*1000:.0f}ms")

# Pre-conv
t0 = time.time()
h2 = decoder.pre_conv(hidden)
mx.eval(h2)
print(f"  Pre-conv: {(time.time()-t0)*1000:.0f}ms")

# Transformer
t0 = time.time()
h3 = decoder.pre_transformer(h2)
mx.eval(h3)
print(f"  Transformer: {(time.time()-t0)*1000:.0f}ms")

# Upsample
t0 = time.time()
h4 = h3
for upsample_layers in decoder.upsample:
    for layer in upsample_layers:
        h4 = layer(h4)
mx.eval(h4)
print(f"  Upsample: {(time.time()-t0)*1000:.0f}ms")

# Main decoder blocks
t0 = time.time()
wav = h4
for decoder_layer in decoder.decoder:
    wav = decoder_layer(wav)
mx.eval(wav)
print(f"  Decoder blocks: {(time.time()-t0)*1000:.0f}ms")
