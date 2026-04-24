"""Deep profile of the codec decoder components.
The decoder accounts for ~40% of total RTF. Can we make it faster?

Decoder pipeline:
1. Dequantize (SplitRVQ) → [batch, codebook_dim, time]
2. Pre-conv (CausalConv1d 3-kernel) → [batch, time, latent_dim]
3. Transformer (DecoderTransformer, 4 layers) → [batch, time, latent_dim]
4. Upsample (2x CausalTransposeConv + ConvNeXt) → upsampled
5. Decoder blocks (InitConv + 4x DecoderBlock + SnakeBeta + OutputConv)
   Each DecoderBlock: SnakeBeta → TransposeConv upsample → 3x ResidualUnit

Profile each component separately to find the bottleneck."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

# Generate tokens first
from fast_generate import fast_generate
audio, n_tokens, gen_ms, dec_ms = fast_generate(
    model, "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
    voice="katie"
)
print(f"Generated {n_tokens} tokens\n")

# Re-generate to get the codes
from experiment_combined import fast_generate_ncb
audio, n_tokens, gen_ms, dec_ms = fast_generate_ncb(model,
    "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
    n_codebooks=12)

# Manually build codes for profiling
config = model.config.talker_config
from mlx_lm.sample_utils import categorical_sampling

input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
    "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
    language="english", speaker="katie"
)
mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

eos_token_id = config.codec_eos_token_id
suppress_start = config.vocab_size - 1024
suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                              if i != eos_token_id], dtype=mx.int32)
cache = model.talker.make_cache()
code_cache = model.talker.code_predictor.make_cache()
get_input_emb = model.talker.get_input_embeddings()
code_pred = model.talker.code_predictor
code_embeddings = code_pred.codec_embedding
trailing_idx = 0
generated_codes = []
actual_extra = 11

for step in range(500):
    logits, hidden = model.talker(input_embeds, cache=cache)
    first_logits = logits[:, -1, :]
    first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
        mx.array(float("-inf"), first_logits.dtype), axis=-1)
    top_k_vals = mx.sort(first_logits, axis=-1)[:, -50:]
    threshold = top_k_vals[:, 0:1]
    first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
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
    for code_idx in range(1, actual_extra):
        code_input = code_embeddings[code_idx - 1](code_tokens[-1])
        code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
        ct = categorical_sampling(code_logits[:, -1, :], 0.6)[:, None]
        code_tokens.append(ct)
    while len(code_tokens) < 16:
        code_tokens.append(mx.zeros_like(next_token))
    all_codes = mx.concatenate(code_tokens, axis=1)

    if trailing_idx < trailing_text_hidden.shape[1]:
        text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
        trailing_idx += 1
    else:
        text_embed = tts_pad_embed
    codec_embed = get_input_emb(next_token)
    for i in range(actual_extra):
        codec_embed = codec_embed + code_embeddings[i](code_tokens[i + 1])
    input_embeds = text_embed + codec_embed
    mx.eval(input_embeds, is_eos)
    if is_eos.item():
        break
    generated_codes.append(all_codes)

codes = mx.stack(generated_codes, axis=1)  # [1, seq, 16]
n = codes.shape[1]
print(f"Have {n} tokens for decode profiling\n")

# Now profile the decode path
decoder = model.speech_tokenizer.decoder
codes_t = mx.transpose(codes, (0, 2, 1))  # [1, 16, seq]

# Full decode timing (3 trials)
print("=== Full decode (chunked_decode, chunk_size=300) ===")
for trial in range(3):
    t0 = time.time()
    audio_out, lengths = model.speech_tokenizer.decode(codes)
    a = audio_out[0]
    mx.eval(a)
    ms = (time.time() - t0) * 1000
    audio_s = a.shape[0] / 24000
    print(f"  Trial {trial+1}: {ms:.0f}ms for {audio_s:.1f}s audio (decode RTF={ms/1000/audio_s:.3f})")

# Direct decoder call (no chunking)
print("\n=== Direct decoder (no chunking) ===")
for trial in range(3):
    t0 = time.time()
    wav = decoder(codes_t)
    wav_out = wav.squeeze(1)[0]
    mx.eval(wav_out)
    ms = (time.time() - t0) * 1000
    audio_s = wav_out.shape[0] / 24000
    print(f"  Trial {trial+1}: {ms:.0f}ms for {audio_s:.1f}s audio (decode RTF={ms/1000/audio_s:.3f})")

# Component-by-component profiling
print("\n=== Component profiling (direct decoder path) ===")

# Step 1: Dequantize
t0 = time.time()
hidden = decoder.quantizer.decode(codes_t)
hidden = mx.transpose(hidden, (0, 2, 1))  # NCL → NLC
mx.eval(hidden)
dequant_ms = (time.time() - t0) * 1000
print(f"  1. Dequantize (SplitRVQ): {dequant_ms:.0f}ms  shape={hidden.shape}")

# Step 2: Pre-conv
t0 = time.time()
h2 = decoder.pre_conv(hidden)
mx.eval(h2)
preconv_ms = (time.time() - t0) * 1000
print(f"  2. Pre-conv (CausalConv1d k=3): {preconv_ms:.0f}ms  shape={h2.shape}")

# Step 3: Transformer
t0 = time.time()
h3 = decoder.pre_transformer(h2)
mx.eval(h3)
transformer_ms = (time.time() - t0) * 1000
print(f"  3. Transformer (4 layers): {transformer_ms:.0f}ms  shape={h3.shape}")

# Step 4: Upsample blocks
t0 = time.time()
h4 = h3
for upsample_layers in decoder.upsample:
    for layer in upsample_layers:
        h4 = layer(h4)
mx.eval(h4)
upsample_ms = (time.time() - t0) * 1000
print(f"  4. Upsample (2x CausalTranspose + ConvNeXt): {upsample_ms:.0f}ms  shape={h4.shape}")

# Step 5: Main decoder blocks
t0 = time.time()
wav = h4
# Initial conv
wav = decoder.decoder[0](wav)
mx.eval(wav)
init_conv_ms = (time.time() - t0) * 1000
print(f"  5a. InitConv: {init_conv_ms:.0f}ms  shape={wav.shape}")

# 4 decoder blocks
for block_idx in range(1, 5):
    t0 = time.time()
    wav = decoder.decoder[block_idx](wav)
    mx.eval(wav)
    block_ms = (time.time() - t0) * 1000
    print(f"  5{chr(97+block_idx)}. DecoderBlock[{block_idx}]: {block_ms:.0f}ms  shape={wav.shape}")

# Output snake + conv
t0 = time.time()
wav = decoder.decoder[5](wav)  # SnakeBeta
wav = decoder.decoder[6](wav)  # OutputConv
wav = mx.transpose(wav, (0, 2, 1))  # NLC → NCL
wav = mx.clip(wav, -1.0, 1.0)
mx.eval(wav)
output_ms = (time.time() - t0) * 1000
print(f"  5f. Output (Snake+Conv+clip): {output_ms:.0f}ms  shape={wav.shape}")

total_components = dequant_ms + preconv_ms + transformer_ms + upsample_ms + init_conv_ms + output_ms
# Add decoder block times
print(f"\n  Total component time: ~{total_components:.0f}ms + decoder blocks")

# Also check: how does the decoder config look?
dc = decoder.config
print(f"\n=== Decoder config ===")
print(f"  upsample_rates: {dc.upsample_rates}")
print(f"  upsampling_ratios: {dc.upsampling_ratios}")
print(f"  decoder_dim: {dc.decoder_dim}")
print(f"  latent_dim: {dc.latent_dim}")
print(f"  codebook_dim: {dc.codebook_dim}")
print(f"  num_quantizers: {dc.num_quantizers}")
print(f"  Total upsample: {decoder.total_upsample}")
print(f"  Transformer layers: {dc.num_hidden_layers}")
print(f"  Transformer hidden: {dc.hidden_size}")
