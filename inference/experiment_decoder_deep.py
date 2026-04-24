"""Deep decoder component profiling — standalone, no other module imports."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

# Generate codes inline
text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."
config = model.config.talker_config
eos_token_id = config.codec_eos_token_id
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
generated_codes = []

# Warmup + generate
for _ in range(2):
    generated_codes = []
    trailing_idx = 0
    cache = model.talker.make_cache()

    ie = input_embeds
    for step in range(500):
        logits, hidden = model.talker(ie, cache=cache)
        fl = logits[:, -1, :]
        fl = mx.put_along_axis(fl, suppress_indices[None, :],
            mx.array(float("-inf"), fl.dtype), axis=-1)
        tkv = mx.sort(fl, axis=-1)[:, -50:]
        thr = tkv[:, 0:1]
        fl = mx.where(fl < thr, float("-inf"), fl)
        nt = categorical_sampling(fl, 0.6)[:, None]
        is_eos = nt[0, 0] == eos_token_id

        code_tokens = [nt]
        ch = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None; c.values = None; c.offset = 0
        c0e = get_input_emb(nt)
        ci = mx.concatenate([ch, c0e], axis=1)
        cl, code_cache, _ = code_pred(ci, cache=code_cache, generation_step=0)
        ct = categorical_sampling(cl[:, -1, :], 0.6)[:, None]
        code_tokens.append(ct)
        for cix in range(1, 11):
            ci = code_embeddings[cix - 1](code_tokens[-1])
            cl, code_cache, _ = code_pred(ci, cache=code_cache, generation_step=cix)
            ct = categorical_sampling(cl[:, -1, :], 0.6)[:, None]
            code_tokens.append(ct)
        while len(code_tokens) < 16:
            code_tokens.append(mx.zeros_like(nt))
        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_text_hidden.shape[1]:
            te = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
            trailing_idx += 1
        else:
            te = tts_pad_embed
        ce = get_input_emb(nt)
        for i in range(11):
            ce = ce + code_embeddings[i](code_tokens[i + 1])
        ie = te + ce
        mx.eval(ie, is_eos)
        if is_eos.item():
            break
        generated_codes.append(all_codes)

codes = mx.stack(generated_codes, axis=1)
n = codes.shape[1]
codes_t = mx.transpose(codes, (0, 2, 1))
mx.eval(codes_t)
print(f"Have {n} tokens for decode profiling\n")

decoder = model.speech_tokenizer.decoder

# Component profiling
print("=== Decoder Component Profiling ===")

# 1. Dequantize
t0 = time.time()
hidden = decoder.quantizer.decode(codes_t)
hidden = mx.transpose(hidden, (0, 2, 1))
mx.eval(hidden)
print(f"  1. Dequantize:     {(time.time()-t0)*1000:6.0f}ms  shape={hidden.shape}")

# 2. Pre-conv
t0 = time.time()
h2 = decoder.pre_conv(hidden)
mx.eval(h2)
print(f"  2. Pre-conv:       {(time.time()-t0)*1000:6.0f}ms  shape={h2.shape}")

# 3. Transformer
t0 = time.time()
h3 = decoder.pre_transformer(h2)
mx.eval(h3)
print(f"  3. Transformer:    {(time.time()-t0)*1000:6.0f}ms  shape={h3.shape}")

# 4. Upsample
t0 = time.time()
h4 = h3
for ul in decoder.upsample:
    for layer in ul:
        h4 = layer(h4)
mx.eval(h4)
print(f"  4. Upsample:       {(time.time()-t0)*1000:6.0f}ms  shape={h4.shape}")

# 5. Decoder blocks
t0 = time.time()
wav = decoder.decoder[0](h4)
mx.eval(wav)
print(f"  5a. InitConv:      {(time.time()-t0)*1000:6.0f}ms  shape={wav.shape}")

for bi in range(1, 5):
    t0 = time.time()
    wav = decoder.decoder[bi](wav)
    mx.eval(wav)
    # Get upsample rate for this block
    ur = decoder.config.upsample_rates[bi-1]
    print(f"  5{chr(97+bi)}. Block[{bi}] (up{ur}x): {(time.time()-t0)*1000:6.0f}ms  shape={wav.shape}")

t0 = time.time()
wav = decoder.decoder[5](wav)  # SnakeBeta
wav = decoder.decoder[6](wav)  # OutputConv
wav = mx.transpose(wav, (0, 2, 1))
wav = mx.clip(wav, -1.0, 1.0)
mx.eval(wav)
print(f"  5f. Output:        {(time.time()-t0)*1000:6.0f}ms  shape={wav.shape}")

# Full decode for comparison
print()
for trial in range(3):
    t0 = time.time()
    wav = decoder(codes_t)
    mx.eval(wav)
    ms = (time.time() - t0) * 1000
    audio_s = wav.shape[-1] / 24000
    print(f"  Full decode (trial {trial+1}): {ms:.0f}ms for {audio_s:.1f}s audio (RTF={ms/1000/audio_s:.4f})")

# Also test: streaming_step with full codes in one chunk (equivalent to direct but with streaming state)
print()
decoder.reset_streaming_state()
t0 = time.time()
wav = decoder.streaming_step(codes_t)
mx.eval(wav)
ms = (time.time() - t0) * 1000
audio_s = wav.shape[-1] / 24000
print(f"  streaming_step (1 chunk): {ms:.0f}ms for {audio_s:.1f}s audio (RTF={ms/1000/audio_s:.4f})")

# Decoder config
dc = decoder.config
print(f"\n=== Decoder config ===")
print(f"  upsample_rates: {dc.upsample_rates}")
print(f"  upsampling_ratios: {dc.upsampling_ratios}")
print(f"  decoder_dim: {dc.decoder_dim}")
print(f"  latent_dim: {dc.latent_dim}")
print(f"  codebook_dim: {dc.codebook_dim}")
print(f"  num_quantizers: {dc.num_quantizers}")
print(f"  Total upsample: {decoder.total_upsample}")
print(f"  Transformer: {dc.num_hidden_layers} layers, hidden={dc.hidden_size}")
print(f"  Tokens to samples: {n} tokens → {n * decoder.total_upsample} samples = {n * decoder.total_upsample / 24000:.1f}s")
