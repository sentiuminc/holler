"""Compare decode strategies for the server:
1. Generate all → decode all (minimum total time but maximum TTFA)
2. Generate N → decode N → stream → repeat (current server approach)
3. Two-phase: small first chunk for TTFA, then larger chunks

The question: what's the optimal chunk size for streaming?"""
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

# Warmup
for r in model.generate(text="Hello.", voice="katie", language="english", temperature=0.6, stream=False):
    pass

text = "Something about the umami thing appeals to me."

# Strategy 1: Generate all → decode all
def gen_then_decode(mdl, text):
    from fast_generate import fast_generate
    audio, n, gen_ms, dec_ms = fast_generate(mdl, text, voice="katie")
    return audio, n, gen_ms, dec_ms

# Strategy 2: Interleaved with various chunk sizes
def gen_interleaved(mdl, text, first_chunk=5, chunk_size=25, n_codebooks=12):
    """Generate tokens, decode in chunks using streaming_step."""
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language="english", speaker="katie"
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
    audio_chunks = []

    word_count = len(text.split())
    safe_max = min(500, max(50, word_count * 20))

    mdl.speech_tokenizer.decoder.reset_streaming_state()
    t_start = time.time()
    ttfa = None

    for step in range(safe_max):
        logits, hidden = mdl.talker(input_embeds, cache=cache)
        fl = logits[:, -1, :]
        fl = mx.put_along_axis(fl, suppress_indices[None, :], mx.array(float("-inf"), fl.dtype), axis=-1)
        tkv = mx.sort(fl, axis=-1)[:, -50:]
        thr = tkv[:, 0:1]
        fl = mx.where(fl < thr, float("-inf"), fl)
        nt = categorical_sampling(fl, 0.6)[:, None]
        is_eos = nt[0, 0] == eos_token_id

        code_tokens = [nt]
        ch = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None; c.values = None; c.offset = 0
        if actual_extra > 0:
            c0e = get_input_emb(nt)
            ci = mx.concatenate([ch, c0e], axis=1)
            cl, code_cache, _ = code_pred(ci, cache=code_cache, generation_step=0)
            ct = categorical_sampling(cl[:, -1, :], 0.6)[:, None]
            code_tokens.append(ct)
            for cix in range(1, actual_extra):
                ci = code_embeddings[cix - 1](code_tokens[-1])
                cl, code_cache, _ = code_pred(ci, cache=code_cache, generation_step=cix)
                ct = categorical_sampling(cl[:, -1, :], 0.6)[:, None]
                code_tokens.append(ct)
        while len(code_tokens) < num_code_groups:
            code_tokens.append(mx.zeros_like(nt))
        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_len:
            te = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :]
            trailing_idx += 1
        else:
            te = tts_pad_embed
        ce = get_input_emb(nt)
        for i in range(min(actual_extra, num_code_groups - 1)):
            ce = ce + code_embeddings[i](code_tokens[i + 1])
        input_embeds = te + ce
        mx.eval(input_embeds, is_eos)
        if is_eos.item():
            break
        generated_codes.append(all_codes)

        n_new = len(generated_codes) - decoded_up_to
        threshold = first_chunk if decoded_up_to == 0 else chunk_size
        if n_new >= threshold:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)
            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = wav.squeeze(1)[0]
            mx.eval(audio_chunk)
            audio_chunks.append(np.array(audio_chunk).flatten().astype(np.float32))
            decoded_up_to = len(generated_codes)
            if ttfa is None:
                ttfa = (time.time() - t_start) * 1000

    # Decode remaining
    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)
        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = wav.squeeze(1)[0]
        mx.eval(audio_chunk)
        audio_chunks.append(np.array(audio_chunk).flatten().astype(np.float32))
        if ttfa is None:
            ttfa = (time.time() - t_start) * 1000

    total_ms = (time.time() - t_start) * 1000
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
    else:
        audio = np.array([], dtype=np.float32)
    return audio, len(generated_codes), total_ms, ttfa

# Test all strategies
texts = [
    "Hey!",
    "Something about the umami thing appeals to me.",
    "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
]

print("=== Strategy comparison ===\n")
print(f"{'Strategy':<40} {'TTFA':>6} {'Total':>7} {'Audio':>6} {'RTF':>6}")
print("-" * 70)

for text in texts:
    print(f"\n  \"{text[:55]}\"")

    # Gen-then-decode (16cb)
    audio, n, gen_ms, dec_ms = gen_then_decode(model, text)
    audio_s = len(audio) / 24000
    rtf = ((gen_ms + dec_ms) / 1000) / audio_s
    print(f"  {'gen-then-decode 16cb':<40} {'N/A':>6} {gen_ms+dec_ms:>6.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}")

    # Interleaved, various chunk sizes (12cb)
    for fc, cs in [(3, 15), (5, 25), (5, 40), (10, 50)]:
        audio, n, total_ms, ttfa = gen_interleaved(model, text, first_chunk=fc, chunk_size=cs)
        audio_s = len(audio) / 24000
        rtf = (total_ms / 1000) / audio_s
        label = f"interleaved 12cb fc={fc} cs={cs}"
        print(f"  {label:<40} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}")

    # Interleaved, gen-all-then-decode equivalent (huge chunk)
    audio, n, total_ms, ttfa = gen_interleaved(model, text, first_chunk=999, chunk_size=999)
    audio_s = len(audio) / 24000
    rtf = (total_ms / 1000) / audio_s
    print(f"  {'interleaved 12cb gen-all':<40} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}")
