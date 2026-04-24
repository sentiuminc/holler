"""Experiment: Greedy (argmax) decoding for code predictor sub-codebooks.

Hypothesis: The code predictor generates acoustic detail tokens. Unlike the main
talker which benefits from sampling for prosody diversity, the code predictor's
tokens reconstruct waveform detail. Greedy decoding should:
1. Be slightly faster (no random number generation)
2. Produce more consistent audio (deterministic reconstruction)
3. Possibly sound slightly different (less acoustic noise)

Saves audio samples for quality comparison."""
import time
import os
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
OUTPUT_DIR = os.path.expanduser("~/Downloads/holler-greedy-codepred")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 24000

model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)


def generate_with_greedy_codepred(mdl, text, voice="katie", language="english",
                                   temperature=0.6, top_k=50, max_tokens=500,
                                   n_codebooks=12):
    """Main token sampled (temp=0.6), code predictor tokens greedy (argmax)."""
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

    word_count = len(text.split())
    safe_max = min(max_tokens, max(50, word_count * 20))

    t_gen = time.time()

    for step in range(safe_max):
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
            # GREEDY for code predictor
            ct = mx.argmax(code_logits[:, -1, :], axis=-1)[:, None]
            code_tokens.append(ct)

            for code_idx in range(1, actual_extra):
                code_input = code_embeddings[code_idx - 1](code_tokens[-1])
                code_logits, code_cache, _ = code_pred(code_input, cache=code_cache, generation_step=code_idx)
                ct = mx.argmax(code_logits[:, -1, :], axis=-1)[:, None]
                code_tokens.append(ct)

        while len(code_tokens) < num_code_groups:
            code_tokens.append(mx.zeros((1, 1), dtype=mx.int32))

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


# Warmup
generate_with_greedy_codepred(model, "Hello.")
print("Warmup done.\n")

texts = [
    "Hey, how are you doing today?",
    "Something about the umami thing really appeals to me.",
    "Hold on, speak a couple sentences, release, and let me know if the gap is gone.",
    "I think we should probably head out soon, it looks like it might rain later.",
    "The server was returning a five hundred error because the database connection pool was exhausted.",
]

# Compare sampled vs greedy code predictor
for mode_name, gen_fn in [
    ("sampled codepred", None),
    ("greedy codepred", generate_with_greedy_codepred),
]:
    print(f"=== {mode_name} (12 codebooks) ===")
    print(f"{'Text':<50} {'GenMs':>7} {'Audio':>6} {'GenRTF':>7} {'TotRTF':>7}")
    print("-" * 85)

    gen_rtfs = []
    for i, text in enumerate(texts):
        if gen_fn:
            audio, n, gen_ms, dec_ms = gen_fn(model, text, n_codebooks=12)
        else:
            # Use the existing fast_generate_ncb for sampled mode
            from experiment_combined import fast_generate_ncb
            audio, n, gen_ms, dec_ms = fast_generate_ncb(model, text, n_codebooks=12)

        audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
        total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999
        gen_rtfs.append(gen_rtf)

        # Save audio
        tag = "greedy" if gen_fn else "sampled"
        fname = os.path.join(OUTPUT_DIR, f"{tag}_text{i}.wav")
        if len(audio) > 0:
            sf.write(fname, audio, SAMPLE_RATE)

        print(f"{text[:49]:<50} {gen_ms:>6.0f}ms {audio_s:>5.1f}s {gen_rtf:>6.3f} {total_rtf:>6.3f}")

    print(f"  Avg gen RTF: {sum(gen_rtfs)/len(gen_rtfs):.3f}\n")

print(f"Audio comparison saved to {OUTPUT_DIR}")
print("Compare sampled_text*.wav vs greedy_text*.wav for quality difference")
