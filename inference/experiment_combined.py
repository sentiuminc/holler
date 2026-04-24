"""Combined optimizations experiment.
Tests: cache limit + fewer codebooks + custom generate loop together."""
import time
import os
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.expanduser("~/Downloads/holler-combined-experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def fast_generate_ncb(mdl, text, voice="katie", language="english", temperature=0.6,
                      top_k=50, max_tokens=500, n_codebooks=16):
    """Fast generate with configurable number of codebooks."""
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
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

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


print("Loading model...")
model = load(CHECKPOINT)
# Warmup
fast_generate_ncb(model, "Hello.", voice="katie")
fast_generate_ncb(model, "Testing warmup.", voice="katie")
print("Warmup done.\n")

# Set optimal cache limit
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

# Test each codebook count across all sentences
for n_cb in [16, 12, 8]:
    print(f"{'='*90}")
    print(f"Codebooks: {n_cb}, cache_limit=2GB")
    print(f"{'='*90}")
    print(f"{'Words':>5} {'Tokens':>6} {'GenMs':>7} {'DecMs':>7} {'Audio':>6} {'GenRTF':>7} {'TotRTF':>7}  Text")
    print("-" * 90)

    gen_rtfs = []
    total_rtfs = []

    for i, text in enumerate(sentences):
        audio, n_tok, gen_ms, dec_ms = fast_generate_ncb(model, text, n_codebooks=n_cb)
        audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
        total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s if audio_s > 0 else 999
        words = len(text.split())

        gen_rtfs.append(gen_rtf)
        total_rtfs.append(total_rtf)

        # Save first run of each config
        fname = os.path.join(OUTPUT_DIR, f"cb{n_cb}_sent{i:02d}.wav")
        if len(audio) > 0:
            sf.write(fname, audio, SAMPLE_RATE)

        print(f"{words:>5} {n_tok:>6} {gen_ms:>6.0f}ms {dec_ms:>6.0f}ms {audio_s:>5.1f}s {gen_rtf:>6.3f} {total_rtf:>6.3f}  {text[:40]}")

    avg_gen = sum(gen_rtfs) / len(gen_rtfs)
    avg_total = sum(total_rtfs) / len(total_rtfs)
    max_gen = max(gen_rtfs)
    max_total = max(total_rtfs)

    target = "✅" if avg_total <= 0.50 and max_total <= 0.70 else "❌"
    print(f"\n  Gen-only:  avg={avg_gen:.3f} max={max_gen:.3f}")
    print(f"  Gen+Dec:   avg={avg_total:.3f} max={max_total:.3f} {target}")
    print()

print(f"Audio samples saved to {OUTPUT_DIR}")
print("Compare: cb16_sent04.wav vs cb12_sent04.wav vs cb8_sent04.wav for quality check")
