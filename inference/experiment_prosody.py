"""Experiment: multi-sentence prosody continuity.

Problem: ivi sends each sentence individually to TTS, causing prosody jumps
at sentence boundaries. This tests three approaches to fix it:

1. Multi-sentence chunking: first sentence solo (fast TTFA), then group
   subsequent sentences together in single TTS calls.

2. Decoder state carry-over: DON'T call reset_streaming_state() between
   sentences. The codec decoder's conv buffers + transformer KV cache may
   carry temporal context forward, producing smoother transitions.

3. ICL audio prefill: feed the previous sentence's audio as reference audio
   for the next sentence, using Qwen3-TTS's built-in ICL mode.

All three produce WAV files in ~/Downloads/holler-prosody-test/ for A/B comparison.
"""
import os
import sys
import time
import wave

import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
OUT_DIR = os.path.expanduser("~/Downloads/holler-prosody-test")
SAMPLE_RATE = 24000
N_CODEBOOKS = 12
TEMPERATURE = 0.6
TOP_K = 50

TEST_PARAGRAPHS = [
    {
        "name": "casual_3sent",
        "sentences": [
            "I think the weather is going to be beautiful today.",
            "We should go for a walk in the park.",
            "Maybe grab some coffee on the way."
        ],
    },
    {
        "name": "explanation_4sent",
        "sentences": [
            "The way it works is actually pretty simple.",
            "You just speak naturally and the system transcribes everything.",
            "Then it processes the text and generates a response.",
            "The whole thing happens in under a second."
        ],
    },
    {
        "name": "emotional_3sent",
        "sentences": [
            "Oh that's really interesting!",
            "I didn't know you could do that.",
            "Tell me more about how it works."
        ],
    },
]

model = None
suppress_indices = None
zero_token = None


def init():
    global model, suppress_indices, zero_token
    print(f"[prosody] Loading model from {CHECKPOINT}...")
    t0 = time.time()
    model = load(CHECKPOINT)
    mx.set_cache_limit(2 * 1024 * 1024 * 1024)

    config = model.config.talker_config
    eos = config.codec_eos_token_id
    suppress_indices = mx.array(
        [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos],
        dtype=mx.int32,
    )
    zero_token = mx.zeros((1, 1), dtype=mx.int32)

    # warmup
    for _ in generate_audio(model, "Hello."):
        pass
    for _ in generate_audio(model, "Testing warmup sentence."):
        pass
    print(f"[prosody] Ready in {time.time()-t0:.1f}s")


def generate_audio(mdl, text, voice="katie", language="english",
                   temperature=TEMPERATURE, top_k=TOP_K, max_tokens=500,
                   reset_decoder=True):
    """Generate speech, yielding float32 audio chunks.

    If reset_decoder=False, decoder streaming state carries over from the
    previous call (experiment 2).
    """
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(N_CODEBOOKS - 1, num_code_groups - 1)

    word_count = len(text.split())
    safe_max = min(max_tokens, max(50, word_count * 20))

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language=language, speaker=voice
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

    cache = mdl.talker.make_cache()
    code_cache = mdl.talker.code_predictor.make_cache()
    get_input_emb = mdl.talker.get_input_embeddings()
    code_pred = mdl.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]
    generated_codes = []
    decoded_up_to = 0

    if reset_decoder:
        mdl.speech_tokenizer.decoder.reset_streaming_state()

    for step in range(safe_max):
        logits, hidden = mdl.talker(input_embeds, cache=cache)

        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(
            first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1
        )
        if top_k > 0:
            top_k_vals = mx.sort(first_logits, axis=-1)[:, -top_k:]
            threshold = top_k_vals[:, 0:1]
            first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
        next_token = categorical_sampling(first_logits, temperature)[:, None]

        is_eos = next_token[0, 0] == eos_token_id

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0

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
            code_tokens.append(zero_token)

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

        # Decode in chunks of 40
        n_new = len(generated_codes) - decoded_up_to
        if n_new >= 40:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)

            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = wav.squeeze(1)[0]
            mx.eval(audio_chunk)

            decoded_up_to = len(generated_codes)
            yield np.array(audio_chunk).flatten().astype(np.float32)

    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)

        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = wav.squeeze(1)[0]
        mx.eval(audio_chunk)

        yield np.array(audio_chunk).flatten().astype(np.float32)

    mx.clear_cache()


def generate_audio_full(mdl, text, **kwargs):
    """Generate and return all audio as a single numpy array."""
    chunks = list(generate_audio(mdl, text, **kwargs))
    if chunks:
        return np.concatenate(chunks)
    return np.array([], dtype=np.float32)


def save_wav(audio, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    dur = len(audio) / SAMPLE_RATE
    print(f"  → {path} ({dur:.1f}s)")


def add_silence(duration_s=0.15):
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


# ─── Experiment 0: Baseline (per-sentence, current behavior) ───

def exp0_baseline(paragraph):
    """Current behavior: each sentence generated independently, concatenated."""
    name = paragraph["name"]
    sentences = paragraph["sentences"]
    print(f"\n[exp0] Baseline — {name}")

    all_audio = []
    for i, sent in enumerate(sentences):
        t0 = time.time()
        audio = generate_audio_full(model, sent, reset_decoder=True)
        elapsed = time.time() - t0
        dur = len(audio) / SAMPLE_RATE
        print(f"  sent {i}: {elapsed*1000:.0f}ms, {dur:.1f}s audio — {sent[:50]}")
        if all_audio:
            all_audio.append(add_silence(0.05))
        all_audio.append(audio)

    combined = np.concatenate(all_audio)
    save_wav(combined, f"{OUT_DIR}/{name}_0_baseline.wav")
    return combined


# ─── Experiment 1: Multi-sentence chunking ───

def exp1_multi_sentence(paragraph):
    """First sentence solo, remaining sentences grouped into one TTS call."""
    name = paragraph["name"]
    sentences = paragraph["sentences"]
    print(f"\n[exp1] Multi-sentence chunking — {name}")

    # First sentence alone (fast TTFA)
    t0 = time.time()
    first_audio = generate_audio_full(model, sentences[0], reset_decoder=True)
    elapsed = time.time() - t0
    dur = len(first_audio) / SAMPLE_RATE
    print(f"  first: {elapsed*1000:.0f}ms, {dur:.1f}s audio — {sentences[0][:50]}")

    # Remaining sentences as one block
    remaining = " ".join(sentences[1:])
    t0 = time.time()
    rest_audio = generate_audio_full(model, remaining, reset_decoder=True)
    elapsed = time.time() - t0
    dur = len(rest_audio) / SAMPLE_RATE
    print(f"  rest:  {elapsed*1000:.0f}ms, {dur:.1f}s audio — {remaining[:50]}")

    combined = np.concatenate([first_audio, add_silence(0.05), rest_audio])
    save_wav(combined, f"{OUT_DIR}/{name}_1_multisent.wav")

    # Also try: all sentences as one block (reference for best possible prosody)
    full_text = " ".join(sentences)
    t0 = time.time()
    full_audio = generate_audio_full(model, full_text, reset_decoder=True)
    elapsed = time.time() - t0
    dur = len(full_audio) / SAMPLE_RATE
    print(f"  full:  {elapsed*1000:.0f}ms, {dur:.1f}s audio (all-at-once reference)")
    save_wav(full_audio, f"{OUT_DIR}/{name}_1_fulltext.wav")

    return combined


# ─── Experiment 2: Decoder state carry-over ───

def exp2_decoder_carryover(paragraph):
    """Don't reset decoder streaming state between sentences."""
    name = paragraph["name"]
    sentences = paragraph["sentences"]
    print(f"\n[exp2] Decoder state carry-over — {name}")

    all_audio = []
    model.speech_tokenizer.decoder.reset_streaming_state()

    for i, sent in enumerate(sentences):
        t0 = time.time()
        # Only reset decoder for the first sentence
        audio = generate_audio_full(model, sent, reset_decoder=(i == 0))
        elapsed = time.time() - t0
        dur = len(audio) / SAMPLE_RATE
        print(f"  sent {i}: {elapsed*1000:.0f}ms, {dur:.1f}s audio — {sent[:50]}")
        if all_audio:
            all_audio.append(add_silence(0.05))
        all_audio.append(audio)

    combined = np.concatenate(all_audio)
    save_wav(combined, f"{OUT_DIR}/{name}_2_carryover.wav")
    return combined


# ─── Experiment 3: ICL audio prefill ───

def exp3_icl_prefill(paragraph):
    """Feed previous sentence's audio as reference for the next sentence."""
    name = paragraph["name"]
    sentences = paragraph["sentences"]
    print(f"\n[exp3] ICL audio prefill — {name}")

    all_audio = []

    # First sentence: normal generation
    t0 = time.time()
    prev_audio = generate_audio_full(model, sentences[0], reset_decoder=True)
    elapsed = time.time() - t0
    dur = len(prev_audio) / SAMPLE_RATE
    print(f"  sent 0: {elapsed*1000:.0f}ms, {dur:.1f}s audio (normal) — {sentences[0][:50]}")
    all_audio.append(prev_audio)
    prev_text = sentences[0]

    # Subsequent sentences: ICL with previous sentence as reference
    for i in range(1, len(sentences)):
        sent = sentences[i]
        t0 = time.time()

        try:
            ref_audio_mx = mx.array(prev_audio)[None, None, :]  # [1, 1, samples]

            input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes = \
                model._prepare_icl_generation_inputs(
                    text=sent,
                    ref_audio=ref_audio_mx.squeeze(),
                    ref_text=prev_text,
                    language="english",
                )
            mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

            # Generate with ICL inputs using our custom loop
            audio = _generate_from_embeds(
                model, input_embeds, trailing_text_hidden, tts_pad_embed
            )
            elapsed = time.time() - t0
            dur = len(audio) / SAMPLE_RATE
            print(f"  sent {i}: {elapsed*1000:.0f}ms, {dur:.1f}s audio (ICL) — {sent[:50]}")

        except Exception as e:
            print(f"  sent {i}: ICL FAILED ({e}), falling back to normal")
            audio = generate_audio_full(model, sent, reset_decoder=True)
            elapsed = time.time() - t0
            dur = len(audio) / SAMPLE_RATE
            print(f"  sent {i}: {elapsed*1000:.0f}ms, {dur:.1f}s audio (fallback) — {sent[:50]}")

        all_audio.append(add_silence(0.05))
        all_audio.append(audio)
        prev_audio = audio
        prev_text = sent

    combined = np.concatenate(all_audio)
    save_wav(combined, f"{OUT_DIR}/{name}_3_icl.wav")
    return combined


def _generate_from_embeds(mdl, input_embeds, trailing_text_hidden, tts_pad_embed,
                          temperature=TEMPERATURE, top_k=TOP_K, max_tokens=500):
    """Run generation loop from pre-computed embeddings (for ICL mode)."""
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(N_CODEBOOKS - 1, num_code_groups - 1)

    cache = mdl.talker.make_cache()
    code_cache = mdl.talker.code_predictor.make_cache()
    get_input_emb = mdl.talker.get_input_embeddings()
    code_pred = mdl.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]
    generated_codes = []
    decoded_up_to = 0

    mdl.speech_tokenizer.decoder.reset_streaming_state()

    all_chunks = []

    for step in range(max_tokens):
        logits, hidden = mdl.talker(input_embeds, cache=cache)

        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(
            first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1
        )
        if top_k > 0:
            top_k_vals = mx.sort(first_logits, axis=-1)[:, -top_k:]
            threshold = top_k_vals[:, 0:1]
            first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
        next_token = categorical_sampling(first_logits, temperature)[:, None]

        is_eos = next_token[0, 0] == eos_token_id

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0

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
            code_tokens.append(zero_token)

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

        n_new = len(generated_codes) - decoded_up_to
        if n_new >= 40:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)

            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = wav.squeeze(1)[0]
            mx.eval(audio_chunk)

            decoded_up_to = len(generated_codes)
            all_chunks.append(np.array(audio_chunk).flatten().astype(np.float32))

    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)

        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = wav.squeeze(1)[0]
        mx.eval(audio_chunk)

        all_chunks.append(np.array(audio_chunk).flatten().astype(np.float32))

    mx.clear_cache()

    if all_chunks:
        return np.concatenate(all_chunks)
    return np.array([], dtype=np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init()

    for paragraph in TEST_PARAGRAPHS:
        print(f"\n{'='*70}")
        print(f"Paragraph: {paragraph['name']}")
        print(f"{'='*70}")

        exp0_baseline(paragraph)
        exp1_multi_sentence(paragraph)
        exp2_decoder_carryover(paragraph)
        exp3_icl_prefill(paragraph)

    print(f"\n{'='*70}")
    print(f"All outputs in: {OUT_DIR}")
    print(f"Compare:")
    print(f"  *_0_baseline.wav  — current behavior (per-sentence)")
    print(f"  *_1_multisent.wav — first solo + rest grouped")
    print(f"  *_1_fulltext.wav  — all text in one call (gold standard)")
    print(f"  *_2_carryover.wav — decoder state carried across sentences")
    print(f"  *_3_icl.wav       — ICL: previous audio as reference")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
