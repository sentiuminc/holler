"""Experiment: Hann crossfade between streaming decode chunks.

When decoding in chunks, each chunk boundary can have a tiny discontinuity.
Hann crossfade overlaps adjacent chunks by N samples, blending with a
cosine-shaped window. Zero performance cost (pure numpy, outside hot loop).

Generates A/B samples: with and without crossfade, various overlap sizes.
"""
import time
import os
import json
import urllib.request
import wave
import io
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.expanduser("~/Downloads/holler-hann-crossfade")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)


def generate_chunks(mdl, text, voice="katie", temperature=0.6, n_codebooks=12,
                    first_chunk_tokens=3, stream_chunk_tokens=40):
    """Generate speech, returning list of raw audio chunks (before any crossfade)."""
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

    word_count = len(text.split())
    safe_max = min(500, max(50, word_count * 20))

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language="english", speaker=voice
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
    chunks = []

    mdl.speech_tokenizer.decoder.reset_streaming_state()

    for step in range(safe_max):
        logits, hidden = mdl.talker(input_embeds, cache=cache)
        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1)
        top_k_vals = mx.sort(first_logits, axis=-1)[:, -50:]
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

        n_new = len(generated_codes) - decoded_up_to
        thresh = first_chunk_tokens if decoded_up_to == 0 else stream_chunk_tokens
        if n_new >= thresh:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)
            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = np.array(wav.squeeze(1)[0]).flatten().astype(np.float32)
            mx.eval(wav)
            chunks.append(audio_chunk)
            decoded_up_to = len(generated_codes)

    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)
        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = np.array(wav.squeeze(1)[0]).flatten().astype(np.float32)
        mx.eval(wav)
        chunks.append(audio_chunk)

    mx.clear_cache()
    return chunks


def concat_no_crossfade(chunks):
    """Simple concatenation — no crossfade."""
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def concat_hann_crossfade(chunks, overlap_samples=480):
    """Concatenate chunks with Hann window crossfade at boundaries.

    overlap_samples: number of samples to overlap (480 = 20ms at 24kHz)
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    # Build Hann windows for fade-out and fade-in
    window = np.hanning(overlap_samples * 2)
    fade_out = window[:overlap_samples]  # first half: 0 → 1 → peak
    fade_in = window[overlap_samples:]   # second half: peak → 1 → 0
    # Actually for crossfade: fade_out should go 1→0, fade_in should go 0→1
    fade_out = fade_out[::-1]  # now goes 1 → 0
    # fade_in already goes 0 → 1 (second half of hann is symmetric)
    # Wait, let me think about this more carefully.
    # np.hanning(N) produces [0, ..., 1, ..., 0] (symmetric)
    # For overlap of size L, we want:
    #   fade_out: the tail of chunk N, going from 1 → 0
    #   fade_in:  the head of chunk N+1, going from 0 → 1
    # Use the second half of hanning(2*L) for fade_out (1 → 0)
    # Use the first half of hanning(2*L) for fade_in (0 → 1)
    fade_in_window = window[:overlap_samples]    # 0 → 1
    fade_out_window = window[overlap_samples:]   # 1 → 0

    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            if len(chunk) > overlap_samples:
                # Keep everything except apply fade-out to the last overlap_samples
                result.append(chunk[:-overlap_samples])
                result.append(chunk[-overlap_samples:] * fade_out_window)
            else:
                result.append(chunk)
        elif i == len(chunks) - 1:
            if len(chunk) > overlap_samples:
                # Apply fade-in to first overlap_samples, blend with previous fade-out
                blended = chunk[:overlap_samples] * fade_in_window
                # Add to the fade-out tail (last element in result)
                if len(result) > 0:
                    prev_tail = result.pop()
                    blended = prev_tail + blended
                result.append(blended)
                result.append(chunk[overlap_samples:])
            else:
                result.append(chunk)
        else:
            if len(chunk) > overlap_samples * 2:
                # Blend start with previous fade-out
                blended = chunk[:overlap_samples] * fade_in_window
                if len(result) > 0:
                    prev_tail = result.pop()
                    blended = prev_tail + blended
                result.append(blended)
                # Middle (untouched)
                result.append(chunk[overlap_samples:-overlap_samples])
                # Fade-out tail for next chunk
                result.append(chunk[-overlap_samples:] * fade_out_window)
            else:
                # Chunk too small for double overlap, just blend start
                blended = chunk[:overlap_samples] * fade_in_window
                if len(result) > 0:
                    prev_tail = result.pop()
                    blended = prev_tail + blended
                result.append(blended)
                if len(chunk) > overlap_samples:
                    result.append(chunk[overlap_samples:])

    return np.concatenate(result)


def save_wav(path, audio):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    with open(path, 'wb') as f:
        f.write(buf.getvalue())


# Warmup
generate_chunks(model, "Hello.")
print("Warmup done.\n")

sentences = [
    "Welcome back, how did you sleep?",
    "Oh nice, I love that for you.",
    "The best tacos in Tbilisi are actually at that place near Vera.",
    "Honestly, I think the simplest approach is usually the right one.",
    "The pull request looks good, but there is one edge case we should handle before merging.",
    "That is a really interesting way to think about it, I had not considered that angle before.",
]

print(f"Generating {len(sentences)} sentences, comparing crossfade variants...\n")

for i, text in enumerate(sentences):
    chunks = generate_chunks(model, text)
    n_chunks = len(chunks)
    chunk_sizes = [len(c) for c in chunks]

    # No crossfade
    audio_raw = concat_no_crossfade(chunks)
    save_wav(os.path.join(OUTPUT_DIR, f"s{i:02d}_no_crossfade.wav"), audio_raw)

    # Hann crossfade with various overlap sizes
    for overlap_ms in [10, 20, 40]:
        overlap_samples = int(overlap_ms * SAMPLE_RATE / 1000)
        audio_xfade = concat_hann_crossfade(chunks, overlap_samples=overlap_samples)
        save_wav(os.path.join(OUTPUT_DIR, f"s{i:02d}_hann_{overlap_ms}ms.wav"), audio_xfade)

    raw_dur = len(audio_raw) / SAMPLE_RATE
    print(f"  [{i}] {n_chunks} chunks, {raw_dur:.1f}s | \"{text[:55]}\"")
    print(f"       chunk sizes: {chunk_sizes}")

print(f"\nSaved to {OUTPUT_DIR}")
print("Compare:")
print("  s00_no_crossfade.wav  — raw chunk concatenation")
print("  s00_hann_10ms.wav     — 10ms Hann crossfade")
print("  s00_hann_20ms.wav     — 20ms Hann crossfade")
print("  s00_hann_40ms.wav     — 40ms Hann crossfade")
