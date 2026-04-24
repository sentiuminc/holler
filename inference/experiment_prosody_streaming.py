"""Experiment: decoder carry-over with streaming chunks.

Tests that decoder carry-over works correctly with the real streaming decode
pattern (first_chunk=3, stream_chunk=40) — exactly matching tts-sidecar-fast.py.

Also measures:
- TTFA per sentence (should stay ~110ms)
- RTF per sentence
- RAM usage stability across sentences
- Whether longer sequences (8+ sentences) accumulate state safely
"""
import os
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
FIRST_CHUNK_TOKENS = 3
STREAM_CHUNK_TOKENS = 40
TEMPERATURE = 0.6
TOP_K = 50

LONG_PARAGRAPH = [
    "Hey, so I've been thinking about this for a while.",
    "The thing is, most voice assistants sound really robotic when they string sentences together.",
    "You can hear the cuts between each sentence, like someone spliced audio clips together.",
    "But what if we could make the transitions completely seamless?",
    "Like, imagine having a conversation where the response just flows naturally.",
    "No weird pauses, no tonal shifts, no uncanny valley.",
    "That's exactly what we're building here.",
    "And honestly, I think we're really close to nailing it.",
]

SHORT_EXCHANGES = [
    "Yeah.",
    "Exactly.",
    "That's what I thought too.",
    "Makes sense.",
    "Got it, let's do that.",
]

model = None
suppress_indices = None
zero_token = None


def init():
    global model, suppress_indices, zero_token
    print(f"[streaming] Loading model...")
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

    for _ in generate_streaming(model, "Hello."):
        pass
    for _ in generate_streaming(model, "Warmup."):
        pass
    print(f"[streaming] Ready in {time.time()-t0:.1f}s\n")


def generate_streaming(mdl, text, voice="katie", language="english",
                       temperature=TEMPERATURE, top_k=TOP_K, max_tokens=500,
                       reset_decoder=True):
    """Generate speech with real streaming chunk pattern.

    Yields float32 audio chunks:
    - First chunk after FIRST_CHUNK_TOKENS codec tokens (~110ms TTFA)
    - Subsequent chunks every STREAM_CHUNK_TOKENS tokens
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

        n_new = len(generated_codes) - decoded_up_to
        thresh = FIRST_CHUNK_TOKENS if decoded_up_to == 0 else STREAM_CHUNK_TOKENS
        if n_new >= thresh:
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


def save_wav(audio, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())


def get_metal_ram():
    try:
        active = mx.metal.get_active_memory() / 1024 / 1024
        peak = mx.metal.get_peak_memory() / 1024 / 1024
        return active, peak
    except Exception:
        return 0, 0


def run_paragraph(name, sentences, reset_between=False):
    """Generate a paragraph sentence-by-sentence with streaming metrics."""
    print(f"\n{'='*70}")
    print(f"  {name} — {'RESET' if reset_between else 'CARRY-OVER'} — {len(sentences)} sentences")
    print(f"{'='*70}")

    all_audio = []
    total_gen_ms = 0

    active0, _ = get_metal_ram()

    for i, sent in enumerate(sentences):
        t0 = time.time()
        ttfa = None
        chunks = []
        n_chunks = 0

        for audio_chunk in generate_streaming(
            model, sent,
            reset_decoder=(reset_between or i == 0),
        ):
            if len(audio_chunk) > 0:
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                chunks.append(audio_chunk)
                n_chunks += 1

        elapsed = (time.time() - t0) * 1000
        total_gen_ms += elapsed

        if chunks:
            sentence_audio = np.concatenate(chunks)
        else:
            sentence_audio = np.array([], dtype=np.float32)

        dur = len(sentence_audio) / SAMPLE_RATE
        rtf = (elapsed / 1000) / dur if dur > 0 else 999
        active, peak = get_metal_ram()

        print(f"  [{i}] TTFA={ttfa:.0f}ms  total={elapsed:.0f}ms  "
              f"audio={dur:.1f}s  RTF={rtf:.3f}  chunks={n_chunks}  "
              f"RAM={active:.0f}MB  | {sent[:55]}")

        all_audio.append(sentence_audio)

    combined = np.concatenate(all_audio)
    total_dur = len(combined) / SAMPLE_RATE
    overall_rtf = (total_gen_ms / 1000) / total_dur if total_dur > 0 else 999

    suffix = "reset" if reset_between else "carry"
    path = f"{OUT_DIR}/{name}_{suffix}.wav"
    save_wav(combined, path)

    active_end, peak_end = get_metal_ram()
    print(f"\n  Total: {total_gen_ms:.0f}ms gen → {total_dur:.1f}s audio, "
          f"RTF={overall_rtf:.3f}")
    print(f"  RAM: start={active0:.0f}MB → end={active_end:.0f}MB, peak={peak_end:.0f}MB")
    print(f"  → {path}")

    return combined


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    init()

    # Test 1: Long paragraph — baseline vs carry-over
    run_paragraph("long_baseline", LONG_PARAGRAPH, reset_between=True)
    run_paragraph("long_carryover", LONG_PARAGRAPH, reset_between=False)

    # Test 2: Short back-and-forth exchanges
    run_paragraph("short_baseline", SHORT_EXCHANGES, reset_between=True)
    run_paragraph("short_carryover", SHORT_EXCHANGES, reset_between=False)

    # Test 3: Stress test — repeat 8 sentences twice (16 total) to check state accumulation
    stress = LONG_PARAGRAPH + LONG_PARAGRAPH
    run_paragraph("stress_carryover", stress, reset_between=False)

    print(f"\n{'='*70}")
    print(f"All outputs in: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
