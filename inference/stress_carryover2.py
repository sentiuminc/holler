#!/usr/bin/env python3
"""Stress test v2: full logging for stutter diagnosis.

Generates 8-sentence chains with carryover, logs all internal state
per sentence so we can correlate stutters with cache/decoder state.

Usage:
    .venv/bin/python inference/stress_carryover2.py
"""
import os
import random
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as srv

SR = 24000

SHORT = ['Hey!', 'Yes.', 'Got it.', 'No.', 'Sure.', 'OK.', 'Right.',
         'Hmm.', 'Wait.', 'Yeah.', 'Thanks.', 'Wow.', 'Oh.', 'Nope.', 'Cool.']
LONG = [
    'What do you think about that?',
    'I completely agree with your assessment on this.',
    'Let me explain what I mean by that.',
    'That is not what I was trying to say at all.',
    'Could you tell me a little more about that?',
    'Keep going, I want to hear the rest of it.',
    'The problem here is really quite clear to me.',
    'But I think we actually need to reconsider this.',
    'Actually, now that I think about it more carefully.',
    'I just realized something really important about this.',
    'Pretty impressive work you have done here.',
    'Basically, the whole approach needs rethinking.',
    'Tell me what you think about this idea.',
    'Going back to your original question about it.',
    'Did you know that was even possible to do?',
    'The key insight is that simplicity always wins.',
    'People tend to overthink these kinds of problems.',
    'I believe the answer is simpler than we think.',
    'Consider what would happen if we tried it differently.',
    'That reminds me of something I read recently.',
]


def save_wav(path, audio):
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())


def gen(model, text, voice, reset):
    chunks = list(srv.generate_audio(model, text, voice=voice, reset_decoder=reset))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def cache_offset():
    cache = srv._carry_over_state.get("talker_cache")
    if cache and len(cache) > 0:
        return cache[0].offset
    return 0


def main():
    import mlx.core as mx
    from mlx_audio.tts import load

    checkpoint = "checkpoints/holler-kit-dakota-6bit"
    out = os.path.expanduser("~/Downloads/stutter-debug")

    model = load(checkpoint)
    mx.set_cache_limit(2 * 1024**3)

    config = model.config.talker_config
    eos = config.codec_eos_token_id
    srv.suppress_indices_cache = mx.array(
        [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos],
        dtype=mx.int32,
    )
    srv.zero_token_cache = mx.zeros((1, 1), dtype=mx.int32)

    # Warmup
    for _ in srv.generate_audio(model, "Hello.", voice="kit", max_tokens=50):
        pass
    model.speech_tokenizer.decoder.reset_streaming_state()
    mx.clear_cache()

    file_idx = 0

    for voice in ["kit", "dakota"]:
        print(f"\n{'='*70}", flush=True)
        print(f"  VOICE: {voice}", flush=True)
        print(f"{'='*70}", flush=True)

        for seq in range(30):
            texts = [random.choice(SHORT)]
            texts += [random.choice(LONG) for _ in range(7)]

            srv._carry_over_state = {}
            model.speech_tokenizer.decoder.reset_streaming_state()
            mx.clear_cache()

            print(f"\n--- seq {seq:02d} ({voice}) ---", flush=True)

            audios = []
            ok = True

            for i, text in enumerate(texts):
                reset = (i == 0)
                pre_offset = cache_offset()

                t0 = time.time()
                audio = gen(model, text, voice, reset=reset)
                elapsed = (time.time() - t0) * 1000

                post_offset = cache_offset()

                if len(audio) < 100:
                    print(f"  S{i+1}: EMPTY  \"{text[:50]}\"", flush=True)
                    ok = False
                    break

                dur = len(audio) / SR
                tokens_added = post_offset - pre_offset
                carryover = "fresh" if reset else f"carry(pre={pre_offset},post={post_offset},+{tokens_added})"

                print(
                    f"  S{i+1}: {dur:.2f}s  {elapsed:.0f}ms  {carryover}  \"{text[:55]}\"",
                    flush=True,
                )
                audios.append((text, audio))

            if not ok or len(audios) < 5:
                continue

            # Save sentences 5-8 (deep carryover)
            for s_idx in range(4, len(audios)):
                file_idx += 1
                pause = np.zeros(int(SR * 0.15), dtype=np.float32)
                parts = []
                for j in range(s_idx + 1):
                    if j > 0:
                        parts.append(pause)
                    parts.append(audios[j][1])
                combined = np.concatenate(parts)

                fname = f"{file_idx:03d}_{voice}_seq{seq:02d}_s{s_idx+1}.wav"
                save_wav(f"{out}/{fname}", combined)

        print(f"\n  [{voice}] Done: {file_idx} files total", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  Total: {file_idx} files in {out}/", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
