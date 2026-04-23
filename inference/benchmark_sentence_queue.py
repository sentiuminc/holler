"""
Sentence-queue benchmark for Holler via mlx-audio.
Simulates ivi's real pattern: LLM streams sentences → TTS generates each → play sequentially.
Measures TTFA, per-sentence timing, and whether generation stays ahead of playback.
"""
import time
import os
import soundfile as sf
import numpy as np
import mlx.core as mx

responses = [
    ("greeting", [
        "Hey!",
        "How's it going today?",
    ]),
    ("short_answer", [
        "Sure, let me check that for you.",
        "It looks like your meeting got moved to three PM.",
        "Want me to update your calendar?",
    ]),
    ("explanation", [
        "Great question.",
        "The weather today is going to be mostly sunny with a high of seventy-two.",
        "There's a slight chance of rain later this evening, so you might want to grab an umbrella just in case.",
        "Tomorrow looks even better though.",
    ]),
    ("technical", [
        "Okay.",
        "I found the issue.",
        "The server was returning a five hundred error because the database connection pool was exhausted.",
        "I've increased the pool size from ten to fifty and restarted the service.",
        "Everything should be back to normal now.",
    ]),
    ("single_word", [
        "Done.",
    ]),
    ("two_words", [
        "Got it.",
    ]),
    ("natural_conversation", [
        "Oh, interesting!",
        "I hadn't thought about it that way before.",
        "Let me think about it and get back to you.",
    ]),
]

checkpoints = [
    ("v6-4bit", "checkpoints/katie-v6-4bit", ["katie"]),
    ("v7-bf16", "checkpoints/katie-joe-v7", ["katie", "joe"]),
]

output_base = os.path.expanduser("~/Downloads/holler-mlx-benchmark")
os.makedirs(output_base, exist_ok=True)

from mlx_audio.tts import load

for cp_name, cp_path, voices in checkpoints:
    full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cp_path)
    print(f"\n{'='*60}")
    print(f"CHECKPOINT: {cp_name} ({full_path})")
    print(f"{'='*60}")

    t_load = time.perf_counter()
    model = load(full_path)
    load_time = time.perf_counter() - t_load

    mem_after_load = mx.metal.get_active_memory() / 1024 / 1024
    print(f"  Load time: {load_time:.3f}s")
    print(f"  Metal RAM after load: {mem_after_load:.0f}MB")

    for voice in voices:
        print(f"\n  === {voice} ({cp_name}) ===")
        voice_dir = os.path.join(output_base, f"{cp_name}-{voice}")
        os.makedirs(voice_dir, exist_ok=True)

        # === TEST 1: Individual sentence timing (non-streaming) ===
        print(f"\n  [Individual sentence timing — non-streaming]")
        for resp_name, sentences in responses:
            for i, sentence in enumerate(sentences):
                t0 = time.perf_counter()
                audio_parts = []
                for result in model.generate(
                    text=sentence,
                    voice=voice,
                    language="english",
                    temperature=0.6,
                    stream=False,
                ):
                    audio_parts.append(np.array(result.audio).flatten())

                elapsed = time.perf_counter() - t0
                if audio_parts:
                    audio = np.concatenate(audio_parts)
                    audio_dur = len(audio) / 24000
                    peak = float(np.max(np.abs(audio)))
                    words = len(sentence.split())
                    print(f"    {elapsed*1000:5.0f}ms → {audio_dur:.2f}s audio | peak={peak:.3f} | {words} words | \"{sentence}\"")

                    wav_path = os.path.join(voice_dir, f"{resp_name}_{i}.wav")
                    sf.write(wav_path, audio, 24000)

        # === TEST 2: Streaming TTFA ===
        print(f"\n  [Streaming TTFA]")
        test_sentences = [s for _, sents in responses[:4] for s in sents]
        for sentence in test_sentences[:8]:
            t0 = time.perf_counter()
            first_chunk_time = None
            chunks = []
            for result in model.generate(
                text=sentence,
                voice=voice,
                language="english",
                temperature=0.6,
                stream=True,
                streaming_interval=0.08,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - t0
                chunks.append(np.array(result.audio).flatten())

            total = time.perf_counter() - t0
            audio = np.concatenate(chunks) if chunks else np.array([])
            audio_dur = len(audio) / 24000
            print(f"    TTFA={first_chunk_time*1000:.0f}ms | total={total*1000:.0f}ms | {len(chunks)} chunks | {audio_dur:.2f}s audio | \"{sentence[:50]}\"")

        # === TEST 3: Sentence queue simulation ===
        print(f"\n  [Sentence queue — simulating ivi pattern]")
        for resp_name, sentences in responses:
            queue_start = time.perf_counter()
            first_audio_ready = None
            all_audio = []

            for i, sentence in enumerate(sentences):
                sent_start = time.perf_counter()
                audio_parts2 = []
                for result in model.generate(
                    text=sentence,
                    voice=voice,
                    language="english",
                    temperature=0.6,
                    stream=False,
                ):
                    audio_parts2.append(np.array(result.audio).flatten())

                audio = np.concatenate(audio_parts2)
                sent_elapsed = time.perf_counter() - sent_start
                all_audio.append(audio)

                if first_audio_ready is None:
                    first_audio_ready = time.perf_counter() - queue_start

                audio_dur = len(audio) / 24000
                cum_audio = sum(len(a) / 24000 for a in all_audio)
                wall_time = time.perf_counter() - queue_start
                ahead = cum_audio - wall_time
                status = f"ahead by {ahead:.2f}s" if ahead > 0 else f"BEHIND by {-ahead:.2f}s"

                if i == 0:
                    print(f"    \"{resp_name}\" ({len(sentences)} sentences):")
                print(f"      [{i}] {sent_elapsed*1000:4.0f}ms gen → {audio_dur:.2f}s audio | {status}")

            total_wall = time.perf_counter() - queue_start
            total_audio = sum(len(a) / 24000 for a in all_audio)
            print(f"      TTFA: {first_audio_ready*1000:.0f}ms | total: {total_wall:.1f}s wall → {total_audio:.1f}s audio")

            concat = np.concatenate(all_audio)
            wav_path = os.path.join(voice_dir, f"queue_{resp_name}.wav")
            sf.write(wav_path, concat, 24000)

    peak_mem = mx.metal.get_peak_memory() / 1024 / 1024
    print(f"\n  Metal RAM peak: {peak_mem:.0f}MB")

print(f"\n\nAll audio saved to: {output_base}")
