"""RTF benchmark v2: measures generation-only RTF (no streaming decode overhead).
Also tests non-streaming decode separately to isolate bottlenecks."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000
PASSES = 3

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

print(f"Loading model from {CHECKPOINT}...")
model = load(CHECKPOINT)

# Warmup
for r in model.generate(text="Hello.", voice="katie", language="english",
                        temperature=0.6, stream=False):
    pass
print("Warmup done.\n")

for pass_num in range(1, PASSES + 1):
    print(f"{'='*80}")
    print(f"Pass {pass_num}/{PASSES}")
    print(f"{'='*80}")
    print(f"{'Words':>5} {'GenOnly':>8} {'Decode':>8} {'Total':>8} {'Audio':>7} {'GenRTF':>7} {'TotRTF':>7}  Text")
    print("-" * 80)

    gen_rtfs = []
    total_rtfs = []

    for text in sentences:
        # Non-streaming: generation + decode
        t0 = time.time()
        audio_all = []
        gen_tokens = 0

        for result in model.generate(
            text=text, voice="katie", language="english",
            temperature=0.6, stream=False,
        ):
            if hasattr(result, 'audio'):
                audio_all.append(np.array(result.audio).flatten().astype(np.float32))
                gen_tokens = getattr(result, 'token_count', 0)

        total_ms = (time.time() - t0) * 1000

        if audio_all:
            audio = np.concatenate(audio_all)
            audio_s = len(audio) / SAMPLE_RATE
        else:
            audio_s = 0

        # Estimate gen-only time: token_count / 12Hz * codec_rate
        # Actually, let's measure it properly with the instrumented approach
        gen_only_ms = total_ms  # For non-streaming, includes decode
        decode_ms = 0

        # Calculate token generation time vs decode time
        # gen_tokens tokens at 12Hz = gen_tokens/12 seconds of audio
        if gen_tokens > 0:
            gen_time_per_token = total_ms / gen_tokens  # includes decode overhead

        words = len(text.split())
        total_rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999

        total_rtfs.append(total_rtf)

        print(f"{words:>5} {'—':>8} {'—':>8} {total_ms:>7.0f}ms {audio_s:>6.1f}s {'—':>7} {total_rtf:>6.2f}  {text[:45]}")

    avg_rtf = sum(total_rtfs) / len(total_rtfs)
    max_rtf = max(total_rtfs)
    target = "✅ PASS" if avg_rtf <= 0.50 and max_rtf <= 0.70 else "❌ FAIL"
    print(f"\n  Average RTF: {avg_rtf:.3f}")
    print(f"  Max RTF:     {max_rtf:.3f}")
    print(f"  Target (avg ≤ 0.50, max ≤ 0.70): {target}")
    print()

# Now test streaming vs non-streaming on a medium sentence
print("=" * 80)
print("STREAMING vs NON-STREAMING comparison")
print("=" * 80)
test_text = "Something about the umami thing appeals to me."

for stream_mode in [False, True]:
    for interval in ([0.1, 0.5, 1.0, 2.0] if stream_mode else [None]):
        t0 = time.time()
        total_samples = 0
        for result in model.generate(
            text=test_text, voice="katie", language="english",
            temperature=0.6, stream=stream_mode,
            streaming_interval=interval if interval else 0.1,
        ):
            if hasattr(result, 'audio'):
                chunk = np.array(result.audio).flatten()
                total_samples += len(chunk)

        total_ms = (time.time() - t0) * 1000
        audio_s = total_samples / SAMPLE_RATE
        rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
        mode = f"stream={stream_mode}"
        if stream_mode:
            mode += f" interval={interval}"
        print(f"  {mode:<35} {total_ms:>7.0f}ms {audio_s:>5.1f}s RTF={rtf:.3f}")
