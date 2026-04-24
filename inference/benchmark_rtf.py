"""RTF benchmark for Holler autoresearch. Measures generation speed vs real-time."""
import time
import os
import numpy as np
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
                        temperature=0.6, stream=True, streaming_interval=0.1):
    pass
print("Warmup done.\n")

for pass_num in range(1, PASSES + 1):
    print(f"{'='*70}")
    print(f"Pass {pass_num}/{PASSES}")
    print(f"{'='*70}")
    print(f"{'Words':>5} {'TTFA':>7} {'Total':>7} {'Audio':>7} {'RTF':>6}  Text")
    print("-" * 70)

    rtfs = []
    ttfas = []

    for text in sentences:
        t0 = time.time()
        ttfa = None
        total_samples = 0
        speech_started = False

        for result in model.generate(
            text=text,
            voice="katie",
            language="english",
            temperature=0.6,
            stream=True,
            streaming_interval=0.1,
        ):
            if hasattr(result, "audio"):
                chunk = np.array(result.audio).flatten().astype(np.float32)
                peak = float(np.abs(chunk).max())
                if not speech_started and peak < 0.02:
                    continue
                speech_started = True
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                total_samples += len(chunk)

        total_ms = (time.time() - t0) * 1000
        audio_s = total_samples / SAMPLE_RATE
        rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
        words = len(text.split())

        rtfs.append(rtf)
        if ttfa:
            ttfas.append(ttfa)

        print(f"{words:>5} {ttfa:>6.0f}ms {total_ms:>6.0f}ms {audio_s:>6.1f}s {rtf:>6.2f}  {text[:55]}")

    avg_rtf = sum(rtfs) / len(rtfs)
    max_rtf = max(rtfs)
    avg_ttfa = sum(ttfas) / len(ttfas) if ttfas else 0

    print(f"\n  Average RTF: {avg_rtf:.3f}")
    print(f"  Max RTF:     {max_rtf:.3f}")
    print(f"  Avg TTFA:    {avg_ttfa:.0f}ms")
    target = "✅ PASS" if avg_rtf <= 0.50 and max_rtf <= 0.70 else "❌ FAIL"
    print(f"  Target (avg ≤ 0.50, max ≤ 0.70): {target}")
    print()
