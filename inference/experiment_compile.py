"""Experiment: Can mx.compile() on the talker/code_predictor forward pass help?
Also: does temperature=0 (greedy) for main token affect speed?
Also: does reducing top_k help?"""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")
SAMPLE_RATE = 24000

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

# Import fast_generate
from fast_generate import fast_generate

# Warmup
fast_generate(model, "Hello.", voice="katie")
fast_generate(model, "Testing warmup.", voice="katie")
print("Warmup done.\n")

test_texts = [
    "Something about the umami thing appeals to me.",
    "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone.",
]

# Experiment 1: Different top_k values (top_k uses sort which may be expensive)
print("=== Experiment: top_k values ===")
for top_k in [0, 10, 25, 50, 100]:
    times = []
    for text in test_texts:
        audio, n, gen_ms, dec_ms = fast_generate(model, text, voice="katie", top_k=top_k)
        audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
        times.append(gen_rtf)
    avg = sum(times) / len(times)
    print(f"  top_k={top_k:>3}: avg gen RTF={avg:.3f}")

# Experiment 2: Temperature values
print("\n=== Experiment: temperature ===")
for temp in [0.1, 0.3, 0.6, 0.9]:
    times = []
    for text in test_texts:
        audio, n, gen_ms, dec_ms = fast_generate(model, text, voice="katie", temperature=temp)
        audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
        times.append(gen_rtf)
    avg = sum(times) / len(times)
    print(f"  temp={temp:.1f}: avg gen RTF={avg:.3f}")

# Experiment 3: try to mx.compile the talker forward pass
print("\n=== Experiment: mx.compile on talker ===")
# The talker.model.__call__ is the transformer stack
original_model_call = model.talker.model.__call__

@mx.compile
def compiled_model_call(inputs_embeds, position_ids=None, mask=None, cache=None, attention_mask=None):
    return original_model_call(inputs_embeds, position_ids, mask, cache, attention_mask)

# Test if compile helps
# Note: compile with KV cache is tricky because cache shapes change each step
# Let's just measure vanilla vs attempting compile
for label, use_compile in [("vanilla", False)]:
    times = []
    for text in test_texts:
        audio, n, gen_ms, dec_ms = fast_generate(model, text, voice="katie")
        audio_s = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        gen_rtf = (gen_ms / 1000) / audio_s if audio_s > 0 else 999
        times.append(gen_rtf)
    avg = sum(times) / len(times)
    print(f"  {label}: avg gen RTF={avg:.3f}")

# Experiment 4: disable repetition penalty (it requires per-token processing)
print("\n=== Experiment: repetition_penalty impact ===")
# Our fast_generate already doesn't use rep penalty (unlike mlx-audio's generate)
# Let's verify there's no difference
print("  (fast_generate already skips rep penalty — this is reflected in baseline)")

# Experiment 5: measure Metal memory during generation
print("\n=== Memory usage ===")
mx.metal.reset_peak_memory()
audio, n, gen_ms, dec_ms = fast_generate(model, test_texts[1], voice="katie")
peak = mx.metal.get_peak_memory() / 1024 / 1024
active = mx.metal.get_active_memory() / 1024 / 1024
print(f"  Peak Metal: {peak:.0f}MB")
print(f"  Active Metal: {active:.0f}MB")

# Final summary
print("\n=== Summary ===")
audio, n, gen_ms, dec_ms = fast_generate(model, test_texts[1], voice="katie")
audio_s = len(audio) / SAMPLE_RATE
gen_rtf = (gen_ms / 1000) / audio_s
total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s
per_token_ms = gen_ms / n if n > 0 else 0
print(f"  Tokens: {n}, Per-token: {per_token_ms:.1f}ms")
print(f"  Gen RTF: {gen_rtf:.3f}, Total RTF: {total_rtf:.3f}")
print(f"  Token rate: {n / (gen_ms/1000):.1f} tok/s (need 24 for RTF 0.5)")
