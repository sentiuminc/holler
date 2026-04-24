"""Investigate EOS reliability with fewer codebooks.
Run each sentence 5 times with 12cb and 16cb, count how often it hits max_tokens."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

from experiment_combined import fast_generate_ncb

# Warmup
fast_generate_ncb(model, "Hello.", n_codebooks=12)
print("Warmup done.\n")

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

N_TRIALS = 5
MAX_TOKENS = 200  # If it goes beyond 200 tokens, it's stuck

for n_cb in [16, 12]:
    print(f"=== {n_cb} codebooks, {N_TRIALS} trials per sentence ===")
    total_ok = 0
    total_stuck = 0

    for text in sentences:
        stuck = 0
        tok_counts = []
        for trial in range(N_TRIALS):
            audio, n_tok, gen_ms, dec_ms = fast_generate_ncb(
                model, text, n_codebooks=n_cb, max_tokens=MAX_TOKENS
            )
            tok_counts.append(n_tok)
            if n_tok >= MAX_TOKENS:
                stuck += 1

        ok = N_TRIALS - stuck
        total_ok += ok
        total_stuck += stuck
        status = "✅" if stuck == 0 else f"⚠️ {stuck}/{N_TRIALS} stuck"
        avg_tok = sum(tok_counts) / len(tok_counts)
        print(f"  {status} avg={avg_tok:.0f}tok [{','.join(str(t) for t in tok_counts)}] \"{text[:50]}\"")

    pct = total_ok / (total_ok + total_stuck) * 100
    print(f"  Total: {total_ok}/{total_ok+total_stuck} OK ({pct:.0f}%)\n")
