"""Experiment: Pre-allocated KV cache to eliminate reallocation overhead.

MLX KVCache grows by concatenation each step. Pre-allocating a fixed-size buffer
and writing into it with index operations could reduce overhead."""
import time
import os
import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling
from mlx_lm.models.cache import KVCache

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "katie-v6-4bit")

print("Loading model...")
model = load(CHECKPOINT)
mx.set_cache_limit(2 * 1024 * 1024 * 1024)

from fast_generate import fast_generate
fast_generate(model, "Hello.", voice="katie")
print("Warmup done.\n")

# Check how KVCache works in MLX
# KVCache.update_and_fetch concatenates new KV to existing
# Let's see if QuantizedKVCache or a trim step helps
print("=== KVCache internals ===")
cache = KVCache()
dummy_k = mx.zeros((1, 8, 1, 128))  # batch, heads, 1 step, head_dim
dummy_v = mx.zeros((1, 8, 1, 128))

# Simulate 100 steps
t0 = time.time()
for i in range(100):
    k, v = cache.update_and_fetch(dummy_k, dummy_v)
    mx.eval(k, v)
cache_100_ms = (time.time() - t0) * 1000
print(f"  100 steps: {cache_100_ms:.0f}ms")
print(f"  KV shape after 100 steps: keys={cache.keys.shape}, offset={cache.offset}")

# Check if there's a QuantizedKVCache or other alternatives
try:
    from mlx_lm.models.cache import QuantizedKVCache
    print("  QuantizedKVCache available!")
except ImportError:
    print("  QuantizedKVCache not available")

try:
    from mlx_lm.models.cache import RotatingKVCache
    print("  RotatingKVCache available!")
except ImportError:
    print("  RotatingKVCache not available")

# Check what cache options exist
import mlx_lm.models.cache as cache_module
print(f"  Available: {[x for x in dir(cache_module) if 'Cache' in x]}")

# Experiment: Does trimming the cache after prefill help?
# The prefill step creates a large initial KV entry, then each subsequent step adds 1 token
# After prefill, we only need the step-by-step increments

# Let's try: use trim_prompt_cache to reduce prefill cache
print("\n=== KV cache with trim_prompt_cache ===")
text = "Hold on, speak a couple sentences, release, and let me know if the gap between sentences is gone."
config = model.config.talker_config
eos_token_id = config.codec_eos_token_id
num_code_groups = config.num_code_groups
actual_extra = 11

input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
    text, language="english", speaker="katie"
)
mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)
print(f"  Prefill input: {input_embeds.shape}")

suppress_start = config.vocab_size - 1024
suppress_indices = mx.array([i for i in range(suppress_start, config.vocab_size)
                              if i != eos_token_id], dtype=mx.int32)

# Normal run with cache trim
cache = model.talker.make_cache()
code_cache = model.talker.code_predictor.make_cache()
get_input_emb = model.talker.get_input_embeddings()
code_pred = model.talker.code_predictor
code_embeddings = code_pred.codec_embedding

# Prefill
logits, hidden = model.talker(input_embeds, cache=cache)
mx.eval(logits)

# Check cache size after prefill
prefill_keys_shape = cache[0].keys.shape if cache[0].keys is not None else "None"
print(f"  Cache after prefill: keys={prefill_keys_shape}, offset={cache[0].offset}")

# Now trim — this reduces stored KV to 1 token (last position)
# Actually, KVCache doesn't have a trim method. Let's check if we need the full prefill
# cached or just the last position...

# The answer is: we need the FULL prefill cached, because attention looks at all past KV pairs
# So pre-allocation is the only option, not trimming

# Let's try the QuantizedKVCache if available
print("\n=== Quantized KV Cache ===")
try:
    from mlx_lm.models.cache import QuantizedKVCache
    qcache = [QuantizedKVCache(group_size=64, bits=8) for _ in model.talker.model.layers]
    # Test if quantized cache works with the talker
    input_e2, _, _ = model._prepare_generation_inputs("Hello test.", language="english", speaker="katie")
    mx.eval(input_e2)

    t0 = time.time()
    logits2, hidden2 = model.talker(input_e2, cache=qcache)
    mx.eval(logits2)
    print(f"  Prefill with QuantizedKVCache: {(time.time()-t0)*1000:.0f}ms")

    # Now generate a few tokens
    first_logits = logits2[:, -1, :]
    first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
        mx.array(float("-inf"), first_logits.dtype), axis=-1)
    top_k_vals = mx.sort(first_logits, axis=-1)[:, -50:]
    threshold = top_k_vals[:, 0:1]
    first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
    next_token = categorical_sampling(first_logits, 0.6)[:, None]
    mx.eval(next_token)

    # Step
    next_embed = get_input_emb(next_token)
    t0 = time.time()
    for _ in range(10):
        logits2, hidden2 = model.talker(next_embed, cache=qcache)
        mx.eval(logits2)
    step_ms = (time.time() - t0) * 1000
    print(f"  10 steps with QuantizedKVCache: {step_ms:.0f}ms ({step_ms/10:.1f}ms/step)")

    # Compare with normal cache
    cache_normal = model.talker.make_cache()
    input_e3, _, _ = model._prepare_generation_inputs("Hello test.", language="english", speaker="katie")
    mx.eval(input_e3)
    logits3, hidden3 = model.talker(input_e3, cache=cache_normal)
    mx.eval(logits3)
    first_logits = logits3[:, -1, :]
    first_logits = mx.put_along_axis(first_logits, suppress_indices[None, :],
        mx.array(float("-inf"), first_logits.dtype), axis=-1)
    top_k_vals = mx.sort(first_logits, axis=-1)[:, -50:]
    threshold = top_k_vals[:, 0:1]
    first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
    next_token = categorical_sampling(first_logits, 0.6)[:, None]
    mx.eval(next_token)
    next_embed = get_input_emb(next_token)
    t0 = time.time()
    for _ in range(10):
        logits3, hidden3 = model.talker(next_embed, cache=cache_normal)
        mx.eval(logits3)
    step_ms_normal = (time.time() - t0) * 1000
    print(f"  10 steps with normal KVCache: {step_ms_normal:.0f}ms ({step_ms_normal/10:.1f}ms/step)")

except ImportError:
    print("  QuantizedKVCache not available")
except Exception as e:
    print(f"  QuantizedKVCache failed: {e}")

# Final: Just confirm our best gen-only RTF
print("\n=== Best configuration (12cb, cache_limit=2GB) ===")
from experiment_combined import fast_generate_ncb
for _ in range(2):
    audio, n, gen_ms, dec_ms = fast_generate_ncb(model, text, n_codebooks=12)
    audio_s = len(audio) / 24000
    gen_rtf = (gen_ms / 1000) / audio_s
    total_rtf = ((gen_ms + dec_ms) / 1000) / audio_s
    print(f"  {n} tokens, gen={gen_ms:.0f}ms ({gen_rtf:.3f}), total={gen_ms+dec_ms:.0f}ms ({total_rtf:.3f})")
