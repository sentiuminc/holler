# GitHub Issue Draft: ml-explore/mlx

**Title:** `mx.clear_cache()` causes SIGSEGV when streaming decoder holds state across calls (0.31.2 regression)

---

## Description

`mx.clear_cache()` between two sequential `mx.eval()` calls causes a SIGSEGV on 0.31.2 when a stateful streaming decoder retains internal Metal buffers across calls. Same code works fine on 0.31.1.

This is a regression introduced by #3282 (smart pointer migration), which removed the `metal::new_scoped_memory_pool()` from `MetalAllocator::clear_cache()`.

## Reproduction

We're running a Qwen3-TTS codec decoder with streaming state (conv buffers + KV cache) that persists across sequential generations for prosody carry-over. Simplified pattern:

```python
import mlx.core as mx
from mlx_audio.tts import load

model = load("checkpoint_path")

# Generation 1 — normal
model.speech_tokenizer.decoder.reset_streaming_state()
# ... generate tokens, decode chunks via streaming_step() ...
mx.clear_cache()  # clean up after generation

# Generation 2 — carry-over (decoder state NOT reset)
# ... generate tokens, decode chunks via streaming_step() ...
# ^^^ SIGSEGV here on 0.31.2, works on 0.31.1
```

The key is that `reset_streaming_state()` is intentionally skipped for generation 2 — the decoder's conv buffers and KV cache from generation 1 are reused for seamless audio transitions. `mx.clear_cache()` between the two generations invalidates Metal buffers that the decoder is still holding.

**Environment:**
- macOS 15.5, M1 Pro
- MLX 0.31.2: SIGSEGV (exit code 139, no Python traceback)
- MLX 0.31.1: works perfectly
- mlx-audio 0.4.x, Qwen3-TTS-0.6B (quantized 6-bit affine)

## Root Cause

PR #3282 changed `MetalAllocator::clear_cache()`:

```cpp
// 0.31.1
void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  auto pool = metal::new_scoped_memory_pool();
  num_resources_ -= buffer_cache_.clear();
}

// 0.31.2
void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}
```

The scoped memory pool was moved into the per-buffer release callback, but the semantics changed: in 0.31.1, all buffer releases happened within one `@autoreleasepool`, ensuring autoreleased ObjC objects were drained atomically. In 0.31.2, each buffer release creates its own pool, changing the timing of Metal object deallocation. Combined with `commandBufferWithUnretainedReferences()`, this can release Metal buffers while they're still referenced by the streaming decoder's internal state.

## Expected Behavior

`mx.clear_cache()` should only release buffers that are genuinely cached (freed by MLX but retained for reuse). Buffers still held by live `mx.array` objects (like a streaming decoder's internal state) should not be affected.

## Workaround

Pin to MLX 0.31.1: `pip install mlx==0.31.1`

Untested but likely fix: add `mx.synchronize()` before `mx.clear_cache()`.

## Semver Note

0.31.1 → 0.31.2 is a patch version bump, but the release includes 81 commits with a 4-PR chain rewriting Metal CommandEncoder lifecycle (#3264, #3316, #3348, #3281), smart pointer migration for all Metal objects (#3282), and thread-local storage changes. This is a significant internal refactor that introduced a breaking regression — not what a patch bump communicates per [semver](https://semver.org/). A minor version bump (0.32.0) would have been more appropriate and would have flagged to users that careful testing is warranted.
