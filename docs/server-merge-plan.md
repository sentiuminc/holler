# Server Merge Plan: holler/inference/server.py ← ivi/sidecar/tts-sidecar-fast.py

**Status: DONE.** Merged and verified 2026-04-25.
**Created:** 2026-04-25
**Executed:** 2026-04-25

Kept for reference — documents what was planned, what extra issues surfaced during implementation, and the MLX 0.31.2 regression discovered along the way.

---

## Design Principles

1. **`holler/inference/server.py` is the source of truth.** Everything fits around it.
2. **Continuous streaming is what matters.** STREAM_CHUNK_TOKENS = 3. Not 40. 40-token chunks cause 1.15s gaps between audio bursts — that's not streaming.
3. **The server should be proper and shippable.** Good API design, argparse, proper HTTP, because this is what people get when they `git clone` holler.
4. **ivi imports from holler, not the other way around.** The ivi sidecar is a thin wrapper that imports `generate_audio` from holler.
5. **Include a test website.** Clone holler, start the server, open a browser, test immediately.

---

## What was ported from ivi sidecar

**Generation improvements:**
- Split `generate_audio()` into `_run_generation()` (inner loop) + `generate_audio()` (retry wrapper)
- 4-piece silence handling: skip silent warmup chunks, sample-level onset trim (150ms pre-roll), 20ms fade-out, post-speech silence buffer + abort after 16 silent tokens
- Retry logic: if first attempt aborts (no speech), retry once with fresh decoder state
- Three helper functions: `_has_speech()`, `_find_speech_onset()`, `_apply_trailing_fadeout()`

**Decoder state carry-over:**
- `reset_decoder` parameter on `generate_audio()` (default True)
- `continue` field in POST body maps to `reset_decoder=not continue`

**True streaming:**
- Uniform 3-token chunks. Threshold logic: `STREAM_CHUNK_TOKENS if speech_started else FIRST_CHUNK_TOKENS`

**Server infrastructure:**
- `threading.Lock()` around generation (MLX model is NOT thread-safe)
- `ThreadingServer(socketserver.ThreadingMixIn, HTTPServer)` for HTTP concurrency
- `protocol_version = "HTTP/1.1"`
- `BrokenPipeError` handling
- `flush=True` on all print statements
- Load timing in `load_model()`
- CORS headers + OPTIONS handler
- Test HTML at `GET /` and `GET /test`

**What holler already had (kept):**
- `argparse` with `--checkpoint` and `--port`
- `GET /tts?text=...` → WAV download
- `GET /benchmark` → 6-sentence benchmark
- `GET /health` → JSON health check
- Per-request config: `voice`, `temperature`, `top_k`, `n_codebooks`
- `0.0.0.0` binding, KeyboardInterrupt handling

---

## 10 issues identified in original plan — all resolved

1. **Variable name mismatch** — kept holler's `suppress_indices_cache`/`zero_token_cache` convention.
2. **Yield signature change** — `(audio_np, is_final)` → bare `audio_np`. All callers updated.
3. **generate_lock** — added at module level, acquired in POST handler, GET handler, benchmark, and keepalive. Lock is in callers, not inside `generate_audio()` (generators can't be wrapped with context manager locks).
4. **CORS on errors** — all `send_error()` replaced with manual responses including `_cors()` headers.
5. **Endpoint mismatch** — test HTML updated to POST to `/tts`.
6. **Trailing fadeout gap** — accepted as minor edge case for now.
7. **Double-abort** — `generate_audio` yields nothing, client gets empty response. Functional.
8. **HTTP version** — `protocol_version = "HTTP/1.1"` added.
9. **Benchmark with small chunks** — benchmark header now shows `chunk_tokens=N`.
10. **N_CODEBOOKS** — accepted as parameter (holler's pattern), not global.

## 5 extra issues found during implementation

These were NOT in the original plan:

11. **`flush=True` missing on all holler prints** — critical when running as subprocess (stdout is block-buffered). Logs arrive delayed or lost on crash without flush.
12. **`ThreadingServer` missing** — plain `HTTPServer` is single-threaded. Keepalive blocks incoming requests. Added `ThreadingMixIn`.
13. **`BrokenPipeError` not caught** — client disconnect mid-stream crashed the server thread. Wrapped streaming loop + chunked terminator in try/except.
14. **No load timing** — added `time.time()` around model load + warmup, printed in "Ready in Xs" message.
15. **Keepalive missing `generate_lock`** — latent race condition. Keepalive thread and HTTP handler both mutating MLX model state. Fixed: keepalive acquires lock.

## Keepalive bug found and fixed

Keepalive was sending `"."` with `max_tokens=5`. With silence handling, a period never produces speech → abort + retry every 45 seconds, filling logs with spam. Fixed: changed to `"Stay gold."` with `max_tokens=20`.

## MLX 0.31.2 regression discovered

Carry-over (`reset_decoder=False`) causes SIGSEGV on MLX 0.31.2 but works on 0.31.1. Root cause: PR #3282 removed `metal::new_scoped_memory_pool()` from `MetalAllocator::clear_cache()`, changing Metal buffer deallocation timing → use-after-free when streaming decoder holds state.

Filed: https://github.com/ml-explore/mlx/issues/3450

**Workaround:** Pin MLX to 0.31.1.

---

## Final state

### `holler/inference/server.py` — 613 lines

Single source of truth. All generation logic, silence handling, carry-over, retry, threading, CORS.

**API:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tts` | Streaming float32 PCM. Body: `text`, `voice`, `temperature`, `top_k`, `n_codebooks`, `continue` |
| GET | `/tts?text=...` | WAV file download |
| GET | `/benchmark` | 6-sentence RTF/TTFA benchmark |
| GET | `/health` | JSON health check |
| GET | `/` or `/test` | Browser test UI |
| OPTIONS | any | CORS preflight |

### `holler/inference/tts-test.html` — 491 lines

Browser test UI: text input, voice selector, carry-over toggle, temperature, streaming playback, TTFA/RTF metrics, replay, WAV download.

### `ivi/sidecar/tts-sidecar-holler.py` — 143 lines

Thin ivi wrapper. Imports `generate_audio` from holler via `importlib`, exposes `/speak` on `127.0.0.1:52946`. Uses holler's `generate_lock` and `keepalive`.

### Verification results (2026-04-25, M1 Pro, MLX 0.31.1, 6-bit affine)

| Test | Result |
|------|--------|
| Health | `{"status": "ok", "model": "affine-6bit-g64"}` |
| WAV download | 1.52s audio, 24kHz mono |
| Streaming POST | TTFA=133ms, RTF=0.493, 9 chunks |
| Carry-over | Both sentences 200, `[cont]` logged |
| Short utterance "Okay." | 0.5s audio, no hang |
| CORS preflight | HTTP 204 |
| Error (empty text) | HTTP 400, `{"error":"empty text"}` |
| Benchmark | Avg RTF 0.492, Avg TTFA 149ms, PASS |

## Future work

- **Talker KV carry-over** — decoder carry-over only prevents sample-level glitches. True prosodic continuity needs the LLM to remember previous sentences. The talker's KV cache is currently always fresh.
- **Bounded decoder KV** — reset transformer KV cache after N sentences to prevent drift/memory bloat, keep conv buffers.
- **MLX 0.31.2 fix** — monitor https://github.com/ml-explore/mlx/issues/3450, upgrade when fixed.
