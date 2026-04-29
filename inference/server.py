"""Holler TTS Server — fast streaming inference for Qwen3-TTS on Apple Silicon.

Single source of truth for Holler inference. Used directly for open-source,
imported by ivi's thin wrapper for product integration.

API:
  POST /tts  — streaming float32 PCM (24kHz mono), chunked transfer encoding
  GET  /tts  — complete WAV file download
  GET  /benchmark — 6-sentence RTF/TTFA benchmark
  GET  /health — JSON health check
  GET  / or /test — browser test UI

Performance: RTF ~0.38, TTFA ~139ms on M1 Pro (6-bit affine, 12 codebooks)
"""
import argparse
import http.server
import io
import json
import os
import random
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
# from scipy.signal import sosfilt, sosfilt_zi  # uncomment with de-ess filter
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "checkpoints", "holler-kit-dakota-6bit")
DEFAULT_PORT = 8100
CHECKPOINT = DEFAULT_CHECKPOINT
PORT = DEFAULT_PORT
SAMPLE_RATE = 24000
DEFAULT_VOICE = "kit"
DEFAULT_TEMP = 0.6
DEFAULT_TOP_K = 50
MAX_TOKENS = 500
FIRST_CHUNK_TOKENS = 3
STREAM_CHUNK_TOKENS = 3
SILENT_ABORT_TOKENS = 16
DEFAULT_CODEBOOKS = 12
CARRYOVER_PAUSE_MIN_MS = 150
CARRYOVER_PAUSE_MAX_MS = 250

# TODO: Dynamic de-esser for 2-6kHz ear fatigue reduction.
# Static EQ (-5dB notch at 3kHz) was tested and works but is too crude —
# cuts presence even when it's fine, and pops at sentence boundaries due
# to filter state reset. Need a proper dynamic de-esser that only attenuates
# when 2-6kHz energy exceeds a threshold (like a sidechain compressor on
# the harshness band). Must work per-chunk in streaming with carried state.

model = None
suppress_indices_cache = None
zero_token_cache = None
available_voices = [DEFAULT_VOICE]

# ---------------------------------------------------------------------------
# MLX Worker — all MLX calls run on this single long-lived thread.
# HTTP request threads post work via inference_queue, read results from
# per-request response queues.  This avoids the fatal crash caused by MLX
# thread-local CompilerCache destructors firing during pthread_exit on
# short-lived ThreadingServer request threads (mlx#2086, mlx#2133).
# ---------------------------------------------------------------------------
import queue as _queue

inference_queue = _queue.Queue()


def _mlx_worker():
    """Permanent MLX owner thread. Processes requests forever."""
    global model, suppress_indices_cache, zero_token_cache

    print(f"[holler] Loading model from {CHECKPOINT}...", flush=True)
    t0 = time.time()
    model = load(CHECKPOINT)
    mx.set_cache_limit(2 * 1024 * 1024 * 1024)

    global available_voices, DEFAULT_VOICE
    config = model.config.talker_config
    spk_id = getattr(config, 'spk_id', None)
    if spk_id and isinstance(spk_id, dict):
        available_voices = sorted(spk_id.keys())
        DEFAULT_VOICE = available_voices[0]
    print(f"[holler] Voices: {available_voices} (default: {DEFAULT_VOICE})", flush=True)
    eos = config.codec_eos_token_id
    suppress_indices_cache = mx.array(
        [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos],
        dtype=mx.int32
    )
    zero_token_cache = mx.zeros((1, 1), dtype=mx.int32)

    for _ in generate_audio(model, "Hello.", voice=DEFAULT_VOICE):
        pass
    for _ in generate_audio(model, "Testing warmup.", voice=DEFAULT_VOICE):
        pass

    print(f"[holler] Ready in {time.time()-t0:.1f}s — http://localhost:{PORT}", flush=True)

    while True:
        try:
            item = inference_queue.get(timeout=45)
        except _queue.Empty:
            if time.time() - last_request_time > 30:
                try:
                    for _ in generate_audio(model, "Stay gold.", max_tokens=20):
                        pass
                except Exception:
                    pass
            continue

        response_q, cancel_event, kwargs = item
        try:
            for audio_chunk in generate_audio(model, **kwargs):
                if cancel_event.is_set():
                    break
                if len(audio_chunk) > 0:
                    response_q.put(audio_chunk)
        except Exception as e:
            response_q.put(e)
        finally:
            response_q.put(None)


_worker_ready = threading.Event()


def load_model():
    """Start the MLX worker thread and wait for it to be ready."""
    def _run():
        _mlx_worker()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for the worker to finish loading by sending a probe request
    probe_q = _queue.Queue()
    probe_cancel = threading.Event()
    inference_queue.put((probe_q, probe_cancel, {
        "text": "Ready.", "voice": DEFAULT_VOICE, "max_tokens": 20
    }))
    while True:
        chunk = probe_q.get()
        if chunk is None:
            break
    _worker_ready.set()


def _has_speech(chunk):
    """Check if a chunk contains speech using 2-of-3 temporal RMS confirmation."""
    window = int(SAMPLE_RATE * 0.01)
    threshold = 0.007
    recent = [False, False, False]
    for i, j in enumerate(range(0, len(chunk) - window, window)):
        rms = float(np.sqrt(np.mean(chunk[j:j + window] ** 2)))
        recent[i % 3] = rms >= threshold
        if sum(recent) >= 2:
            return True
    return False


def _find_speech_onset(chunk):
    """Find speech onset sample index with 150ms pre-roll.

    Scans 10ms windows. Speech confirmed when 2 of 3 consecutive windows
    exceed RMS 0.007 — rejects ghost spikes, catches real speech.
    Returns sample index or -1 if no speech found.
    """
    window = int(SAMPLE_RATE * 0.01)
    pre_roll = int(SAMPLE_RATE * 0.15)
    threshold = 0.007
    recent = [False, False, False]

    for i, j in enumerate(range(0, len(chunk) - window, window)):
        rms = float(np.sqrt(np.mean(chunk[j:j + window] ** 2)))
        recent[i % 3] = rms >= threshold
        if sum(recent) >= 2:
            first_window = max(0, j - 2 * window)
            return max(0, first_window - pre_roll)
    return -1


def _apply_trailing_fadeout(chunk, fade_ms=20):
    """Apply a 20ms linear fade-out to the end of a chunk."""
    fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
    if len(chunk) < fade_samples:
        fade_samples = len(chunk)
    chunk = chunk.copy()
    chunk[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return chunk


def _run_generation(mdl, text, voice, language, temperature, top_k, max_tokens,
                    n_codebooks, reset_decoder, cache_out=None):
    """Core generation loop with 4-piece silence handling.

    Piece 1: skip silent chunks before speech (codec warmup removal)
    Piece 2: sample-level onset trim in first speech chunk (150ms pre-roll)
    Piece 3: 20ms fade-out on final chunk
    Piece 4: post-speech silence buffer + abort after SILENT_ABORT_TOKENS

    Yields (chunk, aborted) tuples. aborted=True signals the caller to retry.

    cache_out: if provided, a dict. On exit, cache_out["talker_cache"] is set
    to the talker KV cache for carry-over into the next call.
    """
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

    word_count = len(text.split())
    safe_max = min(max_tokens, max(50, word_count * 20))

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language=language, speaker=voice
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

    if cache_out is not None and "talker_cache" in cache_out and not reset_decoder:
        cache = cache_out["talker_cache"]
        kv_offset = cache[0].offset if cache else 0
        print(f"[holler] Talker KV carry-over: {kv_offset} tokens in cache", flush=True)
    else:
        cache = mdl.talker.make_cache()
    code_cache = mdl.talker.code_predictor.make_cache()
    get_input_emb = mdl.talker.get_input_embeddings()
    code_pred = mdl.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]
    generated_codes = []
    decoded_up_to = 0
    speech_started = not reset_decoder
    silent_token_count = 0
    aborted = False
    pending_silence = []
    needs_pause = not reset_decoder

    if reset_decoder:
        mdl.speech_tokenizer.decoder.reset_streaming_state()

    # Pre-compute mask for first call when carrying over talker KV cache.
    # The auto-generated causal mask only covers (seq_len, seq_len), but with
    # a warm cache we need (seq_len, seq_len + cache_offset).
    first_call_mask = None
    cache_offset = cache[0].offset if cache and cache[0].offset > 0 else 0
    if cache_offset > 0:
        seq_len = input_embeds.shape[1]
        total_len = seq_len + cache_offset
        rows = mx.arange(seq_len)
        cols = mx.arange(total_len)
        mask = (cols[None, :] > (rows[:, None] + cache_offset))
        first_call_mask = mask.astype(input_embeds.dtype) * mx.finfo(input_embeds.dtype).min

    for step in range(safe_max):
        mask_arg = first_call_mask if step == 0 and first_call_mask is not None else None
        logits, hidden = mdl.talker(input_embeds, cache=cache, mask=mask_arg)

        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(
            first_logits, suppress_indices_cache[None, :],
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
            code_tokens.append(zero_token_cache)

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
        thresh = STREAM_CHUNK_TOKENS if speech_started else FIRST_CHUNK_TOKENS
        if n_new >= thresh:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)

            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = wav.squeeze(1)[0]
            mx.eval(audio_chunk)

            decoded_up_to = len(generated_codes)
            chunk_np = np.array(audio_chunk).flatten().astype(np.float32)

            if not speech_started:
                onset = _find_speech_onset(chunk_np)
                if onset == -1:
                    silent_token_count += n_new
                    if silent_token_count >= SILENT_ABORT_TOKENS:
                        aborted = True
                        break
                    continue
                chunk_np = chunk_np[onset:]
                speech_started = True
                silent_token_count = 0

            if speech_started:
                if _has_speech(chunk_np):
                    if needs_pause:
                        natural_ms = sum(len(p) for p in pending_silence) / SAMPLE_RATE * 1000
                        pause_ms = random.randint(CARRYOVER_PAUSE_MIN_MS, CARRYOVER_PAUSE_MAX_MS)
                        pending_silence.clear()
                        pending_silence.append(np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32))
                        print(f"[holler] Pause: {pause_ms}ms (replaced {natural_ms:.0f}ms natural) | {text[:50]}", flush=True)
                        needs_pause = False
                    if pending_silence:
                        for pending in pending_silence:
                            yield pending, False
                        pending_silence.clear()
                    silent_token_count = 0
                    yield chunk_np, False
                else:
                    pending_silence.append(chunk_np)
                    silent_token_count += n_new
                    if silent_token_count >= SILENT_ABORT_TOKENS:
                        pending_silence.clear()
                        aborted = True
                        break

    if not aborted and len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)

        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = wav.squeeze(1)[0]
        mx.eval(audio_chunk)

        chunk_np = np.array(audio_chunk).flatten().astype(np.float32)
        if not speech_started:
            onset = _find_speech_onset(chunk_np)
            if onset == -1:
                aborted = True
            else:
                chunk_np = chunk_np[onset:]
                speech_started = True
        if not aborted:
            if _has_speech(chunk_np):
                chunk_np = _apply_trailing_fadeout(chunk_np)
                yield chunk_np, False

    if cache_out is not None:
        cache_out["talker_cache"] = cache

    if reset_decoder:
        mx.clear_cache()

    if aborted and not speech_started:
        yield np.array([], dtype=np.float32), True


_carry_over_state = {}

def generate_audio(mdl, text, voice=None, language="english", temperature=0.6,
                   top_k=50, max_tokens=500, n_codebooks=DEFAULT_CODEBOOKS,
                   reset_decoder=True):
    """Generate speech, yielding float32 audio chunks.

    Wraps _run_generation with retry logic: if the first attempt aborts
    (no speech after SILENT_ABORT_TOKENS), retries once with fresh decoder state.

    When reset_decoder=False, persists the talker KV cache across calls
    so the LLM sees previous sentences' token history.

    NOT thread-safe — callers must acquire generate_lock before iterating.
    """
    if voice is None:
        voice = DEFAULT_VOICE
    global _carry_over_state
    cache_out = _carry_over_state if not reset_decoder else {}
    if reset_decoder:
        _carry_over_state = {}

    retry_temps = [temperature, min(temperature + 0.1, 1.0), min(temperature + 0.2, 1.0), min(temperature + 0.3, 1.0)]
    attempts = [
        (reset_decoder, cache_out, retry_temps[0]),
        (True, {}, retry_temps[1]),
        (True, {}, retry_temps[2]),
        (True, {}, retry_temps[3]),
    ]

    # TODO(FIXME): short audio retry may not be needed. The model sometimes
    # produces 0.4s of audio for a 10-word sentence — _has_speech passes it
    # (2-of-3 RMS windows > 0.007) but it sounds like silence. Root cause
    # unclear: either the threshold is too low or the model outputs a brief
    # artifact. Capture a short WAV in the wild to diagnose. This retry is a
    # safety net — if the upstream detection is fixed, remove it.
    word_count = len(text.split())
    min_audio_samples = int(word_count * 0.08 * SAMPLE_RATE)

    for attempt, (rst, cout, temp) in enumerate(attempts):
        if attempt > 0:
            reason = "no speech" if total_samples == 0 else f"too short ({total_samples/SAMPLE_RATE:.1f}s for {word_count}w)"
            print(f"[holler] Abort ({reason}), retry {attempt}/3 temp={temp} | {text}", flush=True)
            mdl.speech_tokenizer.decoder.reset_streaming_state()

        got_speech = False
        total_samples = 0
        for chunk, aborted in _run_generation(
            mdl, text, voice, language, temperature=temp, top_k=top_k,
            max_tokens=max_tokens, n_codebooks=n_codebooks,
            reset_decoder=rst, cache_out=cout
        ):
            if aborted:
                break
            if len(chunk) > 0:
                got_speech = True
                total_samples += len(chunk)
                yield chunk

        if got_speech and total_samples >= min_audio_samples:
            return

    print(f"[holler] All retries failed | {text}", flush=True)


last_request_time = time.time()


class TTSHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        global last_request_time
        last_request_time = time.time()

        if self.path in ("/", "/test"):
            html_path = Path(__file__).parent / "tts-test.html"
            if html_path.exists():
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = b"tts-test.html not found"
            self.send_response(404)

            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/tts"):
            from urllib.parse import urlparse, parse_qs
            params = parse_qs(urlparse(self.path).query)
            text = params.get("text", [""])[0]
            if not text:
                err = b'{"error":"missing text parameter"}'
                self.send_response(400)
    
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return

            voice = params.get("voice", [DEFAULT_VOICE])[0]
            temperature = float(params.get("temperature", [DEFAULT_TEMP])[0])

            t0 = time.time()
            all_audio = []
            response_q = _queue.Queue()
            cancel_event = threading.Event()
            inference_queue.put((response_q, cancel_event, {
                "text": text, "voice": voice, "temperature": temperature
            }))
            while True:
                chunk = response_q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    break
                all_audio.append(chunk)

            if all_audio:
                audio = np.concatenate(all_audio)
            else:
                audio = np.array([], dtype=np.float32)

            total_ms = (time.time() - t0) * 1000
            audio_s = len(audio) / SAMPLE_RATE
            rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999

            import wave
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                wf.writeframes(pcm16.tobytes())

            wav_data = buf.getvalue()
            self.send_response(200)

            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_data)))
            self.send_header("X-RTF", f"{rtf:.3f}")
            self.send_header("X-Audio-Duration", f"{audio_s:.2f}")
            self.send_header("X-Generation-Time", f"{total_ms:.0f}")
            self.end_headers()
            self.wfile.write(wav_data)

            print(f"[holler] GET \"{text[:50]}\" → {audio_s:.1f}s audio, "
                  f"{total_ms:.0f}ms, RTF={rtf:.3f}", flush=True)

        elif self.path == "/benchmark":
            self._run_benchmark()

        elif self.path == "/health":
            body = json.dumps({"status": "ok", "model": os.path.basename(CHECKPOINT), "voices": available_voices}).encode()
            self.send_response(200)

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            body = b'{"error":"not found"}'
            self.send_response(404)

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        global last_request_time
        last_request_time = time.time()

        if self.path != "/tts":
            body = b'{"error":"not found"}'
            self.send_response(404)

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

        text = body.get("text", "").strip()
        if not text:
            err = b'{"error":"empty text"}'
            self.send_response(400)

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        voice = body.get("voice", DEFAULT_VOICE)
        temperature = float(body.get("temperature", DEFAULT_TEMP))
        top_k = int(body.get("top_k", DEFAULT_TOP_K))
        n_codebooks = int(body.get("n_codebooks", DEFAULT_CODEBOOKS))
        continue_prosody = body.get("continue", False)

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
        self.send_header("X-Format", "float32-pcm")
        self.end_headers()

        t0 = time.time()
        ttfa = None
        total_samples = 0
        n_chunks = 0

        response_q = _queue.Queue()
        cancel_event = threading.Event()
        inference_queue.put((response_q, cancel_event, {
            "text": text, "voice": voice, "temperature": temperature,
            "top_k": top_k, "n_codebooks": n_codebooks,
            "reset_decoder": not continue_prosody
        }))

        try:
            while True:
                chunk = response_q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    break

                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000

                raw = chunk.tobytes()
                self.wfile.write(f"{len(raw):x}\r\n".encode())
                self.wfile.write(raw)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                total_samples += len(chunk)
                n_chunks += 1
        except BrokenPipeError:
            cancel_event.set()

        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

        total_ms = (time.time() - t0) * 1000
        audio_len = total_samples / SAMPLE_RATE
        rtf = (total_ms / 1000) / audio_len if audio_len > 0 else 999
        ttfa_str = f"{ttfa:.0f}" if ttfa else "—"
        cont_str = " [cont]" if continue_prosody else ""
        print(f"[holler] TTFA={ttfa_str}ms total={total_ms:.0f}ms audio={audio_len:.1f}s "
              f"RTF={rtf:.3f} chunks={n_chunks}{cont_str} | {text}", flush=True)

    def _run_benchmark(self):
        sentences = [
            "Hey!",
            "Got it.",
            "What is on your mind?",
            "Yeah, that is pretty common with voice input.",
            "Something about the umami thing appeals to me.",
            "Hold on, speak a couple sentences, release, and let me know if the gap is gone.",
        ]

        results = []
        for text in sentences:
            t0 = time.time()
            total_samples = 0
            ttfa = None
            response_q = _queue.Queue()
            cancel_event = threading.Event()
            inference_queue.put((response_q, cancel_event, {"text": text}))
            while True:
                chunk = response_q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    break
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                total_samples += len(chunk)
            total_ms = (time.time() - t0) * 1000
            audio_s = total_samples / SAMPLE_RATE
            rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
            results.append((text, ttfa or 0, total_ms, audio_s, rtf))

        out = f"Holler Benchmark (chunk_tokens={STREAM_CHUNK_TOKENS})\n" + "=" * 70 + "\n"
        out += f"{'Text':<45} {'TTFA':>6} {'Total':>7} {'Audio':>6} {'RTF':>6}\n"
        out += "-" * 70 + "\n"
        for text, ttfa, total_ms, audio_s, rtf in results:
            out += f"{text[:44]:<45} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}\n"

        avg_rtf = sum(r[4] for r in results) / len(results)
        avg_ttfa = sum(r[1] for r in results) / len(results)
        out += f"\nAvg RTF: {avg_rtf:.3f}, Avg TTFA: {avg_ttfa:.0f}ms\n"
        out += f"Target RTF ≤ 0.50: {'PASS' if avg_rtf <= 0.50 else 'FAIL'}\n"

        body = out.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"[holler] Benchmark: avg RTF={avg_rtf:.3f}, avg TTFA={avg_ttfa:.0f}ms", flush=True)

    def log_message(self, format, *args):
        pass


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def keepalive():
    pass


def main():
    global CHECKPOINT, PORT

    parser = argparse.ArgumentParser(description="Holler TTS Server")
    parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint directory")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                        help="Server port (default: 8100)")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    PORT = args.port

    load_model()
    server = ThreadingServer(("0.0.0.0", PORT), TTSHandler)
    print(f"[holler] Server running on http://localhost:{PORT}", flush=True)
    print(f"[holler] POST /tts — streaming float32 PCM", flush=True)
    print(f"[holler] GET  /tts?text=hello — WAV download", flush=True)
    print(f"[holler] GET  /benchmark — RTF benchmark", flush=True)
    print(f"[holler] GET  / — browser test UI", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[holler] Shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
