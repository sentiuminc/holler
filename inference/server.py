"""Holler TTS Server — fast inference server for Qwen3-TTS on Apple Silicon.

Architecture:
- HTTP server on port 8100
- POST /tts with JSON body: {"text": "...", "voice": "katie", "temperature": 0.6}
- Returns streaming audio as float32 PCM chunks (24kHz mono)
- Generates tokens, decodes in chunks, streams as soon as each chunk is ready
- Also supports GET /tts?text=... for simple testing

Performance: RTF ~0.35-0.40 end-to-end on M1 Pro (gen=0.29 + chunked decode)
"""
import argparse
import json
import time
import os
import io
import struct
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import mlx.core as mx
from mlx_audio.tts import load
from mlx_lm.sample_utils import categorical_sampling

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "checkpoints", "quant-experiment", "affine-6bit-g64")
DEFAULT_PORT = 8100
CHECKPOINT = DEFAULT_CHECKPOINT
PORT = DEFAULT_PORT
SAMPLE_RATE = 24000
DEFAULT_VOICE = "katie"
DEFAULT_TEMP = 0.6
DEFAULT_TOP_K = 50
MAX_TOKENS = 500
STREAM_CHUNK_TOKENS = 40  # Decode every N tokens — larger chunks = less overhead
DEFAULT_CODEBOOKS = 12  # 12 of 16 codebooks: 10% faster, no quality loss audible. Use 16 for max quality.

model = None


suppress_indices_cache = None
zero_token_cache = None

def load_model():
    global model, suppress_indices_cache, zero_token_cache
    print(f"[holler] Loading model from {CHECKPOINT}...")
    model = load(CHECKPOINT)
    mx.set_cache_limit(2 * 1024 * 1024 * 1024)

    # Pre-compute suppress indices (reused every token)
    config = model.config.talker_config
    eos = config.codec_eos_token_id
    suppress_indices_cache = mx.array(
        [i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos],
        dtype=mx.int32
    )
    # Pre-compute zero token for codebook padding
    zero_token_cache = mx.zeros((1, 1), dtype=mx.int32)

    # Warmup
    for _ in generate_audio(model, "Hello.", voice=DEFAULT_VOICE):
        pass
    for _ in generate_audio(model, "Testing warmup.", voice=DEFAULT_VOICE):
        pass
    print(f"[holler] Model loaded and warmed up. Default codebooks: {DEFAULT_CODEBOOKS}")


def generate_audio(mdl, text, voice="katie", language="english", temperature=0.6,
                   top_k=50, max_tokens=500, stream_chunk_tokens=STREAM_CHUNK_TOKENS,
                   first_chunk_tokens=3, n_codebooks=DEFAULT_CODEBOOKS):
    """Generate speech, yielding (audio_np, is_final) chunks as they're decoded.

    Uses streaming_step() for incremental decode — no redundant computation.
    First chunk uses fewer tokens for faster TTFA, subsequent chunks are larger.
    n_codebooks: use fewer than 16 sub-codebooks for faster generation (12 recommended).
    """
    config = mdl.config.talker_config
    eos_token_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    actual_extra = min(n_codebooks - 1, num_code_groups - 1)

    # EOS safety: cap max_tokens based on text length (~15 tokens per word typical)
    word_count = len(text.split())
    safe_max = min(max_tokens, max(50, word_count * 20))

    input_embeds, trailing_text_hidden, tts_pad_embed = mdl._prepare_generation_inputs(
        text, language=language, speaker=voice
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed)

    suppress_indices = suppress_indices_cache

    cache = mdl.talker.make_cache()
    code_cache = mdl.talker.code_predictor.make_cache()

    get_input_emb = mdl.talker.get_input_embeddings()
    code_pred = mdl.talker.code_predictor
    code_embeddings = code_pred.codec_embedding

    trailing_idx = 0
    trailing_len = trailing_text_hidden.shape[1]

    generated_codes = []
    decoded_up_to = 0

    # Reset streaming decoder state
    mdl.speech_tokenizer.decoder.reset_streaming_state()

    for step in range(safe_max):
        # Talker forward
        logits, hidden = mdl.talker(input_embeds, cache=cache)

        # Sample first codebook
        first_logits = logits[:, -1, :]
        first_logits = mx.put_along_axis(
            first_logits, suppress_indices[None, :],
            mx.array(float("-inf"), first_logits.dtype), axis=-1
        )
        if top_k > 0:
            top_k_vals = mx.sort(first_logits, axis=-1)[:, -top_k:]
            threshold = top_k_vals[:, 0:1]
            first_logits = mx.where(first_logits < threshold, float("-inf"), first_logits)
        next_token = categorical_sampling(first_logits, temperature)[:, None]

        is_eos = next_token[0, 0] == eos_token_id

        # Code predictor (generate only actual_extra sub-codebooks)
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

        # Pad remaining codebooks with zeros for decoder
        while len(code_tokens) < num_code_groups:
            code_tokens.append(zero_token_cache)

        all_codes = mx.concatenate(code_tokens, axis=1)

        # Prepare next input
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

        # Decode chunk when enough tokens accumulated
        # First chunk is smaller for faster TTFA
        n_new = len(generated_codes) - decoded_up_to
        threshold = first_chunk_tokens if decoded_up_to == 0 else stream_chunk_tokens
        if n_new >= threshold:
            chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
            codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
            mx.eval(codes_for_decoder)

            wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
            audio_chunk = wav.squeeze(1)[0]
            mx.eval(audio_chunk)

            decoded_up_to = len(generated_codes)
            audio_np = np.array(audio_chunk).flatten().astype(np.float32)
            yield audio_np, False

    # Decode remaining tokens
    if len(generated_codes) > decoded_up_to:
        chunk_codes = mx.stack(generated_codes[decoded_up_to:], axis=1)
        codes_for_decoder = mx.transpose(chunk_codes, (0, 2, 1))
        mx.eval(codes_for_decoder)

        wav = mdl.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
        audio_chunk = wav.squeeze(1)[0]
        mx.eval(audio_chunk)

        audio_np = np.array(audio_chunk).flatten().astype(np.float32)
        mx.clear_cache()
        yield audio_np, True
    else:
        mx.clear_cache()
        yield np.array([], dtype=np.float32), True


class TTSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if mark_request:
            mark_request()
        if self.path != "/tts":
            self.send_error(404)
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

        text = body.get("text", "")
        if not text:
            self.send_error(400, "Missing 'text' field")
            return

        voice = body.get("voice", DEFAULT_VOICE)
        temperature = float(body.get("temperature", DEFAULT_TEMP))
        top_k = int(body.get("top_k", DEFAULT_TOP_K))
        chunk_tokens = int(body.get("chunk_tokens", STREAM_CHUNK_TOKENS))
        first_chunk = int(body.get("first_chunk_tokens", 3))
        n_codebooks = int(body.get("n_codebooks", DEFAULT_CODEBOOKS))

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
        self.send_header("X-Format", "float32-pcm")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        t0 = time.time()
        total_samples = 0
        n_chunks = 0
        ttfa = None

        for audio_np, is_final in generate_audio(
            model, text, voice=voice, temperature=temperature,
            top_k=top_k, stream_chunk_tokens=chunk_tokens,
            first_chunk_tokens=first_chunk, n_codebooks=n_codebooks,
        ):
            if len(audio_np) > 0:
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                total_samples += len(audio_np)
                n_chunks += 1
                # Write chunked transfer encoding
                data = audio_np.tobytes()
                self.wfile.write(f"{len(data):x}\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        # Final chunk
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

        total_ms = (time.time() - t0) * 1000
        audio_s = total_samples / SAMPLE_RATE
        rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999

        print(f"[holler] \"{text[:50]}\" → {audio_s:.1f}s audio, "
              f"TTFA={ttfa:.0f}ms, total={total_ms:.0f}ms, RTF={rtf:.3f}, "
              f"{n_chunks} chunks")

    def do_GET(self):
        if mark_request:
            mark_request()
        if self.path.startswith("/tts"):
            from urllib.parse import urlparse, parse_qs
            params = parse_qs(urlparse(self.path).query)
            text = params.get("text", [""])[0]
            if not text:
                self.send_error(400, "Missing 'text' parameter")
                return

            voice = params.get("voice", [DEFAULT_VOICE])[0]
            temperature = float(params.get("temperature", [DEFAULT_TEMP])[0])

            t0 = time.time()
            all_audio = []
            for audio_np, is_final in generate_audio(model, text, voice=voice, temperature=temperature):
                if len(audio_np) > 0:
                    all_audio.append(audio_np)

            if all_audio:
                audio = np.concatenate(all_audio)
            else:
                audio = np.array([], dtype=np.float32)

            total_ms = (time.time() - t0) * 1000
            audio_s = len(audio) / SAMPLE_RATE
            rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999

            # Return as WAV
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
                  f"{total_ms:.0f}ms, RTF={rtf:.3f}")

        elif self.path == "/benchmark":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()

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
                for audio_np, is_final in generate_audio(model, text):
                    if len(audio_np) > 0:
                        if ttfa is None:
                            ttfa = (time.time() - t0) * 1000
                        total_samples += len(audio_np)
                total_ms = (time.time() - t0) * 1000
                audio_s = total_samples / SAMPLE_RATE
                rtf = (total_ms / 1000) / audio_s if audio_s > 0 else 999
                results.append((text, ttfa or 0, total_ms, audio_s, rtf))

            out = "Holler Benchmark\n" + "=" * 70 + "\n"
            out += f"{'Text':<45} {'TTFA':>6} {'Total':>7} {'Audio':>6} {'RTF':>6}\n"
            out += "-" * 70 + "\n"
            for text, ttfa, total_ms, audio_s, rtf in results:
                out += f"{text[:44]:<45} {ttfa:>5.0f}ms {total_ms:>6.0f}ms {audio_s:>5.1f}s {rtf:>5.3f}\n"

            avg_rtf = sum(r[4] for r in results) / len(results)
            avg_ttfa = sum(r[1] for r in results) / len(results)
            out += f"\nAvg RTF: {avg_rtf:.3f}, Avg TTFA: {avg_ttfa:.0f}ms\n"
            out += f"Target RTF ≤ 0.50: {'✅ PASS' if avg_rtf <= 0.50 else '❌ FAIL'}\n"

            self.wfile.write(out.encode())
            print(f"[holler] Benchmark: avg RTF={avg_rtf:.3f}, avg TTFA={avg_ttfa:.0f}ms")

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model": os.path.basename(CHECKPOINT)}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress default access log


def keepalive_loop():
    """Periodically run a tiny generation to keep Metal caches warm.
    Only runs if no real request has been served recently."""
    import threading
    last_request_time = [time.time()]

    def mark_request():
        last_request_time[0] = time.time()

    def _keepalive():
        while True:
            time.sleep(45)
            if time.time() - last_request_time[0] > 30:
                try:
                    for _ in generate_audio(model, ".", max_tokens=5):
                        pass
                except Exception:
                    pass

    t = threading.Thread(target=_keepalive, daemon=True)
    t.start()
    return mark_request


mark_request = None

def main():
    global mark_request, CHECKPOINT, PORT

    parser = argparse.ArgumentParser(description="Holler TTS Server")
    parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint directory")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                        help="Server port (default: 8100)")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    PORT = args.port

    load_model()
    mark_request = keepalive_loop()
    server = HTTPServer(("0.0.0.0", PORT), TTSHandler)
    print(f"[holler] Server running on http://localhost:{PORT}")
    print(f"[holler] POST /tts with JSON body: {{\"text\": \"...\"}}")
    print(f"[holler] GET /tts?text=hello for WAV download")
    print(f"[holler] GET /benchmark for RTF benchmark")
    print(f"[holler] GET /health for health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[holler] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
