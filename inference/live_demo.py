"""Live TTS demo — type text, hear it spoken fast."""
import http.server
import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from mlx_audio.tts import load
import mlx.core as mx

DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent.parent / "checkpoints" / "nora-joe-v1-6bit")
PORT = 8099
SAMPLE_RATE = 24000

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
_args = _parser.parse_args()
CHECKPOINT = _args.checkpoint

print(f"Loading model from {CHECKPOINT}...", flush=True)
t0 = time.time()
model = load(CHECKPOINT)

# Detect available voices from config
_config_path = Path(CHECKPOINT) / "config.json"
VOICES = ["katie"]
if _config_path.exists():
    _cfg = json.load(open(_config_path))
    _spk = _cfg.get("talker_config", {}).get("spk_id", {})
    if _spk:
        VOICES = sorted(_spk.keys())
print(f"Voices: {VOICES}")

# Warmup with first voice
for r in model.generate(text="Hello.", voice=VOICES[0], language="english",
                        temperature=0.6, stream=True, streaming_interval=0.1):
    pass

# Pre-open audio stream at boot — stays open forever
# 100ms buffer: enough to bridge gaps between model chunks (~83ms apart), low enough to feel instant
audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', latency=0.1)
audio_stream.start()

print(f"Ready in {time.time()-t0:.1f}s — http://localhost:{PORT}", flush=True)

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Holler</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; height: 100vh; }}
  .container {{ width: 600px; text-align: center; }}
  h1 {{ font-size: 48px; font-weight: 700; margin-bottom: 8px; }}
  .sub {{ color: #888; margin-bottom: 32px; font-size: 14px; }}
  .input-row {{ display: flex; gap: 10px; align-items: stretch; }}
  select {{ padding: 12px 14px; font-size: 15px; border: 1px solid #333;
            border-radius: 12px; background: #1a1a1a; color: #fff; outline: none;
            appearance: none; -webkit-appearance: none; cursor: pointer;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 12px center;
            padding-right: 32px; }}
  select:focus {{ border-color: #555; }}
  input {{ flex: 1; padding: 16px 20px; font-size: 18px; border: 1px solid #333;
          border-radius: 12px; background: #1a1a1a; color: #fff; outline: none; }}
  input:focus {{ border-color: #555; }}
  input::placeholder {{ color: #555; }}
  #status {{ margin-top: 16px; font-size: 14px; color: #666; min-height: 20px; }}
  #status.speaking {{ color: #4ade80; }}
  #stats {{ margin-top: 8px; font-size: 13px; color: #555; font-family: monospace; }}
</style></head>
<body><div class="container">
  <h1>Holler</h1>
  <p class="sub">Qwen3-TTS 0.6B &middot; 6-bit &middot; Apple Silicon</p>
  <div class="input-row">
    <select id="voice">{voice_options}</select>
    <input id="text" type="text" placeholder="Type anything and press Enter..." autofocus>
  </div>
  <div id="status"></div>
  <div id="stats"></div>
</div>
<script>
const input = document.getElementById('text');
const voiceSel = document.getElementById('voice');
const status = document.getElementById('status');
const stats = document.getElementById('stats');
let busy = false;
input.addEventListener('keydown', async (e) => {{
  if (e.key !== 'Enter' || busy || !input.value.trim()) return;
  busy = true;
  const text = input.value.trim();
  const voice = voiceSel.value;
  status.textContent = 'Speaking...';
  status.className = 'speaking';
  stats.textContent = '';
  try {{
    const res = await fetch('/speak', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{text, voice}})
    }});
    const data = await res.json();
    status.textContent = '';
    status.className = '';
    stats.textContent = `${{voice}} | TTFA: ${{data.ttfa_ms}}ms | Audio: ${{data.audio_len_s.toFixed(1)}}s`;
  }} catch(err) {{
    status.textContent = 'Error: ' + err.message;
    status.className = '';
  }}
  busy = false;
}});
</script></body></html>"""


def build_html():
    options = "".join(f'<option value="{v}">{v.title()}</option>' for v in VOICES)
    return HTML_TEMPLATE.format(voice_options=options)

HTML = build_html()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        if self.path != "/speak":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("text", "")
        voice = body.get("voice", VOICES[0])
        if voice not in VOICES:
            voice = VOICES[0]

        t0 = time.time()
        ttfa = None
        total_samples = 0

        speech_started = False
        for result in model.generate(
            text=text,
            voice=voice,
            language="english",
            temperature=0.6,
            stream=True,
            streaming_interval=0.1,
        ):
            if hasattr(result, 'audio'):
                chunk = np.array(result.audio).flatten().astype(np.float32)
                peak = np.abs(chunk).max()
                if not speech_started and peak < 0.02:
                    continue  # skip leading silence chunks
                speech_started = True
                if ttfa is None:
                    ttfa = (time.time() - t0) * 1000
                audio_stream.write(chunk.reshape(-1, 1))
                total_samples += len(chunk)

        total_ms = (time.time() - t0) * 1000
        audio_len_s = total_samples / SAMPLE_RATE

        print(f"  [{voice}] TTFA={ttfa:.0f}ms total={total_ms:.0f}ms audio={audio_len_s:.1f}s | {text[:60]}", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "ttfa_ms": round(ttfa) if ttfa else 0,
            "total_ms": round(total_ms),
            "audio_len_s": round(audio_len_s, 2),
        }).encode())

    def log_message(self, format, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
print(f"Listening on http://localhost:{PORT}", flush=True)
server.serve_forever()
