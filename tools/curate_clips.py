#!/usr/bin/env python3
"""Audio curation tool — swipe through training clips to keep/reject/maybe."""

import argparse
import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).parent.parent
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip Tinder — {{VOICE}}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
  height: 100vh; display: flex; flex-direction: column; align-items: center;
  user-select: none; overflow: hidden;
}
.progress-bar {
  width: 100%; height: 6px; background: #16213e; position: fixed; top: 0; z-index: 10;
}
.progress-fill {
  height: 100%; background: linear-gradient(90deg, #e94560, #0f3460); transition: width 0.3s;
}
.stats {
  margin-top: 20px; font-size: 14px; color: #888; display: flex; gap: 24px;
}
.stats span { display: flex; align-items: center; gap: 6px; }
.stat-keep { color: #4ecca3; }
.stat-reject { color: #e94560; }
.stat-maybe { color: #f0c040; }
.voice-name { margin-top: 12px; font-size: 13px; color: #555; text-transform: uppercase; letter-spacing: 2px; }
.card-container {
  flex: 1; display: flex; align-items: center; justify-content: center; width: 100%;
  perspective: 800px;
}
.card {
  width: 520px; max-width: 90vw; background: #16213e; border-radius: 16px;
  padding: 48px 40px; text-align: center; cursor: pointer;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4); transition: transform 0.15s, box-shadow 0.15s;
  position: relative;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.card:active { transform: scale(0.98); }
.clip-number { font-size: 13px; color: #555; margin-bottom: 16px; }
.clip-text {
  font-size: 22px; line-height: 1.5; color: #fff; font-weight: 500;
  min-height: 66px; display: flex; align-items: center; justify-content: center;
}
.clip-filename { font-size: 12px; color: #444; margin-top: 20px; font-family: monospace; }
.decision-badge {
  position: absolute; top: 12px; font-size: 13px; font-weight: 600;
  padding: 4px 12px; border-radius: 20px; opacity: 0; transition: opacity 0.15s;
}
.badge-keep { right: 16px; color: #4ecca3; border: 2px solid #4ecca3; }
.badge-reject { left: 16px; color: #e94560; border: 2px solid #e94560; }
.badge-maybe { right: 16px; color: #f0c040; border: 2px solid #f0c040; }
.flash { position: fixed; inset: 0; pointer-events: none; opacity: 0; transition: opacity 0.25s; z-index: 5; }
.hints {
  padding: 20px; font-size: 13px; color: #555; display: flex; gap: 20px; flex-wrap: wrap;
  justify-content: center;
}
.hints kbd {
  background: #16213e; padding: 2px 8px; border-radius: 4px; font-family: monospace;
  border: 1px solid #333;
}
.done-screen {
  display: none; flex-direction: column; align-items: center; justify-content: center;
  gap: 20px; flex: 1;
}
.done-screen h2 { font-size: 28px; color: #4ecca3; }
.done-screen .summary { font-size: 18px; line-height: 2; }
.done-screen button {
  background: #0f3460; color: #fff; border: none; padding: 12px 28px; border-radius: 8px;
  font-size: 15px; cursor: pointer; margin-top: 8px;
}
.done-screen button:hover { background: #1a4a8a; }
.mode-toggle {
  position: fixed; top: 14px; right: 16px; z-index: 20; display: flex; gap: 8px;
}
.mode-toggle button {
  background: #16213e; color: #888; border: 1px solid #333; padding: 4px 12px;
  border-radius: 6px; font-size: 12px; cursor: pointer;
}
.mode-toggle button.active { color: #fff; border-color: #0f3460; background: #0f3460; }
</style>
</head>
<body>

<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>

<div class="mode-toggle">
  <button id="btn-undecided" class="active" onclick="setMode('undecided')">Undecided</button>
  <button id="btn-maybe" onclick="setMode('maybe')">Maybes</button>
  <button id="btn-all" onclick="setMode('all')">All</button>
</div>

<div class="stats" id="stats"></div>
<div class="voice-name" id="voice-label"></div>

<div class="card-container" id="card-container">
  <div class="card" id="card" onclick="replay()">
    <div class="decision-badge badge-keep" id="badge-keep">KEEP</div>
    <div class="decision-badge badge-reject" id="badge-reject">REJECT</div>
    <div class="decision-badge badge-maybe" id="badge-maybe">MAYBE</div>
    <div class="clip-number" id="clip-number"></div>
    <div class="clip-text" id="clip-text"></div>
    <div class="clip-filename" id="clip-filename"></div>
  </div>
</div>

<div class="done-screen" id="done-screen">
  <h2>All done!</h2>
  <div class="summary" id="done-summary"></div>
  <button onclick="setMode('maybe')">Review maybes</button>
  <button onclick="generateJsonl()">Export train_curated.jsonl</button>
</div>

<div class="flash" id="flash"></div>

<div class="hints">
  <span><kbd>←</kbd> reject</span>
  <span><kbd>→</kbd> keep</span>
  <span><kbd>↓</kbd> maybe</span>
  <span><kbd>shift</kbd> undo</span>
  <span>click card to replay</span>
</div>

<script>
const VOICE = "{{VOICE}}";
let clips = [];
let decisions = {};
let queue = [];
let queueIndex = -1;
let history = [];
let audio = null;
let mode = "undecided";

async function init() {
  document.getElementById("voice-label").textContent = VOICE;
  const res = await fetch("/api/clips");
  const data = await res.json();
  clips = data.clips;
  decisions = data.decisions;
  buildQueue();
  if (queue.length > 0) { queueIndex = 0; showClip(); }
  else showDone();
  updateStats();
}

function buildQueue() {
  if (mode === "undecided") queue = clips.filter(c => !decisions[c.file]);
  else if (mode === "maybe") queue = clips.filter(c => decisions[c.file] === "maybe");
  else queue = [...clips];
  queueIndex = 0;
}

function setMode(m) {
  mode = m;
  document.querySelectorAll(".mode-toggle button").forEach(b => b.classList.remove("active"));
  document.getElementById("btn-" + m).classList.add("active");
  history = [];
  buildQueue();
  document.getElementById("done-screen").style.display = "none";
  document.getElementById("card-container").style.display = "flex";
  if (queue.length > 0) showClip();
  else showDone();
  updateStats();
}

function showClip() {
  if (queueIndex >= queue.length) { showDone(); return; }
  const clip = queue[queueIndex];
  document.getElementById("clip-text").textContent = clip.text;
  document.getElementById("clip-filename").textContent = clip.file;

  const totalInMode = queue.length;
  const posInMode = queueIndex + 1;
  document.getElementById("clip-number").textContent = posInMode + " / " + totalInMode;

  document.getElementById("badge-keep").style.opacity = decisions[clip.file] === "keep" ? "1" : "0";
  document.getElementById("badge-reject").style.opacity = decisions[clip.file] === "reject" ? "1" : "0";
  document.getElementById("badge-maybe").style.opacity = decisions[clip.file] === "maybe" ? "1" : "0";

  playAudio(clip.file);
  updateStats();
}

function playAudio(file) {
  if (audio) { audio.pause(); audio = null; }
  audio = new Audio("/audio/" + encodeURIComponent(file));
  audio.play().catch(() => {});
}

function replay() {
  if (queueIndex >= 0 && queueIndex < queue.length) playAudio(queue[queueIndex].file);
}

function flash(color) {
  const el = document.getElementById("flash");
  el.style.background = color;
  el.style.opacity = "0.15";
  setTimeout(() => el.style.opacity = "0", 250);
}

async function decide(decision) {
  if (queueIndex >= queue.length) return;
  const clip = queue[queueIndex];
  decisions[clip.file] = decision;
  history.push({ index: queueIndex, file: clip.file, prev: null });

  if (decision === "keep") flash("rgba(78,204,163,1)");
  else if (decision === "reject") flash("rgba(233,69,96,1)");
  else flash("rgba(240,192,64,1)");

  await fetch("/api/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: clip.file, decision })
  });

  if (mode === "undecided" || mode === "maybe") {
    queue.splice(queueIndex, 1);
  } else {
    queueIndex++;
  }

  if (queueIndex >= queue.length) showDone();
  else showClip();
  updateStats();
}

function undo() {
  if (history.length === 0) return;
  const last = history.pop();
  if (mode === "undecided" || mode === "maybe") {
    queue.splice(last.index, 0, clips.find(c => c.file === last.file));
    queueIndex = last.index;
  } else {
    queueIndex = Math.max(0, queueIndex - 1);
  }
  delete decisions[last.file];
  fetch("/api/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: last.file, decision: null })
  });
  document.getElementById("done-screen").style.display = "none";
  document.getElementById("card-container").style.display = "flex";
  showClip();
}

function updateStats() {
  const keeps = Object.values(decisions).filter(d => d === "keep").length;
  const rejects = Object.values(decisions).filter(d => d === "reject").length;
  const maybes = Object.values(decisions).filter(d => d === "maybe").length;
  const total = clips.length;
  const decided = keeps + rejects + maybes;
  document.getElementById("stats").innerHTML =
    `<span>${decided}/${total} decided</span>` +
    `<span class="stat-keep">● ${keeps} keep</span>` +
    `<span class="stat-reject">● ${rejects} reject</span>` +
    `<span class="stat-maybe">● ${maybes} maybe</span>`;
  document.getElementById("progress").style.width = (decided / total * 100) + "%";
}

function showDone() {
  document.getElementById("card-container").style.display = "none";
  const screen = document.getElementById("done-screen");
  screen.style.display = "flex";
  const keeps = Object.values(decisions).filter(d => d === "keep").length;
  const rejects = Object.values(decisions).filter(d => d === "reject").length;
  const maybes = Object.values(decisions).filter(d => d === "maybe").length;
  const undecided = clips.length - keeps - rejects - maybes;
  let html = `<span class="stat-keep">${keeps} keep</span><br>`;
  html += `<span class="stat-reject">${rejects} reject</span><br>`;
  html += `<span class="stat-maybe">${maybes} maybe</span>`;
  if (undecided > 0) html += `<br><span>${undecided} undecided</span>`;
  if (mode !== "undecided" && mode !== "maybe")
    html += `<br><small style="color:#555">End of list</small>`;
  document.getElementById("done-summary").innerHTML = html;
}

async function generateJsonl() {
  const res = await fetch("/api/export", { method: "POST" });
  const data = await res.json();
  alert("Exported " + data.count + " clips to train_curated.jsonl");
}

document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { e.preventDefault(); decide("reject"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); decide("keep"); }
  else if (e.key === "ArrowDown") { e.preventDefault(); decide("maybe"); }
  else if (e.key === "Shift") { e.preventDefault(); undo(); }
  else if (e.key === " " || e.key === "Alt") { e.preventDefault(); replay(); }
});

init();
</script>
</body>
</html>"""


class CurationHandler(SimpleHTTPRequestHandler):
    voice = None
    voice_dir = None
    clips = None
    decisions = None
    curation_path = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = HTML_PAGE.replace("{{VOICE}}", self.voice)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif parsed.path == "/api/clips":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "clips": self.clips,
                "decisions": self.decisions,
            }).encode())
        elif parsed.path.startswith("/audio/"):
            filename = parsed.path[7:]
            filepath = self.voice_dir / "audio" / filename
            if filepath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(filepath.stat().st_size))
                self.end_headers()
                self.wfile.write(filepath.read_bytes())
            else:
                self.send_error(404)
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == "/api/decide":
            filename = body.get("file")
            decision = body.get("decision")
            if decision is None:
                self.decisions.pop(filename, None)
            else:
                self.decisions[filename] = decision
            save_decisions(self.curation_path, self.decisions)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        elif parsed.path == "/api/export":
            count = export_curated(self.voice_dir, self.clips, self.decisions)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"count": count}).encode())
        else:
            self.send_error(404)


def save_decisions(path, decisions):
    with open(path, "w") as f:
        json.dump(decisions, f, indent=2)


def export_curated(voice_dir, clips, decisions):
    jsonl_path = voice_dir / "train_curated.jsonl"
    count = 0
    with open(jsonl_path, "w") as f:
        for clip in clips:
            if decisions.get(clip["file"]) == "keep":
                f.write(json.dumps({
                    "audio": f"./audio/{clip['file']}",
                    "text": clip["text"],
                    "ref_audio": "./ref.wav",
                }) + "\n")
                count += 1
    print(f"Exported {count} clips to {jsonl_path}")
    return count


def load_clips(voice_dir):
    jsonl_path = voice_dir / "train.jsonl"
    clips = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            filename = os.path.basename(entry["audio"])
            clips.append({"file": filename, "text": entry["text"]})
    return clips


def main():
    parser = argparse.ArgumentParser(description="Audio Curation Tool")
    parser.add_argument("--voice", "-v", required=True, help="Voice name (e.g. katie, joe)")
    parser.add_argument("--port", "-p", type=int, default=8200)
    args = parser.parse_args()

    voice_dir = PROJECT_ROOT / "voices" / args.voice / "training-data"
    if not voice_dir.exists():
        print(f"Error: {voice_dir} not found")
        return

    curation_path = voice_dir / "curation.json"
    decisions = {}
    if curation_path.exists():
        with open(curation_path) as f:
            decisions = json.load(f)

    clips = load_clips(voice_dir)

    CurationHandler.voice = args.voice
    CurationHandler.voice_dir = voice_dir
    CurationHandler.clips = clips
    CurationHandler.decisions = decisions
    CurationHandler.curation_path = curation_path

    server = HTTPServer(("127.0.0.1", args.port), CurationHandler)
    url = f"http://localhost:{args.port}"
    print(f"Clip Tinder — {args.voice} ({len(clips)} clips, {len(decisions)} already decided)")
    print(f"→ {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{len(decisions)} decisions saved to {curation_path}")


if __name__ == "__main__":
    main()
