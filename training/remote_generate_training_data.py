#!/usr/bin/env python3
"""Fast GPU training data generation for Holler voices.

Uses faster-qwen3-tts (CUDA graphs + StaticCache) with multiple model
instances on a single GPU for maximum throughput.

The 1.7B-Base model uses ~5-6 GB VRAM per instance. A 3090 (24 GB) fits
2-3 instances. Each instance achieves ~3-4x RTF via CUDA graphs; parallel
instances multiply wall-clock throughput.

Requires: faster-qwen3-tts, soundfile, numpy, torch (all installed by remote_setup.sh)

Usage:
  # Quick test — 100 clips, 3 workers
  python remote_generate_training_data.py \\
    --ref-audio /workspace/voices/nora/ref.wav \\
    --ref-text "Oh wow, that actually worked!" \\
    --output /workspace/output/nora

  # Full run — all 500 corpus texts
  python remote_generate_training_data.py \\
    --ref-audio /workspace/voices/nora/ref.wav \\
    --ref-text "Oh wow, that actually worked!" \\
    --output /workspace/output/nora \\
    --count 500

  # Resume interrupted run
  python remote_generate_training_data.py \\
    --ref-audio /workspace/voices/nora/ref.wav \\
    --ref-text "Oh wow, that actually worked!" \\
    --output /workspace/output/nora \\
    --count 500 --start 200
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp


def gpu_info():
    """Get GPU name and memory without initializing CUDA in main process."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        name, mem_mb = out.split(", ")
        return name.strip(), float(mem_mb) / 1024
    except Exception:
        return "Unknown GPU", 0.0


def fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    return f"{int(s) // 60}m {int(s) % 60}s"


# ──────────────────────────────────────────────────────────────────────
# Worker process
# ──────────────────────────────────────────────────────────────────────

def worker(wid, model_path, ref_audio, ref_text, temp, text_q, status_q, out_dir,
           silence_pad_ms=100):
    """Load model, warm up CUDA graphs, generate clips from queue."""
    import numpy as np
    import soundfile as sf
    from faster_qwen3_tts import FasterQwen3TTS

    torch.cuda.set_device(0)

    # Pre-pad ref audio with silence (replaces library's 0.5s append_silence)
    if silence_pad_ms > 0:
        ref_data, ref_sr = sf.read(ref_audio)
        pad = np.zeros(int(ref_sr * silence_pad_ms / 1000), dtype=ref_data.dtype)
        padded = np.concatenate([ref_data, pad])
        padded_path = f"/tmp/ref_padded_w{wid}.wav"
        sf.write(padded_path, padded, ref_sr)
        ref_audio = padded_path

    def vram_used():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9

    # ── Load ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model = FasterQwen3TTS.from_pretrained(
            model_path, device="cuda", dtype=torch.bfloat16,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        status_q.put(("oom", wid, "load", 0.0, 0.0))
        return
    load_s = time.time() - t0

    # ── Warmup (triggers CUDA graph capture) ──────────────────────────
    t0 = time.time()
    try:
        model.generate_voice_clone(
            text="Hello, this is a quick warmup generation for graph capture.",
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temp,
            max_new_tokens=2048,
            repetition_penalty=1.05,
            xvec_only=False,
            append_silence=False,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        status_q.put(("oom", wid, "warmup", load_s, 0.0))
        return
    except Exception as e:
        status_q.put(("fatal", wid, f"warmup failed: {e}", load_s, 0.0))
        return
    warmup_s = time.time() - t0

    status_q.put(("ready", wid, load_s, warmup_s, vram_used()))

    # ── Generate ──────────────────────────────────────────────────────
    while True:
        item = text_q.get()
        if item is None:
            break

        idx, text = item
        label = f"clip_{idx + 1:04d}"
        t0 = time.time()

        try:
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=temp,
                max_new_tokens=2048,
                repetition_penalty=1.05,
                xvec_only=False,
                append_silence=False,
            )
            elapsed = time.time() - t0
            audio = audio_list[0]
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = audio.astype("float32").squeeze()
            dur = len(audio) / sr

            sf.write(str(Path(out_dir) / f"{label}.wav"), audio, sr)
            status_q.put(("done", wid, idx, label, text, dur, elapsed))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            status_q.put(("fail", wid, idx, label, text, "OOM", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            status_q.put(("fail", wid, idx, label, text, str(e), elapsed))

    status_q.put(("exit", wid))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fast GPU training data generation for Holler voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ref-audio", required=True, help="Reference WAV path")
    ap.add_argument("--ref-text", required=True, help="Transcript of ref audio")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--model", default="/workspace/models/1.7B-Base",
                    help="Model path or HF ID (default: /workspace/models/1.7B-Base)")
    ap.add_argument("--workers", type=int, default=3,
                    help="Model instances to load (default: 3, falls back on OOM)")
    ap.add_argument("--count", type=int, default=100,
                    help="Clips to generate (default: 100)")
    ap.add_argument("--start", type=int, default=0,
                    help="Corpus start index for resume")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--corpus", default=None,
                    help="Path to corpus.json (default: <script_dir>/corpus.json)")
    args = ap.parse_args()

    # ── Load corpus ───────────────────────────────────────────────────
    corpus_path = Path(args.corpus) if args.corpus else Path(__file__).parent / "corpus.json"
    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found")
        print(f"Upload corpus.json alongside this script, or pass --corpus <path>")
        sys.exit(1)
    all_texts = json.loads(corpus_path.read_text())

    texts = all_texts[args.start : args.start + args.count]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume support ────────────────────────────────────────────────
    existing = {f.stem for f in out_dir.glob("clip_*.wav")}
    work = [(args.start + i, t) for i, t in enumerate(texts)
            if f"clip_{args.start + i + 1:04d}" not in existing]
    skipped = len(texts) - len(work)

    ref_path = Path(args.ref_audio)
    if not ref_path.exists():
        print(f"ERROR: ref audio not found: {ref_path}")
        sys.exit(1)

    gpu_name, gpu_gb = gpu_info()

    print()
    print("=" * 58)
    print("  Holler — GPU Training Data Generator")
    print("=" * 58)
    print()
    print(f"  GPU:         {gpu_name} ({gpu_gb:.0f} GB)")
    print(f"  Model:       {args.model}")
    print(f"  Ref audio:   {args.ref_audio}")
    print(f"  Output:      {out_dir}")
    print(f"  Workers:     {args.workers} (target)")
    print(f"  Clips:       {len(work)} to generate" +
          (f" ({skipped} exist, skipped)" if skipped else ""))
    print(f"  Corpus:      {len(all_texts)} texts, using [{args.start}:{args.start + args.count}]")
    print(f"  Temperature: {args.temperature}")
    print()

    if not work:
        print("  All clips already exist. Nothing to do.")
        return

    # ── Launch workers ────────────────────────────────────────────────
    mp.set_start_method("spawn", force=True)
    text_q = mp.Queue()
    status_q = mp.Queue()

    procs = []
    for i in range(args.workers):
        p = mp.Process(
            target=worker,
            args=(i, args.model, str(ref_path), args.ref_text,
                  args.temperature, text_q, status_q, str(out_dir)),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # ── Wait for workers to load + warmup ─────────────────────────────
    print("  Loading models + CUDA graph warmup...")
    print()
    active = 0
    total_load = 0.0
    total_warmup = 0.0

    for _ in range(args.workers):
        try:
            msg = status_q.get(timeout=600)
        except Exception:
            print("  Timeout waiting for worker — aborting")
            break

        if msg[0] == "ready":
            _, wid, load_s, warmup_s, vram = msg
            total_load += load_s
            total_warmup += warmup_s
            active += 1
            print(f"  Worker {wid}  ✓  load {load_s:5.1f}s   warmup {warmup_s:5.1f}s   "
                  f"VRAM {vram:.1f}/{gpu_gb:.0f} GB")

        elif msg[0] == "oom":
            _, wid, phase, load_s, warmup_s = msg
            print(f"  Worker {wid}  ✗  OOM during {phase}")

        elif msg[0] == "fatal":
            _, wid, err, load_s, warmup_s = msg
            print(f"  Worker {wid}  ✗  {err}")

    if active == 0:
        print("\n  ERROR: No workers available. GPU may not have enough VRAM.")
        for p in procs:
            p.join(timeout=5)
        sys.exit(1)

    print(f"\n  {active} worker{'s' if active > 1 else ''} ready "
          f"(load {total_load / active:.1f}s + warmup {total_warmup / active:.1f}s avg)")
    print()

    # ── Feed work queue ───────────────────────────────────────────────
    for item in work:
        text_q.put(item)
    for _ in range(active):
        text_q.put(None)

    # ── Monitor progress ──────────────────────────────────────────────
    total = len(work)
    report_every = max(1, total // 25)

    done = 0
    failed = 0
    total_audio = 0.0
    total_gen = 0.0
    results = []
    exited = 0
    gen_start = time.time()

    print(f"  Generating {total} clips with {active} workers...")
    print()

    try:
        while exited < active:
            msg = status_q.get(timeout=300)

            if msg[0] == "done":
                _, wid, idx, label, text, dur, elapsed = msg
                done += 1
                total_audio += dur
                total_gen += elapsed
                results.append({
                    "idx": idx, "label": label, "text": text,
                    "duration": dur, "gen_time": elapsed, "worker": wid,
                })

                if done % report_every == 0 or done == total - failed:
                    wall = time.time() - gen_start
                    wall_rtf = total_audio / wall if wall > 0 else 0
                    avg = wall / done
                    left = (total - done - failed) * avg / active
                    print(f"  [{done + failed:4d}/{total}]  "
                          f"{avg:.2f}s/clip  wall RTF {wall_rtf:.1f}x  "
                          f"~{fmt_time(left)} left")

            elif msg[0] == "fail":
                _, wid, idx, label, text, err, elapsed = msg
                done += 1
                failed += 1
                print(f"  FAIL {label} [w{wid}]: {err}")

            elif msg[0] == "exit":
                exited += 1

    except KeyboardInterrupt:
        print("\n\n  Interrupted — writing partial results...")

    gen_wall = time.time() - gen_start

    # ── Write manifest ────────────────────────────────────────────────
    results.sort(key=lambda r: r["idx"])
    manifest = out_dir / "train.jsonl"
    mode = "a" if skipped > 0 and manifest.exists() else "w"
    with open(manifest, mode) as f:
        for r in results:
            f.write(json.dumps({
                "audio": f"./{r['label']}.wav",
                "text": r["text"],
                "ref_audio": str(args.ref_audio),
                "ref_text": args.ref_text,
            }) + "\n")

    # ── Summary ───────────────────────────────────────────────────────
    n_ok = done - failed
    wall_rtf = total_audio / gen_wall if gen_wall > 0 else 0
    per_worker_rtf = total_audio / total_gen if total_gen > 0 else 0

    print()
    print("=" * 58)
    print(f"  Done: {n_ok} clips in {fmt_time(gen_wall)}")
    print()
    print(f"  Audio generated:  {total_audio / 60:.1f} min")
    print(f"  Wall-clock RTF:   {wall_rtf:.1f}x")
    print(f"  Per-worker RTF:   ~{per_worker_rtf:.1f}x × {active} workers")
    print(f"  Avg wall time:    {gen_wall / max(1, n_ok):.2f}s per clip")
    print(f"  Failed:           {failed}")
    print()
    print(f"  Output:           {out_dir}")
    print(f"  Manifest:         {manifest} ({len(results)} entries)")
    print("=" * 58)
    print()

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)


if __name__ == "__main__":
    main()
