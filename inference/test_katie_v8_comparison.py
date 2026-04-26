"""Generate samples from Katie v8 with the same texts as the built-in voice test."""
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from mlx_audio.tts import load
import mlx.core as mx

CHECKPOINT = str(Path(__file__).parent.parent / "checkpoints" / "katie-v8-epoch-1.2-6bit")
OUT = Path.home() / "Downloads" / "katie-v8-comparison-builtin"
OUT.mkdir(exist_ok=True)

TEXTS = [
    ("short", "Got it."),
    ("medium", "I ran the tests and everything passed, but I want to double check one more thing."),
    ("long", "So the way this works is pretty straightforward. You give it the input, it processes everything in the background, and then you get the results back in real time."),
    ("sibilant", "I want to make sure I understand correctly before making any changes."),
    ("question", "Does this match what you were expecting?"),
]

print(f"Loading {CHECKPOINT}...")
t0 = time.time()
model = load(CHECKPOINT)
print(f"Loaded in {time.time()-t0:.1f}s\n")

print("=== katie-v8 ===")
for name, text in TEXTS:
    t0 = time.time()
    chunks = []
    for result in model.generate(
        text=text,
        voice="katie",
        language="english",
        temperature=0.6,
        stream=False,
    ):
        if hasattr(result, 'audio'):
            chunks.append(result.audio)
    elapsed = time.time() - t0
    audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
    path = OUT / f"katie-v8_{name}.wav"
    sf.write(str(path), audio, 24000)
    dur = len(audio) / 24000
    flag = " <-- MAX?" if dur > 19 else ""
    print(f"  [{name}] {elapsed:.2f}s, {dur:.2f}s audio{flag}")

print(f"\nDone! Samples at {OUT}")
