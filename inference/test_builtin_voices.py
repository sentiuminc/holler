"""Generate samples from Qwen3-TTS 0.6B CustomVoice built-in English voices.
Baseline comparison: what does the model sound like WITHOUT fine-tuning?"""
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from mlx_audio.tts import load
import mlx.core as mx

MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
OUT = Path.home() / "Downloads" / "qwen3-builtin-voices"
OUT.mkdir(exist_ok=True)

VOICES = ["aiden", "ryan", "serena", "vivian"]

TEXTS = [
    ("short", "Got it."),
    ("medium", "I ran the tests and everything passed, but I want to double check one more thing."),
    ("long", "So the way this works is pretty straightforward. You give it the input, it processes everything in the background, and then you get the results back in real time."),
    ("sibilant", "I want to make sure I understand correctly before making any changes."),
    ("question", "Does this match what you were expecting?"),
]

print(f"Loading {MODEL}...")
t0 = time.time()
model = load(MODEL)
print(f"Loaded in {time.time()-t0:.1f}s\n")

for voice in VOICES:
    print(f"=== {voice} ===")
    for name, text in TEXTS:
        t0 = time.time()
        chunks = []
        for result in model.generate(
            text=text,
            voice=voice,
            language="english",
            temperature=0.6,
            stream=False,
        ):
            if hasattr(result, 'audio'):
                chunks.append(result.audio)
        elapsed = time.time() - t0
        audio = np.array(mx.concatenate(chunks, axis=-1)).flatten()
        path = OUT / f"{voice}_{name}.wav"
        sf.write(str(path), audio, 24000)
        dur = len(audio) / 24000
        flag = " <-- MAX?" if dur > 19 else ""
        print(f"  [{name}] {elapsed:.2f}s, {dur:.2f}s audio{flag}")
    print()

print(f"Done! Samples at {OUT}")
