"""Test v6 epochs 0, 1, 2 with diverse texts."""
import time
import os
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

epochs_to_test = [0, 1, 2]
out_dir = "/workspace/ivi_v6"
os.makedirs(out_dir, exist_ok=True)

texts = [
    ("greeting", "Hello! I am your personal AI assistant."),
    ("short", "Sure, let me check that for you."),
    ("question", "What kind of music are you in the mood for?"),
    ("long", "Your CPU usage is at 47 percent and you have three meetings this afternoon."),
    ("emotion", "Oh wow, that's really exciting news!"),
]

for e in epochs_to_test:
    model_path = f"/workspace/output/checkpoint-epoch-{e}"
    if not os.path.exists(model_path):
        print(f"SKIP epoch {e}: checkpoint missing", flush=True)
        continue
    print(f"\n=== Epoch {e} ===", flush=True)

    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    for name, text in texts:
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            wavs, sr = tts.generate_custom_voice(
                text=text,
                language="English",
                speaker="ivi_female",
                max_new_tokens=256,
            )
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            path = f"{out_dir}/e{e:02d}_{name}.wav"
            sf.write(path, wavs[0], sr)
            audio_len = len(wavs[0])/sr
            flag = " <-- HIT MAX" if audio_len > 19 else ""
            print(f"  [{name}] gen={elapsed:.1f}s len={audio_len:.2f}s{flag}", flush=True)
        except Exception as exc:
            print(f"  [{name}] ERROR: {exc}", flush=True)

    del tts
    torch.cuda.empty_cache()

print("\nDONE", flush=True)
