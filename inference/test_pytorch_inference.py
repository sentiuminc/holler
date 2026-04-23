"""Test PyTorch inference on the fine-tuned ivi_female model."""
import sys
import time
import torch
import soundfile as sf

print("[1/3] imports done", flush=True)

from qwen_tts import Qwen3TTSModel

model_path = sys.argv[1]
out_dir = sys.argv[2]

t_load = time.time()
tts = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print(f"[2/3] model loaded in {time.time()-t_load:.1f}s from {model_path}", flush=True)

texts = [
    ("greeting", "Hello! I am your personal AI assistant."),
    ("short", "Sure, let me check that for you."),
    ("question", "What kind of music are you in the mood for?"),
    ("technical", "Your CPU usage is at 47 percent and you have 3 meetings this afternoon."),
]

for name, text in texts:
    torch.cuda.synchronize()
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=text,
        language="English",
        speaker="ivi_female",
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    path = f"{out_dir}/ivi_{name}.wav"
    sf.write(path, wavs[0], sr)
    print(f"[{name}] gen={elapsed:.1f}s sr={sr} audio_len={len(wavs[0])/sr:.2f}s", flush=True)

print("[3/3] DONE", flush=True)
