# coding=utf-8
# Qwen3-TTS 0.6B-Base fine-tuning — WORKING recipe for custom voice.
#
# THIS IS A PATCHED VERSION of upstream finetuning/sft_12hz.py with ONE change:
#   Line ~89 wraps `text_embedding` with `text_projection` — required for 0.6B
#   because text_embedding is dim 2048 while codec_embedding is dim 1024.
#   Without this patch, 0.6B training fails with a dimension mismatch error.
#
# DO NOT apply other "fixes" circulating in the community (double-shift fix,
# sub-codebook loop removal). At low learning rates those break training.
#
# Winning recipe (confirmed on 2026-04-21 by Chris Nagy, ivi project):
#   python3 sft_12hz_patched.py \
#     --init_model_path /workspace/models/0.6B-Base \
#     --output_model_path /workspace/output \
#     --train_jsonl /workspace/training-data/train_with_codes.jsonl \
#     --batch_size 2 \
#     --lr 1e-7 \
#     --num_epochs 2 \
#     --speaker_name katie
#
# Voice naming convention: fantasy names (katie, joe, ...). Default is katie —
# the first ivi voice, at slot 3000. Future voices get 3001, 3002, ... in the
# 3000–3071 custom-voice region.
#
# Epoch 1 checkpoint is the pick: voice accurate, emotion present, EOS clean.
# See ~/Desktop/Files/AI/ivi/audio-test/qwen3-tts-finetune/README.md for full notes.

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--speaker_name", type=str, default="katie")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # THE ONLY PATCH: text_projection wrap for 0.6B dimension mismatch
                input_text_embedding = model.talker.text_projection(model.talker.model.text_embedding(input_text_ids)) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
