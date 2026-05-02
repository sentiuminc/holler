# coding=utf-8
# Qwen3-TTS 0.6B-Base single-voice fine-tuning.
# Based on upstream finetuning/sft_12hz.py with text_projection wrap for 0.6B
# (text_embedding is dim 2048, codec_embedding is dim 1024).
#
# For multi-voice training, use sft_12hz_multivoice.py instead.
#
# Usage:
#   python3 sft_12hz.py \
#     --init_model_path /workspace/models/0.6B-Base \
#     --output_model_path /workspace/output \
#     --train_jsonl /workspace/training-data/train_with_codes.jsonl \
#     --batch_size 2 \
#     --lr 1e-7 \
#     --num_epochs 2 \
#     --speaker_name kit

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
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save checkpoint every N steps (0=epoch-only)")
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
    steps_per_epoch = len(train_dataloader)
    model.train()

    def save_checkpoint(label):
        if not accelerator.is_main_process:
            return
        output_dir = os.path.join(args.output_model_path, f"checkpoint-{label}")
        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
        input_config_file = os.path.join(MODEL_PATH, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {args.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {args.speaker_name: False}
        config_dict["talker_config"] = talker_config
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
        keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
        for k in keys_to_drop:
            del state_dict[k]
        weight = state_dict['talker.model.codec_embedding.weight']
        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        accelerator.print(f"  Saved checkpoint: {label}")

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
                accelerator.print(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | Loss: {loss.item():.4f}")

            if args.save_every_steps > 0 and (step + 1) % args.save_every_steps == 0 and (step + 1) < steps_per_epoch:
                frac = epoch + (step + 1) / steps_per_epoch
                save_checkpoint(f"epoch-{frac:.1f}")

        save_checkpoint(f"epoch-{epoch}")

if __name__ == "__main__":
    train()
