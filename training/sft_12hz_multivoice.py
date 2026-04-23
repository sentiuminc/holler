# coding=utf-8
# Qwen3-TTS 0.6B-Base multi-voice fine-tuning.
# Adapts sft_12hz_patched.py (rekuenkdr single-voice recipe) to train N voices jointly.
#
# Key changes vs sft_12hz_patched.py:
#   1. JSONL entries must include a `voice_name` field per sample
#   2. Dataset passes voice_name through batch (wrapped collate_fn)
#   3. target_speaker_embeddings is a dict: voice_name -> first-seen embedding
#   4. At checkpoint save, each voice writes to its own slot (3000+i), and
#      spk_id / spk_is_dialect config maps are populated for all voices
#
# Same upstream patch as single-voice: text_projection wrap for 0.6B dim mismatch.
# Same recipe: lr=1e-7, 2 epochs, batch_size=2, grad_accum=4.

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


class VoiceAwareDataset(TTSDataset):
    """Wraps TTSDataset to carry voice_name per sample through collate."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["voice_name"] = self.data_list[idx].get("voice_name", "speaker_default")
        return item

    def collate_fn(self, batch):
        voice_names = [b.pop("voice_name") for b in batch]
        # Pad ref_mels to the max time dimension in this batch.
        # Upstream dataset returns ref_mel of shape [1, T, 128] (after mel_spectrogram
        # + transpose(1,2)). T varies per-voice since different voices have different
        # ref.wav durations, so pad dim 1 before super's torch.cat(ref_mels, dim=0).
        max_T = max(b["ref_mel"].shape[1] for b in batch)
        for b in batch:
            ref = b["ref_mel"]
            cur_T = ref.shape[1]
            if cur_T < max_T:
                pad_amount = max_T - cur_T
                # pytorch pad arg is reversed: (last_left, last_right, dim1_left, dim1_right)
                # ref is [1, T, 128]. We pad dim 1 (T) on the right.
                b["ref_mel"] = torch.nn.functional.pad(
                    ref, (0, 0, 0, pad_amount), mode="constant", value=0
                )
        collated = super().collate_fn(batch)
        collated["voice_names"] = voice_names
        return collated


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="JSONL with per-sample voice_name field")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--voice_slot_map_json", type=str, required=True,
                        help='JSON string mapping voice_name -> slot, e.g. \'{"katie":3000,"joe":3001}\'')
    args = parser.parse_args()

    voice_slot_map = json.loads(args.voice_slot_map_json)
    print(f"Voice slot map: {voice_slot_map}")

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = [json.loads(line) for line in open(args.train_jsonl)]
    # Sanity: every sample must have a voice_name, and every voice_name must be in voice_slot_map
    seen_voices = {entry.get("voice_name", "???") for entry in train_data}
    missing = seen_voices - set(voice_slot_map.keys())
    if missing:
        raise ValueError(f"Voices in JSONL but not in voice_slot_map: {missing}")
    print(f"Training {len(train_data)} samples across voices: {sorted(seen_voices)}")

    dataset = VoiceAwareDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    # Dict of voice_name -> first-seen speaker embedding (shape [1, D])
    target_speaker_embeddings = {}

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]
                voice_names = batch["voice_names"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()

                # Cache per-voice first-seen embedding
                for b_idx, vname in enumerate(voice_names):
                    if vname not in target_speaker_embeddings:
                        target_speaker_embeddings[vname] = speaker_embedding[b_idx:b_idx+1]

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # PATCH: text_projection wrap for 0.6B dim mismatch
                input_text_embedding = model.talker.text_projection(
                    model.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask
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
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                voices_seen = list(target_speaker_embeddings.keys())
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | voices cached: {voices_seen}"
                )

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            # Update config.json with multi-voice spk_id / spk_is_dialect maps
            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = dict(voice_slot_map)
            talker_config["spk_is_dialect"] = {v: False for v in voice_slot_map}
            config_dict["talker_config"] = talker_config
            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            for k in [k for k in state_dict if k.startswith(drop_prefix)]:
                del state_dict[k]

            # Write each voice's embedding to its own slot in codec_embedding
            weight = state_dict["talker.model.codec_embedding.weight"]
            for vname, slot in voice_slot_map.items():
                if vname not in target_speaker_embeddings:
                    print(f"WARNING: voice {vname} has no cached embedding (no training samples?)")
                    continue
                emb = target_speaker_embeddings[vname][0].detach().to(weight.device).to(weight.dtype)
                state_dict["talker.model.codec_embedding.weight"][slot] = emb
                print(f"  wrote voice '{vname}' → slot {slot}")

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            print(f"Saved epoch {epoch} checkpoint → {output_dir}")


if __name__ == "__main__":
    train()
