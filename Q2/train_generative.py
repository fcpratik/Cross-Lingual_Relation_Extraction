"""
Task 2: Relation Extraction via Autoregressive Generation
==========================================================
- Base model: Qwen/Qwen2.5-1.5B with LoRA
- Fine-tune to generate relation labels as text given sentence + entity pairs
- Evaluated on: en, hi, kn, tcy, or
- Trains on English data, adapts using Indic labeled data
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter

# ============================================================
# Config
# ============================================================

class Config:
    model_name = "Qwen/Qwen2.5-1.5B"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    batch_size = 4
    gradient_accumulation_steps = 8  # effective batch = 32
    learning_rate = 2e-4
    num_epochs = 3
    max_input_len = 220
    max_output_len = 40
    max_seq_len = 260  # input + output
    warmup_ratio = 0.06
    weight_decay = 0.01
    max_grad_norm = 1.0

    max_en_samples = 60000
    max_train_minutes = 150
    seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Loading
# ============================================================

def load_label_maps(sft_dir):
    indic_to_en = {}
    en_to_indic = {}
    if not os.path.isdir(sft_dir):
        return indic_to_en, en_to_indic
    for fname in os.listdir(sft_dir):
        if fname.endswith("_map.json"):
            lang = fname.replace("_map.json", "")
            with open(os.path.join(sft_dir, fname), "r", encoding="utf-8") as f:
                mapping = json.load(f)
            en_to_indic[lang] = mapping
            for en_l, indic_l in mapping.items():
                indic_to_en[indic_l] = en_l
    return indic_to_en, en_to_indic


def load_jsonl(filepath):
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return entries
    try:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
    except json.JSONDecodeError:
        pass
    buffer = ""
    brace_count = 0
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        buffer += stripped + " "
        brace_count += stripped.count("{") - stripped.count("}")
        if brace_count == 0 and buffer.strip():
            try:
                entries.append(json.loads(buffer.strip()))
            except json.JSONDecodeError:
                pass
            buffer = ""
    return entries


def build_label_set(en_train_file, sft_dir):
    labels = set()
    if os.path.exists(en_train_file):
        for entry in load_jsonl(en_train_file):
            for rm in entry.get("relationMentions", []):
                lbl = rm.get("label", "")
                if lbl:
                    labels.add(lbl)
    if os.path.isdir(sft_dir):
        for fname in os.listdir(sft_dir):
            if fname.endswith("_map.json"):
                with open(os.path.join(sft_dir, fname), "r", encoding="utf-8") as f:
                    labels.update(json.load(f).keys())
    labels.discard("")
    return sorted(labels)


def format_input(sent, em1, em2):
    """Format input prompt for the model."""
    return (
        f"Extract the relation between the two entities in the sentence.\n"
        f"Sentence: {sent}\n"
        f"Entity 1: {em1}\n"
        f"Entity 2: {em2}\n"
        f"Relation:"
    )


def format_output(label):
    """Format expected output."""
    return f" {label}"


def load_training_samples(en_train_file, sft_dir, indic_to_en, max_en):
    samples = []  # list of (input_text, output_text)

    # English
    print("Loading English data...")
    en_entries = load_jsonl(en_train_file)
    en_samples = []
    for entry in en_entries:
        sent = entry.get("sentText", "")
        for rm in entry.get("relationMentions", []):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            label = rm.get("label", "NA")
            inp = format_input(sent, em1, em2)
            out = format_output(label)
            en_samples.append((inp, out))

    if len(en_samples) > max_en:
        random.shuffle(en_samples)
        en_samples = en_samples[:max_en]
    samples.extend(en_samples)
    print(f"  English: {len(en_samples)}")

    # Indic languages (all available for Task 2)
    indic_samples = []
    for lang in ["hi", "kn", "or", "tcy"]:
        candidates = [
            os.path.join(sft_dir, f"{lang}_train.jsonl"),
            os.path.join(sft_dir, f"{lang}_val.jsonl"),
        ]
        lang_file = None
        for c in candidates:
            if os.path.exists(c):
                lang_file = c
                break
        if not lang_file:
            continue
        print(f"Loading {lang} from {lang_file}...")
        for entry in load_jsonl(lang_file):
            sent = entry.get("sentText", "")
            for rm in entry.get("relationMentions", []):
                em1 = rm.get("em1Text", "")
                em2 = rm.get("em2Text", "")
                indic_label = rm.get("label", "NA")
                # Convert to English label for training
                en_label = indic_to_en.get(indic_label, indic_label)
                inp = format_input(sent, em1, em2)
                out = format_output(en_label)
                indic_samples.append((inp, out))

    print(f"  Indic (raw): {len(indic_samples)}")
    if indic_samples and len(en_samples) > 0:
        factor = min(20, max(1, len(en_samples) // (5 * max(len(indic_samples), 1))))
        indic_samples = indic_samples * factor
        print(f"  Indic oversampled {factor}x: {len(indic_samples)}")
    samples.extend(indic_samples)

    random.shuffle(samples)
    print(f"  Total: {len(samples)}")
    return samples


# ============================================================
# Dataset
# ============================================================

class GenerativeREDataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_len, max_seq_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_text, out_text = self.samples[idx]
        full_text = inp_text + out_text

        # Tokenize input (for masking) and full sequence
        input_enc = self.tokenizer(
            inp_text, max_length=self.max_input_len,
            truncation=True, add_special_tokens=False,
        )
        full_enc = self.tokenizer(
            full_text, max_length=self.max_seq_len,
            truncation=True, padding="max_length", add_special_tokens=False,
        )

        input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)

        # Labels: mask input tokens with -100 (only train on output)
        labels = input_ids.clone()
        input_len = len(input_enc["input_ids"])
        labels[:input_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Training
# ============================================================

def save_checkpoint(model, tokenizer, en_to_indic, valid_labels, config, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    with open(os.path.join(output_dir, "en_to_indic.json"), "w", encoding="utf-8") as f:
        json.dump(en_to_indic, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "valid_labels.json"), "w", encoding="utf-8") as f:
        json.dump(valid_labels, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model_name,
            "max_input_len": config.max_input_len,
            "max_output_len": config.max_output_len,
            "max_seq_len": config.max_seq_len,
            "epoch": epoch,
        }, f, indent=2)
    print(f"  [Checkpoint saved: epoch {epoch}]")


def train(config, output_dir, root_dir):
    set_seed(config.seed)
    train_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    en_sft_dir = os.path.join(root_dir, "en_sft_dataset")
    sft_dir = os.path.join(root_dir, "sft_dataset")
    en_train_file = os.path.join(en_sft_dir, "train.jsonl")

    if not os.path.exists(en_train_file):
        print(f"ERROR: {en_train_file} not found"); sys.exit(1)

    # Labels
    indic_to_en, en_to_indic = load_label_maps(sft_dir)
    valid_labels = build_label_set(en_train_file, sft_dir)
    print(f"Valid labels ({len(valid_labels)}): {valid_labels}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data
    samples = load_training_samples(en_train_file, sft_dir, indic_to_en, config.max_en_samples)
    dataset = GenerativeREDataset(samples, tokenizer, config.max_input_len, config.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = len(dataloader)
    total_opt_steps = (steps_per_epoch // config.gradient_accumulation_steps) * config.num_epochs
    print(f"Samples: {len(dataset)} | Batches/epoch: {steps_per_epoch} | Opt steps: {total_opt_steps}")

    # Model
    print(f"Loading {config.model_name}...")
    model_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=model_dtype)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    model = model.to(device)

    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem:.1f} GB")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    warmup = int(total_opt_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_opt_steps)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # Train
    print(f"\n{'='*50}")
    print(f"TRAINING: {config.num_epochs} epochs, budget {config.max_train_minutes}m")
    print(f"{'='*50}")

    model.train()
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            elapsed = (time.time() - train_start) / 60
            if elapsed > config.max_train_minutes:
                print(f"\n  TIME LIMIT ({elapsed:.1f}m). Saving.")
                save_checkpoint(model, tokenizer, en_to_indic, valid_labels, config, output_dir, epoch + 1)
                return

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / config.gradient_accumulation_steps
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % 500 == 0:
                total_elapsed = (time.time() - train_start) / 60
                print(f"  E{epoch+1} | B {batch_idx+1}/{steps_per_epoch} | "
                      f"Loss: {epoch_loss/num_batches:.4f} | Time: {total_elapsed:.1f}m")

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = (time.time() - epoch_start) / 60
        total_time = (time.time() - train_start) / 60
        print(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {avg_loss:.4f} | "
              f"Epoch: {epoch_time:.1f}m | Total: {total_time:.1f}m")

        save_checkpoint(model, tokenizer, en_to_indic, valid_labels, config, output_dir, epoch + 1)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone! Total: {(time.time() - train_start) / 60:.1f}m")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    root_dir = sys.argv[2] if len(sys.argv) > 2 else ".."
    train(Config(), output_dir, root_dir)