"""
Task 1: Relation Extraction with a Classification Head
=======================================================
- Base model: Qwen/Qwen2.5-1.5B (frozen, NO lm_head — uses AutoModel)
- LoRA adapters on attention layers
- Classification head on pooled entity marker hidden states
- Gradient checkpointing to save memory
- Saves checkpoint after every epoch
- Time-budgeted: targets < 150 min on V100
"""

import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter

# ============================================================
# 1. Configuration
# ============================================================

class Config:
    model_name = "Qwen/Qwen2.5-1.5B"

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Training — tuned for T4/V100 16GB
    batch_size = 8
    gradient_accumulation_steps = 4  # effective batch = 32
    learning_rate_lora = 2e-4
    learning_rate_head = 1e-3
    num_epochs = 3
    max_seq_len = 180
    warmup_ratio = 0.06
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Data budget
    max_en_samples = 60000

    # Time budget
    max_train_minutes = 140

    seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Label Mapping
# ============================================================

def load_label_maps(sft_dataset_dir):
    indic_to_en = {}
    en_to_indic = {}
    if not os.path.isdir(sft_dataset_dir):
        return indic_to_en, en_to_indic
    for fname in os.listdir(sft_dataset_dir):
        if fname.endswith("_map.json"):
            lang_code = fname.replace("_map.json", "")
            with open(os.path.join(sft_dataset_dir, fname), "r", encoding="utf-8") as f:
                mapping = json.load(f)
            en_to_indic[lang_code] = mapping
            for en_label, indic_label in mapping.items():
                indic_to_en[indic_label] = en_label
    return indic_to_en, en_to_indic


def build_english_label_set(en_train_file, sft_dataset_dir):
    labels = set()
    if os.path.exists(en_train_file):
        for entry in load_jsonl(en_train_file):
            for rm in entry.get("relationMentions", []):
                lbl = rm.get("label", "")
                if lbl:
                    labels.add(lbl)
    if os.path.isdir(sft_dataset_dir):
        for fname in os.listdir(sft_dataset_dir):
            if fname.endswith("_map.json"):
                with open(os.path.join(sft_dataset_dir, fname), "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                labels.update(mapping.keys())
    labels.discard("NA")
    labels.discard("")
    sorted_labels = ["NA"] + sorted(labels)
    label2id = {l: i for i, l in enumerate(sorted_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# ============================================================
# 3. Data Loading
# ============================================================

def load_jsonl(filepath):
    entries = []
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found.")
        return entries
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return entries
    # Try standard JSONL
    try:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        print(f"  Loaded {len(entries)} entries from {filepath}")
        return entries
    except json.JSONDecodeError:
        pass
    # Fallback: pretty-printed
    entries = []
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
    print(f"  Loaded {len(entries)} entries from {filepath} (pretty-printed)")
    return entries


def load_training_data(en_train_file, sft_dataset_dir, label2id, indic_to_en, max_en_samples):
    samples = []

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
            if label not in label2id:
                label = "NA"
            en_samples.append({"sent": sent, "em1": em1, "em2": em2, "label": label})

    if len(en_samples) > max_en_samples:
        print(f"  Subsampling English: {len(en_samples)} -> {max_en_samples}")
        random.shuffle(en_samples)
        by_label = {}
        for s in en_samples:
            by_label.setdefault(s["label"], []).append(s)
        per_label = max(10, max_en_samples // len(by_label))
        subsampled = []
        for lbl, items in by_label.items():
            subsampled.extend(items[:per_label])
        remaining = max_en_samples - len(subsampled)
        if remaining > 0:
            used = set(id(s) for s in subsampled)
            rest = [s for s in en_samples if id(s) not in used]
            random.shuffle(rest)
            subsampled.extend(rest[:remaining])
        en_samples = subsampled[:max_en_samples]

    samples.extend(en_samples)
    en_count = len(samples)
    print(f"  English samples: {en_count}")

    # Indic (hi, kn)
    indic_samples = []
    for lang in ["hi", "kn"]:
        candidates = [
            os.path.join(sft_dataset_dir, f"{lang}_train.jsonl"),
            os.path.join(sft_dataset_dir, f"train_{lang}.jsonl"),
            os.path.join(sft_dataset_dir, f"{lang}_val.jsonl"),
        ]
        lang_file = None
        for c in candidates:
            if os.path.exists(c):
                lang_file = c
                break
        if lang_file is None:
            continue
        print(f"Loading {lang} data from {lang_file}...")
        for entry in load_jsonl(lang_file):
            sent = entry.get("sentText", "")
            for rm in entry.get("relationMentions", []):
                em1 = rm.get("em1Text", "")
                em2 = rm.get("em2Text", "")
                indic_label = rm.get("label", "NA")
                en_label = indic_to_en.get(indic_label, indic_label)
                if en_label not in label2id:
                    en_label = "NA"
                indic_samples.append({"sent": sent, "em1": em1, "em2": em2, "label": en_label})

    print(f"  Indic samples (raw): {len(indic_samples)}")
    if indic_samples and en_count > 0:
        factor = min(20, max(1, en_count // (5 * max(len(indic_samples), 1))))
        oversampled = indic_samples * factor
        print(f"  Indic oversampled {factor}x: {len(oversampled)}")
        samples.extend(oversampled)
    else:
        samples.extend(indic_samples)

    random.shuffle(samples)
    print(f"  Total training samples: {len(samples)}")
    return samples


# ============================================================
# 4. Dataset
# ============================================================

class REDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, max_seq_len=180):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len
        self.samples = samples
        label_counts = Counter(s["label"] for s in self.samples)
        total = len(self.samples)
        self.class_weights = {}
        for lbl, count in label_counts.items():
            self.class_weights[self.label2id[lbl]] = min(total / (len(label_counts) * count), 10.0)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def mark_entities(sent, em1, em2):
        entities = [(em1, "[E1_START]", "[E1_END]"), (em2, "[E2_START]", "[E2_END]")]
        entities.sort(key=lambda x: len(x[0]), reverse=True)
        marked = sent
        for ent_text, start_marker, end_marker in entities:
            if ent_text in marked:
                marked = marked.replace(ent_text, f"{start_marker} {ent_text} {end_marker}", 1)
            else:
                marked = f"{marked} {start_marker} {ent_text} {end_marker}"
        return marked

    def __getitem__(self, idx):
        sample = self.samples[idx]
        marked_sent = self.mark_entities(sample["sent"], sample["em1"], sample["em2"])
        label_id = self.label2id[sample["label"]]
        encoding = self.tokenizer(
            marked_sent,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        e1_start_id = self.tokenizer.convert_tokens_to_ids("[E1_START]")
        e2_start_id = self.tokenizer.convert_tokens_to_ids("[E2_START]")
        e1_pos = (input_ids == e1_start_id).nonzero(as_tuple=True)[0]
        e2_pos = (input_ids == e2_start_id).nonzero(as_tuple=True)[0]
        last_pos = attention_mask.sum().item() - 1
        e1_position = e1_pos[0].item() if len(e1_pos) > 0 else last_pos
        e2_position = e2_pos[0].item() if len(e2_pos) > 0 else last_pos
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_position": torch.tensor(e1_position, dtype=torch.long),
            "e2_position": torch.tensor(e2_position, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


# ============================================================
# 5. Classification Head Model
# ============================================================

class REClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.base_model = base_model  # This is the PEFT-wrapped AutoModel (no lm_head)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, e1_position, e2_position):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # AutoModel returns last_hidden_state directly (no lm_head computation!)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        e1_hidden = hidden_states[batch_idx, e1_position]
        e2_hidden = hidden_states[batch_idx, e2_position]
        pooled = torch.cat([e1_hidden, e2_hidden], dim=-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 6. Save Checkpoint
# ============================================================

def save_checkpoint(model, peft_model, tokenizer, label2id, id2label,
                    en_to_indic, config, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    torch.save(model.classifier.state_dict(), os.path.join(output_dir, "classifier_head.pt"))
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "en_to_indic.json"), "w", encoding="utf-8") as f:
        json.dump(en_to_indic, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": config.lora_target_modules,
            "max_seq_len": config.max_seq_len,
            "num_labels": len(label2id),
            "hidden_size": peft_model.config.hidden_size,
            "epoch": epoch,
        }, f, indent=2)
    print(f"  [Checkpoint saved: epoch {epoch} -> {output_dir}]")


# ============================================================
# 7. Training Loop
# ============================================================

def train(config, output_dir, root_dir):
    set_seed(config.seed)
    train_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data paths ---
    en_sft_dir = os.path.join(root_dir, "en_sft_dataset")
    sft_dir = os.path.join(root_dir, "sft_dataset")
    en_train_file = os.path.join(en_sft_dir, "train.jsonl")
    if not os.path.exists(en_train_file):
        print(f"ERROR: {en_train_file} not found"); sys.exit(1)
    if not os.path.isdir(sft_dir):
        print(f"ERROR: {sft_dir} not found"); sys.exit(1)
    print(f"English train: {en_train_file}")
    print(f"SFT dataset:   {sft_dir}")
    print(f"SFT contents:  {os.listdir(sft_dir)}")

    # --- Labels ---
    print("\n--- Label Mappings ---")
    indic_to_en, en_to_indic = load_label_maps(sft_dir)
    print(f"Languages with maps: {list(en_to_indic.keys())}")

    print("\n--- Label Ontology ---")
    label2id, id2label = build_english_label_set(en_train_file, sft_dir)
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {list(label2id.keys())}")

    # --- Tokenizer ---
    print(f"\n--- Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens = ["[E1_START]", "[E1_END]", "[E2_START]", "[E2_END]"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added} special tokens")

    # --- Data ---
    print("\n--- Loading Data ---")
    samples = load_training_data(en_train_file, sft_dir, label2id, indic_to_en, config.max_en_samples)
    if not samples:
        print("ERROR: No training samples!"); sys.exit(1)

    dataset = REDataset(samples, tokenizer, label2id, config.max_seq_len)
    class_weight_tensor = torch.ones(num_labels)
    for idx, w in dataset.class_weights.items():
        class_weight_tensor[idx] = w
    class_weight_tensor = class_weight_tensor.to(device)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    steps_per_epoch = len(dataloader)
    total_opt_steps = (steps_per_epoch // config.gradient_accumulation_steps) * config.num_epochs
    print(f"\nSamples: {len(dataset)} | Batches/epoch: {steps_per_epoch} | "
          f"Opt steps: {total_opt_steps} | Budget: {config.max_train_minutes}m")

    # --- Model (AutoModel — NO lm_head, saves ~900MB VRAM) ---
    print(f"\n--- Model (AutoModel, no lm_head) ---")
    print(f"Loading {config.model_name}...")

    model_dtype = torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16

    base_model = AutoModel.from_pretrained(
        config.model_name,
        torch_dtype=model_dtype,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to save memory
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # Not CAUSAL_LM since we use AutoModel
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    hidden_size = peft_model.config.hidden_size

    model = REClassificationModel(
        base_model=peft_model,
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout=config.lora_dropout,
    )
    model = model.to(device)
    model.classifier.float()

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory after model load: {mem_alloc:.1f} / {mem_total:.1f} GB")

    # --- Optimizer ---
    optimizer_grouped_params = [
        {"params": [p for n, p in model.base_model.named_parameters() if p.requires_grad],
         "lr": config.learning_rate_lora, "weight_decay": config.weight_decay},
        {"params": model.classifier.parameters(),
         "lr": config.learning_rate_head, "weight_decay": config.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params)
    warmup_steps = int(total_opt_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # --- Training ---
    print(f"\n{'='*50}")
    print(f"TRAINING: {config.num_epochs} epochs, budget {config.max_train_minutes}m")
    print(f"{'='*50}")

    model.train()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Time check
            elapsed_min = (time.time() - train_start) / 60
            if elapsed_min > config.max_train_minutes:
                print(f"\n  TIME LIMIT ({elapsed_min:.1f}m). Saving and stopping.")
                save_checkpoint(model, peft_model, tokenizer, label2id, id2label,
                                en_to_indic, config, output_dir, epoch + 1)
                print(f"Total: {elapsed_min:.1f}m")
                return

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            e1_position = batch["e1_position"].to(device, non_blocking=True)
            e2_position = batch["e2_position"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(input_ids, attention_mask, e1_position, e2_position)
                    logits_fp32 = logits.float()
                    loss = F.cross_entropy(logits_fp32, labels, weight=class_weight_tensor)
                    loss = loss / config.gradient_accumulation_steps
            else:
                logits = model(input_ids, attention_mask, e1_position, e2_position)
                logits_fp32 = logits.float()
                loss = F.cross_entropy(logits_fp32, labels, weight=class_weight_tensor)
                loss = loss / config.gradient_accumulation_steps

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
            preds = logits_fp32.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 500 == 0:
                batch_elapsed = (time.time() - epoch_start) / 60
                total_elapsed = (time.time() - train_start) / 60
                eta = batch_elapsed / (batch_idx + 1) * steps_per_epoch
                print(f"  E{epoch+1} | B {batch_idx+1}/{steps_per_epoch} | "
                      f"Loss: {epoch_loss/num_batches:.4f} | Acc: {correct/total:.4f} | "
                      f"Time: {total_elapsed:.1f}m | ETA: {eta:.1f}m")

        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        epoch_time = (time.time() - epoch_start) / 60
        total_time = (time.time() - train_start) / 60
        print(f"\nEpoch {epoch+1}/{config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
              f"Epoch: {epoch_time:.1f}m | Total: {total_time:.1f}m")

        # Save after every epoch
        save_checkpoint(model, peft_model, tokenizer, label2id, id2label,
                        en_to_indic, config, output_dir, epoch + 1)

        # Clear cache between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = (time.time() - train_start) / 60
    print(f"\nTraining complete! Total: {total_time:.1f}m")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    root_dir = sys.argv[2] if len(sys.argv) > 2 else ".."
    config = Config()
    train(config, output_dir, root_dir)