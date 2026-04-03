"""
Task 1: Relation Extraction with a Classification Head
=======================================================
- Base model: Qwen/Qwen2.5-1.5B (frozen)
- LoRA adapters on attention layers
- Classification head on pooled hidden states (entity boundary tokens)
- Entity markers: [E1_START] entity1 [E1_END] ... [E2_START] entity2 [E2_END]

Directory layout expected:
    Cross-Lingual_Relation_Extraction/
    ├── en_sft_dataset/
    │   ├── train.jsonl
    │   └── valid.jsonl
    ├── sft_dataset/
    │   ├── hi_map.json, hi_train.jsonl
    │   ├── kn_map.json, kn_train.jsonl
    │   ├── or_map.json, or_train.jsonl
    │   └── tcy_map.json, tcy_val.jsonl
    └── Q1/
        ├── train_classifier.py  (this file)
        └── ...
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
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

    # Training
    batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate_lora = 2e-4
    learning_rate_head = 1e-3
    num_epochs = 5
    max_seq_len = 256
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0

    seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Label Mapping Utilities
# ============================================================

def load_label_maps(sft_dataset_dir):
    """
    Load all *_map.json files.
    Returns:
        indic_to_en: {indic_label -> english_label}
        en_to_indic: {lang_code: {en_label -> indic_label}}
    """
    indic_to_en = {}
    en_to_indic = {}

    if not os.path.isdir(sft_dataset_dir):
        return indic_to_en, en_to_indic

    for fname in os.listdir(sft_dataset_dir):
        if fname.endswith("_map.json"):
            lang_code = fname.replace("_map.json", "")
            fpath = os.path.join(sft_dataset_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                mapping = json.load(f)  # {en_label: indic_label}

            en_to_indic[lang_code] = mapping
            for en_label, indic_label in mapping.items():
                indic_to_en[indic_label] = en_label

    return indic_to_en, en_to_indic


def build_english_label_set(en_train_file, sft_dataset_dir):
    """Build canonical English label ontology."""
    labels = set()

    # From English training data
    if os.path.exists(en_train_file):
        for entry in load_jsonl(en_train_file):
            for rm in entry.get("relationMentions", []):
                lbl = rm.get("label", "")
                if lbl:
                    labels.add(lbl)

    # From map files (English keys)
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
# 3. Data Loading (handles both JSONL and pretty-printed JSON)
# ============================================================

def load_jsonl(filepath):
    """Load a JSONL file, handling both single-line and pretty-printed JSON."""
    entries = []
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found, skipping.")
        return entries

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return entries

    # Try standard JSONL first (one JSON object per line)
    try:
        for line in content.split("\n"):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        print(f"  Loaded {len(entries)} entries from {filepath}")
        return entries
    except json.JSONDecodeError:
        pass

    # Fallback: pretty-printed JSON — accumulate braces
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
                entry = json.loads(buffer.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                pass
            buffer = ""

    print(f"  Loaded {len(entries)} entries from {filepath} (pretty-printed)")
    return entries


def load_training_data(en_train_file, sft_dataset_dir, label2id, indic_to_en):
    """Load all training samples, converting Indic labels to English."""
    samples = []

    # --- English data ---
    print("Loading English data...")
    en_entries = load_jsonl(en_train_file)
    for entry in en_entries:
        sent = entry.get("sentText", "")
        for rm in entry.get("relationMentions", []):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            label = rm.get("label", "NA")
            if label not in label2id:
                label = "NA"
            samples.append({"sent": sent, "em1": em1, "em2": em2, "label": label})

    en_count = len(samples)
    print(f"  English samples: {en_count}")

    # --- Indic language data (Task 1: hi, kn) ---
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
            print(f"  No training file found for {lang}, skipping.")
            continue

        print(f"Loading {lang} data from {lang_file}...")
        lang_entries = load_jsonl(lang_file)
        for entry in lang_entries:
            sent = entry.get("sentText", "")
            for rm in entry.get("relationMentions", []):
                em1 = rm.get("em1Text", "")
                em2 = rm.get("em2Text", "")
                indic_label = rm.get("label", "NA")

                # Convert Indic label -> English
                en_label = indic_to_en.get(indic_label, indic_label)
                if en_label not in label2id:
                    en_label = "NA"

                indic_samples.append({"sent": sent, "em1": em1, "em2": em2, "label": en_label})

    print(f"  Indic samples (raw): {len(indic_samples)}")

    # Oversample Indic data
    if indic_samples and en_count > 0:
        oversample_factor = max(1, en_count // (5 * max(len(indic_samples), 1)))
        oversample_factor = min(oversample_factor, 20)
        oversampled = indic_samples * oversample_factor
        print(f"  Indic oversampled {oversample_factor}x: {len(oversampled)}")
        samples.extend(oversampled)
    else:
        samples.extend(indic_samples)

    print(f"  Total training samples: {len(samples)}")
    return samples


# ============================================================
# 4. Dataset
# ============================================================

class REDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, max_seq_len=256):
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
        self.base_model = base_model
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
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)

        e1_hidden = hidden_states[batch_idx, e1_position]
        e2_hidden = hidden_states[batch_idx, e2_position]

        pooled = torch.cat([e1_hidden, e2_hidden], dim=-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 6. Training Loop
# ============================================================

def train(config, output_dir, root_dir):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Resolve data paths ---
    en_sft_dir = os.path.join(root_dir, "en_sft_dataset")
    sft_dir = os.path.join(root_dir, "sft_dataset")
    en_train_file = os.path.join(en_sft_dir, "train.jsonl")

    if not os.path.exists(en_train_file):
        print(f"ERROR: English training file not found at {en_train_file}")
        print(f"  Root dir contents: {os.listdir(root_dir) if os.path.isdir(root_dir) else 'NOT FOUND'}")
        sys.exit(1)

    if not os.path.isdir(sft_dir):
        print(f"ERROR: sft_dataset dir not found at {sft_dir}")
        sys.exit(1)

    print(f"English train: {en_train_file}")
    print(f"SFT dataset:   {sft_dir}")
    print(f"SFT contents:  {os.listdir(sft_dir)}")

    # --- Load label maps ---
    print("\n--- Label Mappings ---")
    indic_to_en, en_to_indic = load_label_maps(sft_dir)
    print(f"Languages with maps: {list(en_to_indic.keys())}")

    # --- Build label ontology ---
    print("\n--- Label Ontology ---")
    label2id, id2label = build_english_label_set(en_train_file, sft_dir)
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {list(label2id.keys())}")

    # --- Tokenizer ---
    print(f"\n--- Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens = ["[E1_START]", "[E1_END]", "[E2_START]", "[E2_END]"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added} special tokens")

    # --- Load data ---
    print("\n--- Loading Data ---")
    samples = load_training_data(en_train_file, sft_dir, label2id, indic_to_en)

    if len(samples) == 0:
        print("ERROR: No training samples found!")
        sys.exit(1)

    dataset = REDataset(samples, tokenizer, label2id, config.max_seq_len)

    class_weight_tensor = torch.ones(num_labels)
    for idx, w in dataset.class_weights.items():
        class_weight_tensor[idx] = w
    class_weight_tensor = class_weight_tensor.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # --- Model ---
    print(f"\n--- Model ---")
    print(f"Loading {config.model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
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

    if device.type == "cuda":
        model.base_model.half()
    model.classifier.float()

    # --- Optimizer ---
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.base_model.named_parameters() if p.requires_grad],
            "lr": config.learning_rate_lora,
            "weight_decay": config.weight_decay,
        },
        {
            "params": model.classifier.parameters(),
            "lr": config.learning_rate_head,
            "weight_decay": config.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params)

    total_steps = (len(dataloader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Training ---
    print(f"\n--- Training ---")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batches/epoch: {len(dataloader)}")
    print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}")

    model.train()

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            e1_position = batch["e1_position"].to(device)
            e2_position = batch["e2_position"].to(device)
            labels = batch["label"].to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids, attention_mask, e1_position, e2_position)
                    logits = logits.float()
                    loss = F.cross_entropy(logits, labels, weight=class_weight_tensor)
                    loss = loss / config.gradient_accumulation_steps
            else:
                logits = model(input_ids, attention_mask, e1_position, e2_position)
                loss = F.cross_entropy(logits, labels, weight=class_weight_tensor)
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {epoch_loss/num_batches:.4f} | Acc: {correct/total:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    # --- Save ---
    print(f"\n--- Saving to {output_dir} ---")
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
            "num_labels": num_labels,
            "hidden_size": hidden_size,
        }, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    # root_dir = parent of Q1/, should contain en_sft_dataset/ and sft_dataset/
    root_dir = sys.argv[2] if len(sys.argv) > 2 else ".."

    config = Config()
    train(config, output_dir, root_dir)