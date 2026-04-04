"""
Task 1: Inference — Relation Extraction with Classification Head
================================================================
- Batched inference for speed (targets < 30 min for 500 samples on V100)
- Converts English predictions to Indic labels for non-English languages
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Classification Head (must match training)
# ============================================================

class REClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels, dropout=0.0):
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
# Utilities
# ============================================================

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


def load_jsonl(filepath):
    entries = []
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


# ============================================================
# Inference
# ============================================================

def infer(lang, test_file, output_dir):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Language:   {lang}")
    print(f"Test file:  {test_file}")
    print(f"Model dir:  {output_dir}")

    # --- Load config & labels ---
    with open(os.path.join(output_dir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(output_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(output_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)

    en_to_indic = {}
    en_to_indic_path = os.path.join(output_dir, "en_to_indic.json")
    if os.path.exists(en_to_indic_path):
        with open(en_to_indic_path, "r", encoding="utf-8") as f:
            en_to_indic = json.load(f)

    lang_map = en_to_indic.get(lang, {})

    num_labels = config["num_labels"]
    hidden_size = config["hidden_size"]
    max_seq_len = config["max_seq_len"]
    model_name = config["model_name"]

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    e1_start_id = tokenizer.convert_tokens_to_ids("[E1_START]")
    e2_start_id = tokenizer.convert_tokens_to_ids("[E2_START]")

    # --- Model ---
    print(f"Loading model...")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=amp_dtype,
        device_map=None,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base_model, os.path.join(output_dir, "lora_adapter"))

    model = REClassificationModel(
        base_model=peft_model,
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout=0.0,
    )
    classifier_state = torch.load(
        os.path.join(output_dir, "classifier_head.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.classifier.load_state_dict(classifier_state)
    model = model.to(device)
    model.eval()

    load_time = (time.time() - start_time) / 60
    print(f"Model loaded in {load_time:.1f} min")

    # --- Read test data ---
    test_data = load_jsonl(test_file)
    print(f"Test samples: {len(test_data)}")

    # --- Flatten all (entry_idx, rm_idx, marked_sent) for batched inference ---
    all_items = []  # (entry_idx, rm_idx, marked_sent, em1, em2)
    for entry_idx, entry in enumerate(test_data):
        sent = entry.get("sentText", "")
        for rm_idx, rm in enumerate(entry.get("relationMentions", [])):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            marked_sent = mark_entities(sent, em1, em2)
            all_items.append((entry_idx, rm_idx, marked_sent, em1, em2))

    print(f"Total relation mentions to predict: {len(all_items)}")

    # --- Batched inference ---
    INFER_BATCH_SIZE = 32
    predictions = {}  # (entry_idx, rm_idx) -> pred_label

    use_amp = device.type == "cuda"

    for batch_start in range(0, len(all_items), INFER_BATCH_SIZE):
        batch_items = all_items[batch_start:batch_start + INFER_BATCH_SIZE]

        # Tokenize batch
        texts = [item[2] for item in batch_items]
        encoding = tokenizer(
            texts,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Find entity positions for each item in batch
        e1_positions = []
        e2_positions = []
        for i in range(len(batch_items)):
            ids = input_ids[i]
            mask = attention_mask[i]

            e1_pos = (ids == e1_start_id).nonzero(as_tuple=True)[0]
            e2_pos = (ids == e2_start_id).nonzero(as_tuple=True)[0]
            last_pos = mask.sum().item() - 1

            e1_positions.append(e1_pos[0].item() if len(e1_pos) > 0 else last_pos)
            e2_positions.append(e2_pos[0].item() if len(e2_pos) > 0 else last_pos)

        e1_pos_tensor = torch.tensor(e1_positions, device=device)
        e2_pos_tensor = torch.tensor(e2_positions, device=device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(input_ids, attention_mask, e1_pos_tensor, e2_pos_tensor)
            else:
                logits = model(input_ids, attention_mask, e1_pos_tensor, e2_pos_tensor)

        pred_ids = logits.argmax(dim=-1).cpu().tolist()

        for i, item in enumerate(batch_items):
            entry_idx, rm_idx = item[0], item[1]
            en_label = id2label[str(pred_ids[i])]

            if lang != "en" and lang_map:
                pred_label = lang_map.get(en_label, en_label)
            else:
                pred_label = en_label

            predictions[(entry_idx, rm_idx)] = pred_label

        if (batch_start // INFER_BATCH_SIZE + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            done = batch_start + len(batch_items)
            print(f"  Processed {done}/{len(all_items)} | Time: {elapsed:.1f}m")

    # --- Write output ---
    output_file = os.path.join(output_dir, f"output_{lang}.jsonl")

    with open(output_file, "w", encoding="utf-8") as fout:
        for entry_idx, entry in enumerate(test_data):
            sent = entry.get("sentText", "")
            article_id = entry.get("articleId", "")
            sent_id = entry.get("sentId", "")
            relation_mentions = entry.get("relationMentions", [])

            predicted_relations = []
            for rm_idx, rm in enumerate(relation_mentions):
                em1 = rm.get("em1Text", "")
                em2 = rm.get("em2Text", "")
                pred_label = predictions.get((entry_idx, rm_idx), "NA")
                predicted_relations.append({
                    "em1Text": em1,
                    "em2Text": em2,
                    "label": pred_label,
                })

            output_entry = {
                "articleId": article_id,
                "sentId": sent_id,
                "sentText": sent,
                "relationMentions": predicted_relations,
            }
            fout.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    total_time = (time.time() - start_time) / 60
    print(f"\nPredictions saved to {output_file}")
    print(f"Total inference time: {total_time:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python infer_classifier.py <lang_code> <test_file> <output_dir>")
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2], sys.argv[3])