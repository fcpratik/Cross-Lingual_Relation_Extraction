"""
Task 1: Inference — Relation Extraction with Classification Head
================================================================
Uses AutoModel (no lm_head) — matches training script.
Batched inference for speed.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


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
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        e1_hidden = hidden_states[batch_idx, e1_position]
        e2_hidden = hidden_states[batch_idx, e2_position]
        pooled = torch.cat([e1_hidden, e2_hidden], dim=-1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


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


def infer(lang, test_file, output_dir):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Lang: {lang} | Test: {test_file} | Model: {output_dir}")

    # Load config & labels
    with open(os.path.join(output_dir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(output_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(output_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)

    en_to_indic = {}
    path = os.path.join(output_dir, "en_to_indic.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            en_to_indic = json.load(f)
    lang_map = en_to_indic.get(lang, {})

    num_labels = config["num_labels"]
    hidden_size = config["hidden_size"]
    max_seq_len = config["max_seq_len"]
    model_name = config["model_name"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    e1_start_id = tokenizer.convert_tokens_to_ids("[E1_START]")
    e2_start_id = tokenizer.convert_tokens_to_ids("[E2_START]")

    # Model — AutoModel (no lm_head)
    print("Loading model...")
    model_dtype = torch.float32
    if device.type == "cuda":
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = AutoModel.from_pretrained(model_name, torch_dtype=model_dtype)
    base_model.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base_model, os.path.join(output_dir, "lora_adapter"))

    model = REClassificationModel(
        base_model=peft_model, hidden_size=hidden_size,
        num_labels=num_labels, dropout=0.0,
    )
    model.classifier.load_state_dict(torch.load(
        os.path.join(output_dir, "classifier_head.pt"),
        map_location="cpu", weights_only=True,
    ))
    model = model.to(device)
    model.eval()

    load_time = (time.time() - start_time) / 60
    print(f"Model loaded in {load_time:.1f}m")

    # Test data
    test_data = load_jsonl(test_file)
    print(f"Test samples: {len(test_data)}")

    # Flatten all relation mentions
    all_items = []
    for eidx, entry in enumerate(test_data):
        sent = entry.get("sentText", "")
        for ridx, rm in enumerate(entry.get("relationMentions", [])):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            all_items.append((eidx, ridx, mark_entities(sent, em1, em2), em1, em2))

    print(f"Total predictions: {len(all_items)}")

    # Batched inference
    BATCH = 32
    predictions = {}
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for bs in range(0, len(all_items), BATCH):
        batch = all_items[bs:bs + BATCH]
        texts = [it[2] for it in batch]
        enc = tokenizer(texts, max_length=max_seq_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        e1_pos, e2_pos = [], []
        for i in range(len(batch)):
            ids = input_ids[i]
            mask = attention_mask[i]
            last = mask.sum().item() - 1
            e1 = (ids == e1_start_id).nonzero(as_tuple=True)[0]
            e2 = (ids == e2_start_id).nonzero(as_tuple=True)[0]
            e1_pos.append(e1[0].item() if len(e1) > 0 else last)
            e2_pos.append(e2[0].item() if len(e2) > 0 else last)

        e1_t = torch.tensor(e1_pos, device=device)
        e2_t = torch.tensor(e2_pos, device=device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(input_ids, attention_mask, e1_t, e2_t)
            else:
                logits = model(input_ids, attention_mask, e1_t, e2_t)

        preds = logits.argmax(dim=-1).cpu().tolist()
        for i, item in enumerate(batch):
            en_label = id2label[str(preds[i])]
            pred_label = lang_map.get(en_label, en_label) if (lang != "en" and lang_map) else en_label
            predictions[(item[0], item[1])] = pred_label

        if ((bs // BATCH) + 1) % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"  {bs + len(batch)}/{len(all_items)} | {elapsed:.1f}m")

    # Write output
    output_file = os.path.join(output_dir, f"output_{lang}.jsonl")
    with open(output_file, "w", encoding="utf-8") as fout:
        for eidx, entry in enumerate(test_data):
            out = {
                "articleId": entry.get("articleId", ""),
                "sentId": entry.get("sentId", ""),
                "sentText": entry.get("sentText", ""),
                "relationMentions": [
                    {"em1Text": rm.get("em1Text", ""),
                     "em2Text": rm.get("em2Text", ""),
                     "label": predictions.get((eidx, ridx), "NA")}
                    for ridx, rm in enumerate(entry.get("relationMentions", []))
                ],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    total = (time.time() - start_time) / 60
    print(f"\nSaved: {output_file} | Time: {total:.1f}m")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python infer_classifier.py <lang> <test_file> <output_dir>")
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2], sys.argv[3])