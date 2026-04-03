"""
Task 1: Inference — Relation Extraction with Classification Head
================================================================
Loads trained LoRA adapter + classification head, produces predictions.
For Indic languages, converts English predicted labels to the target language.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Classification Head (must match training)
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
# Entity Marking
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


# ============================================================
# Load test data (handles both JSONL and pretty-printed)
# ============================================================

def load_jsonl(filepath):
    entries = []
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
        return entries
    except json.JSONDecodeError:
        pass

    # Fallback: pretty-printed
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Language:   {lang}")
    print(f"Test file:  {test_file}")
    print(f"Model dir:  {output_dir}")

    # --- Load config ---
    with open(os.path.join(output_dir, "config.json"), "r") as f:
        config = json.load(f)

    # --- Load label maps ---
    with open(os.path.join(output_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(output_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)

    # --- Load en_to_indic maps for label conversion ---
    en_to_indic = {}
    en_to_indic_path = os.path.join(output_dir, "en_to_indic.json")
    if os.path.exists(en_to_indic_path):
        with open(en_to_indic_path, "r", encoding="utf-8") as f:
            en_to_indic = json.load(f)

    # Build label converter for target language
    lang_map = en_to_indic.get(lang, {})  # {en_label: indic_label}

    num_labels = config["num_labels"]
    hidden_size = config["hidden_size"]
    max_seq_len = config["max_seq_len"]
    model_name = config["model_name"]

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Model ---
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    print("Loading LoRA adapter...")
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

    e1_start_id = tokenizer.convert_tokens_to_ids("[E1_START]")
    e2_start_id = tokenizer.convert_tokens_to_ids("[E2_START]")

    # --- Read test data ---
    test_data = load_jsonl(test_file)
    print(f"Test samples: {len(test_data)}")

    # --- Predict ---
    output_file = os.path.join(output_dir, f"output_{lang}.jsonl")

    with open(output_file, "w", encoding="utf-8") as fout:
        for i, entry in enumerate(test_data):
            sent = entry.get("sentText", "")
            article_id = entry.get("articleId", "")
            sent_id = entry.get("sentId", "")
            relation_mentions = entry.get("relationMentions", [])

            predicted_relations = []

            for rm in relation_mentions:
                em1 = rm.get("em1Text", "")
                em2 = rm.get("em2Text", "")

                marked_sent = mark_entities(sent, em1, em2)

                encoding = tokenizer(
                    marked_sent,
                    max_length=max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                e1_pos = (input_ids[0] == e1_start_id).nonzero(as_tuple=True)[0]
                e2_pos = (input_ids[0] == e2_start_id).nonzero(as_tuple=True)[0]

                last_pos = attention_mask[0].sum().item() - 1
                e1_position = torch.tensor([e1_pos[0].item() if len(e1_pos) > 0 else last_pos], device=device)
                e2_position = torch.tensor([e2_pos[0].item() if len(e2_pos) > 0 else last_pos], device=device)

                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            logits = model(input_ids, attention_mask, e1_position, e2_position)
                    else:
                        logits = model(input_ids, attention_mask, e1_position, e2_position)

                pred_id = logits.argmax(dim=-1).item()
                en_label = id2label[str(pred_id)]

                # Convert to target language label if needed
                if lang != "en" and lang_map:
                    pred_label = lang_map.get(en_label, en_label)
                else:
                    pred_label = en_label

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

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_data)}")

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python infer_classifier.py <lang_code> <test_file> <output_dir>")
        print("  lang_code: en / hi / kn / or / tcy")
        sys.exit(1)

    lang = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = sys.argv[3]

    infer(lang, test_file, output_dir)