"""
Task 2: Inference — Autoregressive Generation for RE
=====================================================
- Loads LoRA adapter on Qwen2.5-1.5B
- Generates relation labels as text
- Post-processes to enforce valid ontology labels
- Converts to Indic labels for non-English languages
"""

import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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


def format_input(sent, em1, em2):
    return (
        f"Extract the relation between the two entities in the sentence.\n"
        f"Sentence: {sent}\n"
        f"Entity 1: {em1}\n"
        f"Entity 2: {em2}\n"
        f"Relation:"
    )


def find_closest_label(generated, valid_labels):
    """
    Post-process: match generated text to closest valid ontology label.
    Uses exact match first, then substring match, then defaults to NA.
    """
    generated = generated.strip()

    # Exact match
    if generated in valid_labels:
        return generated

    # Case-insensitive exact match
    gen_lower = generated.lower()
    for lbl in valid_labels:
        if lbl.lower() == gen_lower:
            return lbl

    # Substring match (generated contains a valid label)
    for lbl in valid_labels:
        if lbl in generated:
            return lbl

    # Substring match (valid label contains generated)
    for lbl in valid_labels:
        if generated in lbl and len(generated) > 3:
            return lbl

    # Partial path match (match last component)
    gen_parts = generated.strip("/").split("/")
    best_match = None
    best_score = 0
    for lbl in valid_labels:
        lbl_parts = lbl.strip("/").split("/")
        # Count matching parts from the end
        score = 0
        for gp, lp in zip(reversed(gen_parts), reversed(lbl_parts)):
            if gp.lower() == lp.lower():
                score += 1
            else:
                break
        if score > best_score:
            best_score = score
            best_match = lbl

    if best_match and best_score > 0:
        return best_match

    return "NA"


def infer(lang, test_file, output_dir):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Lang: {lang} | Test: {test_file}")

    # Load config
    with open(os.path.join(output_dir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(output_dir, "valid_labels.json"), "r", encoding="utf-8") as f:
        valid_labels = json.load(f)

    en_to_indic = {}
    path = os.path.join(output_dir, "en_to_indic.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            en_to_indic = json.load(f)
    lang_map = en_to_indic.get(lang, {})

    model_name = config["model_name"]
    max_output_len = config.get("max_output_len", 40)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # for batch generation

    # Model
    print("Loading model...")
    model_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype)
    model = PeftModel.from_pretrained(base_model, os.path.join(output_dir, "lora_adapter"))
    model = model.to(device)
    model.eval()

    print(f"Model loaded in {(time.time() - start_time) / 60:.1f}m")

    # Test data
    test_data = load_jsonl(test_file)
    print(f"Test samples: {len(test_data)}")

    # Flatten
    all_items = []
    for eidx, entry in enumerate(test_data):
        sent = entry.get("sentText", "")
        for ridx, rm in enumerate(entry.get("relationMentions", [])):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            prompt = format_input(sent, em1, em2)
            all_items.append((eidx, ridx, prompt))

    print(f"Total predictions: {len(all_items)}")

    # Batched generation
    BATCH = 16
    predictions = {}
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for bs in range(0, len(all_items), BATCH):
        batch = all_items[bs:bs + BATCH]
        prompts = [it[2] for it in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=config.get("max_input_len", 220),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_output_len,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_output_len,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        for i, item in enumerate(batch):
            generated_ids = outputs[i][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Post-process to valid label
            en_label = find_closest_label(generated_text, valid_labels)

            # Convert to target language
            if lang != "en" and lang_map:
                pred_label = lang_map.get(en_label, en_label)
            else:
                pred_label = en_label

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
        print("Usage: python infer_generative.py <lang> <test_file> <output_dir>")
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2], sys.argv[3])