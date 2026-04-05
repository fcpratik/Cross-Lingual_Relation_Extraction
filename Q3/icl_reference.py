"""
Task 3: In-Context Learning for Cross-Lingual Relation Extraction
==================================================================
- Model: Meta-Llama-3.1-8B-Instruct (NO gradient updates)
- Uses vLLM for efficient batch inference
- Similarity-based demonstration retrieval using sentence embeddings
- Evaluated on: hi, kn, or, tcy

Strategy:
1. Load all labeled Indic data as demonstration pool
2. For each test instance, retrieve most similar demos via TF-IDF + cosine similarity
3. Construct prompt with demonstrations + test query
4. Parse model output to extract relation label
5. Post-process to enforce valid ontology labels
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

# ============================================================
# Data Loading
# ============================================================

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


def build_valid_labels(en_train_file, sft_dir):
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


# ============================================================
# Demo Pool & Retrieval
# ============================================================

def load_demo_pool(sft_dir, en_train_file, indic_to_en, target_lang):
    """
    Load demonstrations from:
    1. Target language labeled data (highest priority)
    2. Other Indic language data
    3. A small sample of English data
    """
    demos = []  # list of (sentence, em1, em2, en_label, original_label)

    # Target language first
    for lang in [target_lang, "hi", "kn", "or", "tcy"]:
        candidates = [
            os.path.join(sft_dir, f"{lang}_train.jsonl"),
            os.path.join(sft_dir, f"{lang}_val.jsonl"),
        ]
        for c in candidates:
            if not os.path.exists(c):
                continue
            for entry in load_jsonl(c):
                sent = entry.get("sentText", "")
                for rm in entry.get("relationMentions", []):
                    em1 = rm.get("em1Text", "")
                    em2 = rm.get("em2Text", "")
                    label = rm.get("label", "NA")
                    en_label = indic_to_en.get(label, label)
                    demos.append({
                        "sent": sent, "em1": em1, "em2": em2,
                        "en_label": en_label, "orig_label": label,
                        "lang": lang,
                    })

    # Add some English demos for label coverage
    if os.path.exists(en_train_file):
        en_entries = load_jsonl(en_train_file)
        # Sample diverse English examples (stratified by label)
        by_label = defaultdict(list)
        for entry in en_entries:
            sent = entry.get("sentText", "")
            for rm in entry.get("relationMentions", []):
                lbl = rm.get("label", "NA")
                if lbl != "NA":
                    by_label[lbl].append({
                        "sent": sent,
                        "em1": rm.get("em1Text", ""),
                        "em2": rm.get("em2Text", ""),
                        "en_label": lbl,
                        "orig_label": lbl,
                        "lang": "en",
                    })
        # Take 3 per label
        for lbl, items in by_label.items():
            demos.extend(items[:3])

    print(f"  Demo pool size: {len(demos)}")
    return demos


class TFIDFRetriever:
    """Simple TF-IDF based retriever for finding similar demonstrations."""

    def __init__(self, demos):
        self.demos = demos
        self.texts = [f"{d['sent']} {d['em1']} {d['em2']}" for d in demos]
        self._build_index()

    def _build_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Use character n-grams to handle multilingual text
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=50000, sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query_sent, query_em1, query_em2, k=8):
        """Retrieve top-k most similar demonstrations."""
        query = f"{query_sent} {query_em1} {query_em2}"
        query_vec = self.vectorizer.transform([query])

        # Cosine similarity
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.demos[i] for i in top_indices]


def select_stratified_demos(demos, target_lang, k=8):
    """Fallback: select demos stratified by label, preferring target language."""
    by_label = defaultdict(list)
    for d in demos:
        by_label[d["en_label"]].append(d)

    selected = []
    # Prioritize target language demos
    for lbl, items in by_label.items():
        lang_items = [d for d in items if d["lang"] == target_lang]
        if lang_items:
            selected.append(lang_items[0])
        elif items:
            selected.append(items[0])

    # Fill remaining slots
    if len(selected) < k:
        remaining = [d for d in demos if d not in selected]
        selected.extend(remaining[:k - len(selected)])

    return selected[:k]


# ============================================================
# Prompt Construction
# ============================================================

def build_prompt(test_sent, test_em1, test_em2, demos, valid_labels, target_lang, en_to_indic):
    """Build the ICL prompt with demonstrations."""
    lang_map = en_to_indic.get(target_lang, {})

    # System instruction
    label_list = "\n".join(f"  - {l}" for l in valid_labels)
    system = (
        f"You are a relation extraction system. Given a sentence and two entities, "
        f"predict the relation between them from this list:\n{label_list}\n"
        f"If no relation holds, predict: NA\n"
        f"Output ONLY the relation label, nothing else.\n"
    )

    # Demonstrations
    demo_text = ""
    for d in demos:
        demo_text += (
            f"\nSentence: {d['sent']}\n"
            f"Entity 1: {d['em1']}\n"
            f"Entity 2: {d['em2']}\n"
            f"Relation: {d['en_label']}\n"
        )

    # Test query
    query = (
        f"\nSentence: {test_sent}\n"
        f"Entity 1: {test_em1}\n"
        f"Entity 2: {test_em2}\n"
        f"Relation:"
    )

    full_prompt = system + demo_text + query
    return full_prompt


def build_chat_prompt(test_sent, test_em1, test_em2, demos, valid_labels):
    """Build prompt in Llama-3.1-Instruct chat format."""
    label_list = ", ".join(valid_labels)

    system_msg = (
        f"You are a relation extraction system. Given a sentence and two entities, "
        f"predict the relation between them. Valid relations: {label_list}. "
        f"If no relation holds, output NA. Output ONLY the relation label."
    )

    # Build few-shot examples as user/assistant turns
    messages = [{"role": "system", "content": system_msg}]

    for d in demos:
        user_msg = (
            f"Sentence: {d['sent']}\n"
            f"Entity 1: {d['em1']}\n"
            f"Entity 2: {d['em2']}"
        )
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": d["en_label"]})

    # Test query
    test_msg = (
        f"Sentence: {test_sent}\n"
        f"Entity 1: {test_em1}\n"
        f"Entity 2: {test_em2}"
    )
    messages.append({"role": "user", "content": test_msg})

    return messages


# ============================================================
# Post-processing
# ============================================================

def find_closest_label(generated, valid_labels):
    generated = generated.strip().split("\n")[0].strip()

    if generated in valid_labels:
        return generated
    if generated == "NA" or generated.lower() == "na" or generated.lower() == "none":
        return "NA"

    gen_lower = generated.lower()
    for lbl in valid_labels:
        if lbl.lower() == gen_lower:
            return lbl
    for lbl in valid_labels:
        if lbl in generated:
            return lbl
    for lbl in valid_labels:
        if generated in lbl and len(generated) > 3:
            return lbl

    # Partial path match
    gen_parts = generated.strip("/").split("/")
    best_match, best_score = None, 0
    for lbl in valid_labels:
        lbl_parts = lbl.strip("/").split("/")
        score = sum(1 for gp, lp in zip(reversed(gen_parts), reversed(lbl_parts))
                    if gp.lower() == lp.lower())
        if score > best_score:
            best_score = score
            best_match = lbl
    if best_match and best_score > 0:
        return best_match

    return "NA"


# ============================================================
# Main Inference
# ============================================================

def infer(lang, test_file, output_dir):
    start_time = time.time()
    print(f"Lang: {lang} | Test: {test_file} | Output: {output_dir}")

    # Resolve paths — output_dir for Q3 is where we save results
    # Data is in the parent directory structure
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sft_dir = os.path.join(root_dir, "sft_dataset")
    en_sft_dir = os.path.join(root_dir, "en_sft_dataset")
    en_train_file = os.path.join(en_sft_dir, "train.jsonl")

    # Label maps
    indic_to_en, en_to_indic = load_label_maps(sft_dir)
    lang_map = en_to_indic.get(lang, {})
    valid_labels = build_valid_labels(en_train_file, sft_dir)
    valid_labels_with_na = valid_labels + ["NA"]
    print(f"Valid labels: {len(valid_labels)}")

    # Load demo pool
    print("Loading demonstration pool...")
    demos = load_demo_pool(sft_dir, en_train_file, indic_to_en, lang)

    # Build retriever
    print("Building TF-IDF retriever...")
    retriever = TFIDFRetriever(demos) if len(demos) > 0 else None

    # Load test data
    test_data = load_jsonl(test_file)
    print(f"Test samples: {len(test_data)}")

    # Flatten test items
    all_items = []
    for eidx, entry in enumerate(test_data):
        sent = entry.get("sentText", "")
        for ridx, rm in enumerate(entry.get("relationMentions", [])):
            em1 = rm.get("em1Text", "")
            em2 = rm.get("em2Text", "")
            all_items.append((eidx, ridx, sent, em1, em2))

    print(f"Total predictions: {len(all_items)}")

    # Build prompts with retrieved demos
    print("Building prompts...")
    NUM_DEMOS = 8  # number of demonstrations per query
    prompts = []
    for item in all_items:
        eidx, ridx, sent, em1, em2 = item

        # Retrieve similar demonstrations
        if retriever:
            selected_demos = retriever.retrieve(sent, em1, em2, k=NUM_DEMOS)
        else:
            selected_demos = []

        prompt = build_prompt(sent, em1, em2, selected_demos, valid_labels, lang, en_to_indic)
        prompts.append(prompt)

    # --- vLLM Generation ---
    print("Loading vLLM model...")
    from vllm import LLM, SamplingParams

    model_name = "Meta-Llama-3.1-8B-Instruct"
    llm = LLM(model=model_name, max_model_len=4096)
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=30,
        stop=["\n", "\n\n"],
    )

    print(f"Generating {len(prompts)} predictions...")
    outputs = llm.generate(prompts, sampling_params)

    # Extract predictions
    predictions = {}
    for i, (item, output) in enumerate(zip(all_items, outputs)):
        eidx, ridx = item[0], item[1]
        generated = output.outputs[0].text.strip()

        # Post-process
        en_label = find_closest_label(generated, valid_labels_with_na)

        # Convert to target language
        if lang != "en" and lang_map:
            pred_label = lang_map.get(en_label, en_label)
        else:
            pred_label = en_label

        predictions[(eidx, ridx)] = pred_label

    # Write output
    os.makedirs(output_dir, exist_ok=True)
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
        print("Usage: python icl_inference.py <lang> <test_file> <output_dir>")
        sys.exit(1)
    infer(sys.argv[1], sys.argv[2], sys.argv[3])