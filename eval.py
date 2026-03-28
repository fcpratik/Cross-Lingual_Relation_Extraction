import sys
import json
import re
from sklearn.metrics import f1_score

def normalize_entities(data):
    """
    Normalizes both raw dataset format ('relationMentions') and 
    processed generative format ('entities') into a standard list of dicts.
    """
    if "entities" in data:
        return data["entities"]
    elif "relationMentions" in data:
        return [
            {
                "em1": r.get("em1Text", ""), 
                "em2": r.get("em2Text", ""), 
                "relation": r.get("label", "")
            } 
            for r in data["relationMentions"]
        ]
    return []

def evaluate_files(pred_file, ref_file):
    all_true = []
    all_pred = []

    try:
        with open(pred_file, 'r', encoding='utf-8') as f_pred, \
             open(ref_file, 'r', encoding='utf-8') as f_ref:
            
            for line_idx, (pred_line, ref_line) in enumerate(zip(f_pred, f_ref), 1):
                pred_line = pred_line.strip()
                ref_line = ref_line.strip()
                
                if not ref_line:
                    continue

                # --- 1. Parse Reference (Ground Truth) ---
                try:
                    ref_data = json.loads(ref_line)
                    true_rels = normalize_entities(ref_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in reference file at line {line_idx}")
                    true_rels = []

                # --- 2. Parse Predictions ---
                predicted_rels = []
                if pred_line:
                    try:
                        # Try direct JSON parsing first
                        pred_data = json.loads(pred_line)
                        predicted_rels = normalize_entities(pred_data)
                    except json.JSONDecodeError:
                        # Fallback: regex search if the model output contains extra text
                        match = re.search(r'\{.*\}', pred_line.replace('\n', ''))
                        if match:
                            try:
                                pred_data = json.loads(match.group(0))
                                predicted_rels = normalize_entities(pred_data)
                            except json.JSONDecodeError:
                                pass # Failed regex extraction

                # --- 3. Align and Score ---
                # Map predictions by (em1, em2) for easy lookup
                pred_dict = {
                    (p.get('em1', ''), p.get('em2', '')): p.get('relation', '') 
                    for p in predicted_rels
                }

                # Iterate ONLY through the ground truth to ignore extra hallucinated pairs
                for t in true_rels:
                    em1 = t.get('em1', '')
                    em2 = t.get('em2', '')
                    true_rel = t.get('relation', '')
                    
                    all_true.append(true_rel)
                    
                    if (em1, em2) in pred_dict:
                        all_pred.append(pred_dict[(em1, em2)])
                    else:
                        all_pred.append("None")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- 4. Calculate F1 Scores ---
    labels = list(set(all_true + all_pred))
    if "None" in labels: 
        labels.remove("None")

    if not labels:
        print("No valid relations found to evaluate.")
        sys.exit(0)

    macro_f1 = f1_score(all_true, all_pred, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(all_true, all_pred, labels=labels, average='micro', zero_division=0)

    print(f"Evaluation Results:")
    print(f"-------------------")
    print(f"Total True Relations: {len(all_true)}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <predictions_jsonl> <reference_jsonl>")
        sys.exit(1)

    predictions_path = sys.argv[1]
    reference_path = sys.argv[2]
    
    evaluate_files(predictions_path, reference_path)