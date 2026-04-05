"""Task 1: Inference with Classification Head"""
import os, sys, json, time, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


class REModel(nn.Module):
    def __init__(self, base, hs, nl):
        super().__init__();
        self.base = base
        self.clf = nn.Sequential(nn.Linear(hs * 2, hs), nn.GELU(), nn.Dropout(0), nn.Linear(hs, nl))

    def forward(self, ids, mask, e1, e2):
        h = self.base(input_ids=ids, attention_mask=mask).last_hidden_state
        b = torch.arange(h.size(0), device=h.device)
        return self.clf(torch.cat([h[b, e1], h[b, e2]], dim=-1))


def mark(sent, e1, e2):
    ents = [(e1, "[E1S]", "[E1E]"), (e2, "[E2S]", "[E2E]")]
    ents.sort(key=lambda x: len(x[0]), reverse=True)
    m = sent
    for et, sm, em in ents:
        if et in m:
            m = m.replace(et, f"{sm} {et} {em}", 1)
        else:
            m = f"{m} {sm} {et} {em}"
    return m


def load_jsonl(fp):
    if not os.path.exists(fp): return []
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content: return []
    try:
        return [json.loads(l) for l in content.split("\n") if l.strip()]
    except json.JSONDecodeError:
        pass
    entries, buf, bc = [], "", 0
    for line in content.split("\n"):
        s = line.strip()
        if not s: continue
        buf += s + " ";
        bc += s.count("{") - s.count("}")
        if bc == 0 and buf.strip():
            try:
                entries.append(json.loads(buf.strip()))
            except:
                pass
            buf = ""
    return entries


def infer(lang, test_file, odir):
    t0 = time.time();
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{dev}|Lang:{lang}|Test:{test_file}")
    cfg = json.load(open(os.path.join(odir, "config.json")))
    l2i = json.load(open(os.path.join(odir, "label2id.json"), encoding="utf-8"))
    i2l = json.load(open(os.path.join(odir, "id2label.json"), encoding="utf-8"))
    e2i = {}
    p = os.path.join(odir, "en_to_indic.json")
    if os.path.exists(p): e2i = json.load(open(p, encoding="utf-8"))
    lmap = e2i.get(lang, {})
    tok = AutoTokenizer.from_pretrained(os.path.join(odir, "tokenizer"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    e1id = tok.convert_tokens_to_ids("[E1S]");
    e2id = tok.convert_tokens_to_ids("[E2S]")
    dt = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModel.from_pretrained(cfg["model_name"], torch_dtype=dt)
    base.resize_token_embeddings(len(tok))
    peft = PeftModel.from_pretrained(base, os.path.join(odir, "lora_adapter"))
    model = REModel(peft, cfg["hidden_size"], cfg["num_labels"])
    model.clf.load_state_dict(
        torch.load(os.path.join(odir, "classifier_head.pt"), map_location="cpu", weights_only=True))
    model.to(dev).eval()
    print(f"Loaded in {(time.time() - t0) / 60:.1f}m")

    data = load_jsonl(test_file);
    print(f"Samples:{len(data)}")
    items = []
    for ei, e in enumerate(data):
        s = e.get("sentText", "")
        for ri, rm in enumerate(e.get("relationMentions", [])):
            items.append((ei, ri, mark(s, rm.get("em1Text", ""), rm.get("em2Text", ""))))

    BS = 64;
    preds = {};
    amp = dev.type == "cuda"
    amp_dt = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
    for bs in range(0, len(items), BS):
        batch = items[bs:bs + BS]
        enc = tok([it[2] for it in batch], max_length=cfg["max_seq_len"], truncation=True, padding="max_length",
                  return_tensors="pt")
        ids = enc["input_ids"].to(dev);
        mask = enc["attention_mask"].to(dev)
        e1p, e2p = [], []
        for i in range(len(batch)):
            last = mask[i].sum().item() - 1
            e1 = (ids[i] == e1id).nonzero(as_tuple=True)[0];
            e2 = (ids[i] == e2id).nonzero(as_tuple=True)[0]
            e1p.append(e1[0].item() if len(e1) > 0 else last);
            e2p.append(e2[0].item() if len(e2) > 0 else last)
        with torch.no_grad():
            if amp:
                with torch.amp.autocast('cuda', dtype=amp_dt):
                    logits = model(ids, mask, torch.tensor(e1p, device=dev), torch.tensor(e2p, device=dev))
            else:
                logits = model(ids, mask, torch.tensor(e1p, device=dev), torch.tensor(e2p, device=dev))
        for i, it in enumerate(batch):
            el = i2l[str(logits.argmax(-1)[i].item())]
            preds[(it[0], it[1])] = lmap.get(el, el) if lang != "en" and lmap else el

    of = os.path.join(odir, f"output_{lang}.jsonl")
    with open(of, "w", encoding="utf-8") as f:
        for ei, e in enumerate(data):
            f.write(json.dumps(
                {"articleId": e.get("articleId", ""), "sentId": e.get("sentId", ""), "sentText": e.get("sentText", ""),
                 "relationMentions": [{"em1Text": rm.get("em1Text", ""), "em2Text": rm.get("em2Text", ""),
                                       "label": preds.get((ei, ri), "NA")} for ri, rm in
                                      enumerate(e.get("relationMentions", []))]}, ensure_ascii=False) + "\n")
    print(f"Saved:{of}|{(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    if len(sys.argv) < 4: print("Usage: python infer_classifier.py <lang> <test_file> <output_dir>"); sys.exit(1)
    infer(sys.argv[1], sys.argv[2], sys.argv[3])