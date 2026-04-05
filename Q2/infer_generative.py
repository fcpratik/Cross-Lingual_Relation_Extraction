"""Task 2: Inference - Autoregressive Generation"""
import os,sys,json,time,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

def load_jsonl(fp):
    if not os.path.exists(fp): return []
    with open(fp,"r",encoding="utf-8") as f: content=f.read().strip()
    if not content: return []
    try: return [json.loads(l) for l in content.split("\n") if l.strip()]
    except json.JSONDecodeError: pass
    entries,buf,bc=[],""  ,0
    for line in content.split("\n"):
        s=line.strip()
        if not s: continue
        buf+=s+" "; bc+=s.count("{")-s.count("}")
        if bc==0 and buf.strip():
            try: entries.append(json.loads(buf.strip()))
            except: pass
            buf=""
    return entries

def fmt_in(s,e1,e2):
    return f"Extract the relation between entities.\nSentence: {s}\nEntity 1: {e1}\nEntity 2: {e2}\nRelation:"

def closest_label(gen,valid):
    g=gen.strip().split("\n")[0].strip()
    if g in valid: return g
    if g.lower() in ["na","none",""]: return "NA"
    gl=g.lower()
    for l in valid:
        if l.lower()==gl: return l
    for l in valid:
        if l in g: return l
    for l in valid:
        if g in l and len(g)>3: return l
    # partial path
    gp=g.strip("/").split("/")
    best,bsc=None,0
    for l in valid:
        lp=l.strip("/").split("/")
        sc=sum(1 for a,b in zip(reversed(gp),reversed(lp)) if a.lower()==b.lower())
        if sc>bsc: bsc=sc; best=l
    if best and bsc>0: return best
    return "NA"

def infer(lang,test_file,odir):
    t0=time.time(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{dev}|Lang:{lang}")
    cfg=json.load(open(os.path.join(odir,"config.json")))
    vlabels=json.load(open(os.path.join(odir,"valid_labels.json"),encoding="utf-8"))
    e2i={}
    p=os.path.join(odir,"en_to_indic.json")
    if os.path.exists(p): e2i=json.load(open(p,encoding="utf-8"))
    lmap=e2i.get(lang,{})

    tok=AutoTokenizer.from_pretrained(os.path.join(odir,"tokenizer"))
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    dt=torch.bfloat16 if dev.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base=AutoModelForCausalLM.from_pretrained(cfg["model_name"],torch_dtype=dt)
    model=PeftModel.from_pretrained(base,os.path.join(odir,"lora_adapter")).to(dev).eval()
    print(f"Loaded in {(time.time()-t0)/60:.1f}m")

    data=load_jsonl(test_file); print(f"Samples:{len(data)}")
    items=[]
    for ei,e in enumerate(data):
        s=e.get("sentText","")
        for ri,rm in enumerate(e.get("relationMentions",[])):
            items.append((ei,ri,fmt_in(s,rm.get("em1Text",""),rm.get("em2Text",""))))
    print(f"Predictions:{len(items)}")

    BS=16; preds={}; amp=dev.type=="cuda"
    amp_dt=torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
    for bs in range(0,len(items),BS):
        batch=items[bs:bs+BS]
        enc=tok([it[2] for it in batch],return_tensors="pt",padding=True,truncation=True,max_length=cfg.get("max_input_len",180))
        ids=enc["input_ids"].to(dev); mask=enc["attention_mask"].to(dev); ilen=ids.shape[1]
        with torch.no_grad():
            if amp:
                with torch.amp.autocast('cuda',dtype=amp_dt):
                    out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=cfg.get("max_output_len",40),
                                        do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id)
            else:
                out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=cfg.get("max_output_len",40),
                                    do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id)
        for i,it in enumerate(batch):
            gen=tok.decode(out[i][ilen:],skip_special_tokens=True).strip()
            el=closest_label(gen,vlabels+["NA"])
            preds[(it[0],it[1])]=lmap.get(el,el) if lang!="en" and lmap else el
        if ((bs//BS)+1)%10==0: print(f"  {bs+len(batch)}/{len(items)}|{(time.time()-t0)/60:.1f}m")

    of=os.path.join(odir,f"output_{lang}.jsonl")
    with open(of,"w",encoding="utf-8") as f:
        for ei,e in enumerate(data):
            f.write(json.dumps({"articleId":e.get("articleId",""),"sentId":e.get("sentId",""),"sentText":e.get("sentText",""),
                "relationMentions":[{"em1Text":rm.get("em1Text",""),"em2Text":rm.get("em2Text",""),
                    "label":preds.get((ei,ri),"NA")} for ri,rm in enumerate(e.get("relationMentions",[]))]},ensure_ascii=False)+"\n")
    print(f"Saved:{of}|{(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    if len(sys.argv)<4: print("Usage: python infer_generative.py <lang> <test> <odir>"); sys.exit(1)
    infer(sys.argv[1],sys.argv[2],sys.argv[3])