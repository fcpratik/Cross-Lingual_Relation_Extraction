"""
Task 3: In-Context Learning for Cross-Lingual RE
=================================================
Uses Meta-Llama-3.1-8B-Instruct with few-shot prompting.
Two backends: vLLM (preferred) or transformers (fallback for Kaggle).
TF-IDF based demo retrieval with character n-grams for multilingual.
"""
import os,sys,json,time
import numpy as np
from collections import defaultdict

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

def load_label_maps(sft_dir):
    i2e,e2i={},{}
    if not os.path.isdir(sft_dir): return i2e,e2i
    for f in os.listdir(sft_dir):
        if f.endswith("_map.json"):
            lang=f.replace("_map.json","")
            with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh: m=json.load(fh)
            e2i[lang]=m
            for ek,iv in m.items(): i2e[iv]=ek
    return i2e,e2i

def build_valid_labels(en_file,sft_dir):
    labels=set()
    for e in load_jsonl(en_file):
        for rm in e.get("relationMentions",[]):
            l=rm.get("label","");
            if l: labels.add(l)
    if os.path.isdir(sft_dir):
        for f in os.listdir(sft_dir):
            if f.endswith("_map.json"):
                with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh: labels.update(json.load(fh).keys())
    labels.discard("")
    return sorted(labels)

def load_demo_pool(sft_dir,en_file,i2e,target_lang):
    demos=[]
    for lang in [target_lang,"hi","kn","or","tcy"]:
        for c in [f"{lang}_train.jsonl",f"{lang}_val.jsonl"]:
            fp=os.path.join(sft_dir,c)
            if not os.path.exists(fp): continue
            for e in load_jsonl(fp):
                s=e.get("sentText","")
                for rm in e.get("relationMentions",[]):
                    l=rm.get("label","NA"); el=i2e.get(l,l)
                    demos.append({"sent":s,"em1":rm.get("em1Text",""),"em2":rm.get("em2Text",""),"en_label":el,"lang":lang})
    # Add stratified English
    if os.path.exists(en_file):
        by_label=defaultdict(list)
        for e in load_jsonl(en_file):
            for rm in e.get("relationMentions",[]):
                l=rm.get("label","NA")
                if l!="NA":
                    by_label[l].append({"sent":e.get("sentText",""),"em1":rm.get("em1Text",""),
                                         "em2":rm.get("em2Text",""),"en_label":l,"lang":"en"})
        for lbl,items in by_label.items(): demos.extend(items[:3])
    print(f"  Demo pool: {len(demos)}")
    return demos

class TFIDFRetriever:
    def __init__(self,demos):
        self.demos=demos
        self.texts=[f"{d['sent']} {d['em1']} {d['em2']}" for d in demos]
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),max_features=50000,sublinear_tf=True)
        self.mat=self.vec.fit_transform(self.texts)
    def retrieve(self,sent,e1,e2,k=8):
        q=self.vec.transform([f"{sent} {e1} {e2}"])
        scores=(self.mat@q.T).toarray().flatten()
        top=np.argsort(scores)[-k:][::-1]
        return [self.demos[i] for i in top]

def closest_label(gen,valid):
    g=gen.strip().split("\n")[0].strip()
    if g in valid: return g
    if g.lower() in ["na","none",""]: return "NA"
    for l in valid:
        if l.lower()==g.lower(): return l
    for l in valid:
        if l in g: return l
    for l in valid:
        if g in l and len(g)>3: return l
    gp=g.strip("/").split("/")
    best,bsc=None,0
    for l in valid:
        lp=l.strip("/").split("/")
        sc=sum(1 for a,b in zip(reversed(gp),reversed(lp)) if a.lower()==b.lower())
        if sc>bsc: bsc=sc; best=l
    if best and bsc>0: return best
    return "NA"

def build_prompt(sent,e1,e2,demos,vlabels):
    llist=", ".join(vlabels)
    sys_msg=f"You are a relation extraction system. Given a sentence and two entities, predict the relation. Valid: {llist}. If none, output NA. Output ONLY the label."
    demo_txt=""
    for d in demos:
        demo_txt+=f"\nSentence: {d['sent']}\nEntity 1: {d['em1']}\nEntity 2: {d['em2']}\nRelation: {d['en_label']}\n"
    query=f"\nSentence: {sent}\nEntity 1: {e1}\nEntity 2: {e2}\nRelation:"
    return sys_msg+"\n"+demo_txt+query

def infer(lang,test_file,odir):
    t0=time.time()
    print(f"Lang:{lang}|Test:{test_file}")
    root=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
    sft_dir=os.path.join(root,"sft_dataset")
    en_file=os.path.join(root,"en_sft_dataset","train.jsonl")

    i2e,e2i=load_label_maps(sft_dir)
    lmap=e2i.get(lang,{})
    vlabels=build_valid_labels(en_file,sft_dir)
    print(f"Labels: {len(vlabels)}")

    print("Loading demos...")
    demos=load_demo_pool(sft_dir,en_file,i2e,lang)
    print("Building retriever...")
    retriever=TFIDFRetriever(demos) if demos else None

    data=load_jsonl(test_file); print(f"Test:{len(data)}")
    items=[]
    for ei,e in enumerate(data):
        s=e.get("sentText","")
        for ri,rm in enumerate(e.get("relationMentions",[])):
            items.append((ei,ri,s,rm.get("em1Text",""),rm.get("em2Text","")))
    print(f"Predictions: {len(items)}")

    # Build prompts
    NUM_DEMOS=6
    prompts=[]
    for it in items:
        _,_,sent,e1,e2=it
        sel=retriever.retrieve(sent,e1,e2,k=NUM_DEMOS) if retriever else []
        prompts.append(build_prompt(sent,e1,e2,sel,vlabels))

    # Try vLLM first, fallback to transformers
    try:
        from vllm import LLM,SamplingParams
        print("Using vLLM backend...")
        model_name="Meta-Llama-3.1-8B-Instruct"
        llm=LLM(model=model_name,max_model_len=4096,gpu_memory_utilization=0.9)
        sp=SamplingParams(temperature=0.0,max_tokens=30,stop=["\n"])
        outputs=llm.generate(prompts,sp)
        generated=[o.outputs[0].text.strip() for o in outputs]
    except Exception as ex:
        print(f"vLLM failed ({ex}), using transformers...")
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        model_name="Meta-Llama-3.1-8B-Instruct"
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt=torch.bfloat16 if dev.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
        tok=AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        tok.padding_side="left"
        model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dt,device_map="auto")
        model.eval()
        generated=[]
        BS=4
        for bs in range(0,len(prompts),BS):
            batch=prompts[bs:bs+BS]
            enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=3500)
            ids=enc["input_ids"].to(dev); mask=enc["attention_mask"].to(dev); ilen=ids.shape[1]
            with torch.no_grad():
                with torch.amp.autocast('cuda',dtype=dt):
                    out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=30,
                                        do_sample=False,pad_token_id=tok.pad_token_id)
            for i in range(len(batch)):
                gen=tok.decode(out[i][ilen:],skip_special_tokens=True).strip().split("\n")[0]
                generated.append(gen)
            if ((bs//BS)+1)%10==0: print(f"  {bs+len(batch)}/{len(items)}|{(time.time()-t0)/60:.1f}m")

    # Map predictions
    preds={}
    for i,it in enumerate(items):
        el=closest_label(generated[i],vlabels+["NA"])
        preds[(it[0],it[1])]=lmap.get(el,el) if lang!="en" and lmap else el

    os.makedirs(odir,exist_ok=True)
    of=os.path.join(odir,f"output_{lang}.jsonl")
    with open(of,"w",encoding="utf-8") as f:
        for ei,e in enumerate(data):
            f.write(json.dumps({"articleId":e.get("articleId",""),"sentId":e.get("sentId",""),"sentText":e.get("sentText",""),
                "relationMentions":[{"em1Text":rm.get("em1Text",""),"em2Text":rm.get("em2Text",""),
                    "label":preds.get((ei,ri),"NA")} for ri,rm in enumerate(e.get("relationMentions",[]))]},ensure_ascii=False)+"\n")
    print(f"\nSaved:{of}|{(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    if len(sys.argv)<4: print("Usage: python icl_inference.py <lang> <test> <odir>"); sys.exit(1)
    infer(sys.argv[1],sys.argv[2],sys.argv[3])