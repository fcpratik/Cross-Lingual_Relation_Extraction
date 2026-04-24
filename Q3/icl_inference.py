"""
Task 3: ICL for Cross-Lingual RE
- Primary: meta-llama/Meta-Llama-3.1-8B-Instruct (HF id) or TA-provided local path
- Fallback: Qwen/Qwen2.5-7B-Instruct
- TF-IDF retrieval for demo selection
"""
import os,sys,json,time,numpy as np
from collections import defaultdict

# TA-provided local model path (Update 2)
LLAMA_LOCAL_PATH = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

def read_jsonl(fp):
    data=[]
    if not os.path.exists(fp):return data
    with open(fp,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                try:data.append(json.loads(line))
                except:pass
    if data:return data
    with open(fp,'r',encoding='utf-8') as f:content=f.read().strip()
    buf,bc=""  ,0
    for line in content.split("\n"):
        s=line.strip()
        if not s:continue
        buf+=s+" ";bc+=s.count("{")-s.count("}")
        if bc==0 and buf.strip():
            try:data.append(json.loads(buf.strip()))
            except:pass
            buf=""
    return data

def load_label_maps(sft_dir):
    i2e,e2i={},{}
    if not os.path.isdir(sft_dir):return i2e,e2i
    for f in os.listdir(sft_dir):
        if f.endswith("_map.json"):
            lang=f.replace("_map.json","")
            with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh:m=json.load(fh)
            e2i[lang]=m
            for ek,iv in m.items():i2e[iv]=ek
    return i2e,e2i

def build_valid_labels(en_file,sft_dir):
    labels=set()
    for e in read_jsonl(en_file):
        for rm in e.get("relationMentions",[]):
            l=rm.get("label","")
            if l:labels.add(l)
    if os.path.isdir(sft_dir):
        for f in os.listdir(sft_dir):
            if f.endswith("_map.json"):
                with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh:labels.update(json.load(fh).keys())
    labels.discard("")
    return sorted(labels)

def load_demo_pool(sft_dir,en_file,i2e,target_lang):
    demos=[]
    for lang in [target_lang,"hi","kn","or","tcy"]:
        for c in [f"{lang}_train.jsonl",f"{lang}_val.jsonl"]:
            fp=os.path.join(sft_dir,c)
            if not os.path.exists(fp):continue
            for e in read_jsonl(fp):
                s=e.get("sentText","")
                for rm in e.get("relationMentions",[]):
                    l=rm.get("label","NA");el=i2e.get(l,l)
                    demos.append({"sent":s,"em1":rm.get("em1Text",""),"em2":rm.get("em2Text",""),"en_label":el,"lang":lang})
    if os.path.exists(en_file):
        by_label=defaultdict(list)
        for e in read_jsonl(en_file):
            for rm in e.get("relationMentions",[]):
                l=rm.get("label","NA")
                if l!="NA":
                    by_label[l].append({"sent":e.get("sentText",""),"em1":rm.get("em1Text",""),
                                         "em2":rm.get("em2Text",""),"en_label":l,"lang":"en"})
        for lbl,items in by_label.items():demos.extend(items[:3])
    print(f"  Demo pool: {len(demos)}")
    return demos

class TFIDFRetriever:
    def __init__(self,demos):
        self.demos=demos
        self.texts=[f"{d['sent']} {d['em1']} {d['em2']}" for d in demos]
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),max_features=50000,sublinear_tf=True)
        self.mat=self.vec.fit_transform(self.texts)
    def retrieve(self,sent,e1,e2,k=6):
        q=self.vec.transform([f"{sent} {e1} {e2}"])
        scores=(self.mat@q.T).toarray().flatten()
        top=np.argsort(scores)[-k:][::-1]
        return[self.demos[i] for i in top]

def closest_label(gen,valid):
    g=gen.strip().split("\n")[0].strip()
    if g in valid:return g
    if g.lower() in["na","none",""]:return"NA"
    for l in valid:
        if l.lower()==g.lower():return l
    for l in valid:
        if l in g:return l
    for l in valid:
        if g in l and len(g)>3:return l
    gp=g.strip("/").split("/")
    best,bsc=None,0
    for l in valid:
        lp=l.strip("/").split("/")
        sc=sum(1 for a,b in zip(reversed(gp),reversed(lp)) if a.lower()==b.lower())
        if sc>bsc:bsc=sc;best=l
    if best and bsc>0:return best
    return"NA"

def build_prompt(sent,e1,e2,demos,vlabels):
    llist=", ".join(vlabels)
    sys_msg=f"You are a relation extraction system. Given a sentence and two entities, predict the relation. Valid: {llist}. If none, output NA. Output ONLY the label."
    demo_txt=""
    for d in demos:
        demo_txt+=f"\nSentence: {d['sent']}\nEntity 1: {d['em1']}\nEntity 2: {d['em2']}\nRelation: {d['en_label']}\n"
    query=f"\nSentence: {sent}\nEntity 1: {e1}\nEntity 2: {e2}\nRelation:"
    return sys_msg+"\n"+demo_txt+query

def get_model_candidates():
    """Return list of model paths/names to try in order."""
    candidates = []
    # First try TA-provided local path if it exists
    if os.path.exists(LLAMA_LOCAL_PATH):
        candidates.append(LLAMA_LOCAL_PATH)
    # Then HF model IDs
    candidates.extend([
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ])
    return candidates

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
    data=read_jsonl(test_file);print(f"Test:{len(data)}")
    items=[]
    for ei,e in enumerate(data):
        s=e.get("sentText","")
        for ri,rm in enumerate(e.get("relationMentions",[])):
            items.append((ei,ri,s,rm.get("em1Text",""),rm.get("em2Text","")))
    print(f"Predictions: {len(items)}")

    NUM_DEMOS=6
    prompts=[]
    for it in items:
        _,_,sent,e1,e2=it
        sel=retriever.retrieve(sent,e1,e2,k=NUM_DEMOS) if retriever else[]
        prompts.append(build_prompt(sent,e1,e2,sel,vlabels))

    generated = None
    model_candidates = get_model_candidates()

    # Try vLLM first
    try:
        from vllm import LLM,SamplingParams
        print("Using vLLM backend...")
        for mname in model_candidates:
            try:
                print(f"  Trying {mname}")
                llm=LLM(model=mname,max_model_len=4096,gpu_memory_utilization=0.9)
                sp=SamplingParams(temperature=0.0,max_tokens=30,stop=["\n"])
                outputs=llm.generate(prompts,sp)
                generated=[o.outputs[0].text.strip() for o in outputs]
                print(f"  vLLM OK with {mname}")
                break
            except Exception as ex:
                print(f"  vLLM+{mname} failed: {str(ex)[:150]}")
                continue
    except ImportError:
        print("vLLM not available.")

    if generated is None:
        print("Using transformers backend...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt=torch.bfloat16 if dev.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32

        tok, model = None, None
        for mname in model_candidates:
            try:
                print(f"  Loading {mname}")
                tok = AutoTokenizer.from_pretrained(mname)
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=dt, device_map="auto")
                model.eval()
                print(f"  Loaded: {mname}")
                break
            except Exception as ex:
                print(f"  Failed: {mname} ({type(ex).__name__}: {str(ex)[:100]})")
                continue

        if model is None:
            print("ERROR: No model loaded. Using NA for all.")
            generated = ["NA"] * len(prompts)
        else:
            generated=[]
            BS=2
            for bs in range(0,len(prompts),BS):
                batch=prompts[bs:bs+BS]
                enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=3500)
                ids=enc["input_ids"].to(dev);mask=enc["attention_mask"].to(dev);ilen=ids.shape[1]
                with torch.no_grad():
                    out=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=30,
                                        do_sample=False,pad_token_id=tok.pad_token_id)
                for i in range(len(batch)):
                    gen=tok.decode(out[i][ilen:],skip_special_tokens=True).strip().split("\n")[0]
                    generated.append(gen)
                if((bs//BS)+1)%20==0:print(f"  {bs+len(batch)}/{len(items)}|{(time.time()-t0)/60:.1f}m")

    preds={}
    for i,it in enumerate(items):
        el=closest_label(generated[i],vlabels+["NA"])
        preds[(it[0],it[1])]=lmap.get(el,el) if lang!="en" and lmap else el

    os.makedirs(odir,exist_ok=True)
    # TA-required output name: Q3_<lang>.jsonl
    of=os.path.join(odir,f"Q3_{lang}.jsonl")
    with open(of,"w",encoding="utf-8") as f:
        for ei,e in enumerate(data):
            out={"articleId":e.get("articleId",""),"sentId":e.get("sentId",""),
                "sentText":e.get("sentText",""),
                "relationMentions":[{"em1Text":rm.get("em1Text",""),"em2Text":rm.get("em2Text",""),
                    "label":preds.get((ei,ri),"NA")} for ri,rm in enumerate(e.get("relationMentions",[]))]}
            f.write(json.dumps(out,ensure_ascii=False)+"\n")
    print(f"\nSaved:{of}|{(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    if len(sys.argv)<4:print("Usage: python icl_inference.py <lang> <test> <odir>");sys.exit(1)
    infer(sys.argv[1],sys.argv[2],sys.argv[3])
