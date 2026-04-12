"""
Task 2: Relation Extraction via Autoregressive Generation (Optimized)
=====================================================================
Key changes: max_en=15000, batch=8, grad_accum=4, max_seq_len=220
Targets 3 full epochs in ~120 min on T4
"""
import os,sys,json,random,time
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM,get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model,TaskType

class Config:
    model_name="Qwen/Qwen2.5-1.5B"
    lora_r=16;lora_alpha=32;lora_dropout=0.05
    lora_target_modules=["q_proj","v_proj","k_proj","o_proj"]
    batch_size=4;gradient_accumulation_steps=8
    learning_rate=2e-4;num_epochs=3
    max_input_len=180;max_output_len=40;max_seq_len=220
    warmup_ratio=0.06;weight_decay=0.01;max_grad_norm=1.0
    max_en_samples=8000          # Reduced from 15000 for full 3 epochs on T4
    max_train_minutes=140
    seed=42

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

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
            l=rm.get("label","")
            if l: labels.add(l)
    if os.path.isdir(sft_dir):
        for f in os.listdir(sft_dir):
            if f.endswith("_map.json"):
                with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh: labels.update(json.load(fh).keys())
    labels.discard("")
    return sorted(labels)

def fmt_in(s,e1,e2):
    return f"Extract the relation between entities.\nSentence: {s}\nEntity 1: {e1}\nEntity 2: {e2}\nRelation:"

def fmt_out(l): return f" {l}"

def load_samples(en_file,sft_dir,i2e,max_en):
    samples=[]
    en=[]
    for e in load_jsonl(en_file):
        s=e.get("sentText","")
        for rm in e.get("relationMentions",[]):
            en.append((fmt_in(s,rm.get("em1Text",""),rm.get("em2Text","")),fmt_out(rm.get("label","NA"))))
    if len(en)>max_en: random.shuffle(en); en=en[:max_en]
    samples.extend(en); print(f"  English: {len(en)}")

    indic=[]
    for lang in ["hi","kn","or","tcy"]:
        for c in [f"{lang}_train.jsonl",f"{lang}_val.jsonl"]:
            fp=os.path.join(sft_dir,c)
            if os.path.exists(fp):
                for e in load_jsonl(fp):
                    s=e.get("sentText","")
                    for rm in e.get("relationMentions",[]):
                        il=rm.get("label","NA"); el=i2e.get(il,il)
                        indic.append((fmt_in(s,rm.get("em1Text",""),rm.get("em2Text","")),fmt_out(el)))
                break
    if indic and len(en)>0:
        factor=min(15,max(1,len(en)//(5*max(len(indic),1))))
        indic=indic*factor; print(f"  Indic {factor}x: {len(indic)}")
    samples.extend(indic); random.shuffle(samples)
    print(f"  Total: {len(samples)}")
    return samples

class GenDS(Dataset):
    def __init__(self,samples,tok,max_in,max_seq):
        self.samples=samples;self.tok=tok;self.max_in=max_in;self.max_seq=max_seq
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        inp,out=self.samples[i]
        ie=self.tok(inp,max_length=self.max_in,truncation=True,add_special_tokens=False)
        fe=self.tok(inp+out,max_length=self.max_seq,truncation=True,padding="max_length",add_special_tokens=False)
        ids=torch.tensor(fe["input_ids"]); mask=torch.tensor(fe["attention_mask"])
        labels=ids.clone(); labels[:len(ie["input_ids"])]=-100; labels[mask==0]=-100
        return {"input_ids":ids,"attention_mask":mask,"labels":labels}

def save(model,tok,e2i,vlabels,cfg,odir,ep):
    os.makedirs(odir,exist_ok=True)
    model.save_pretrained(os.path.join(odir,"lora_adapter"))
    tok.save_pretrained(os.path.join(odir,"tokenizer"))
    for n,d in [("en_to_indic.json",e2i),("valid_labels.json",vlabels)]:
        with open(os.path.join(odir,n),"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False)
    with open(os.path.join(odir,"config.json"),"w") as f:
        json.dump({"model_name":cfg.model_name,"max_input_len":cfg.max_input_len,
                    "max_output_len":cfg.max_output_len,"max_seq_len":cfg.max_seq_len,"epoch":ep},f,indent=2)
    print(f"  [Saved epoch {ep}]")

def train(cfg,odir,root):
    set_seed(cfg.seed); t0=time.time()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    en_file=os.path.join(root,"en_sft_dataset","train.jsonl")
    sft_dir=os.path.join(root,"sft_dataset")
    i2e,e2i=load_label_maps(sft_dir)
    vlabels=build_valid_labels(en_file,sft_dir)
    print(f"Labels: {len(vlabels)}")

    tok=AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token; tok.pad_token_id=tok.eos_token_id

    samples=load_samples(en_file,sft_dir,i2e,cfg.max_en_samples)
    ds=GenDS(samples,tok,cfg.max_input_len,cfg.max_seq_len)
    dl=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    spe=len(dl); tot_steps=(spe//cfg.gradient_accumulation_steps)*cfg.num_epochs
    print(f"Batches/epoch: {spe} | Steps: {tot_steps}")

    dt=torch.bfloat16 if dev.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base=AutoModelForCausalLM.from_pretrained(cfg.model_name,torch_dtype=dt)
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg.lora_r,lora_alpha=cfg.lora_alpha,
                     lora_dropout=cfg.lora_dropout,target_modules=cfg.lora_target_modules,bias="none")
    model=get_peft_model(base,lora); model.gradient_checkpointing_enable()
    model.print_trainable_parameters(); model.to(dev)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    sched=get_linear_schedule_with_warmup(opt,int(tot_steps*cfg.warmup_ratio),tot_steps)
    amp_on=dev.type=="cuda"; amp_dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=(amp_on and amp_dt==torch.float16))

    print(f"\n{'='*50}\nTRAINING: {cfg.num_epochs} epochs\n{'='*50}")
    model.train()
    for ep in range(cfg.num_epochs):
        ep0=time.time(); eloss=0; nb=0; opt.zero_grad()
        for bi,batch in enumerate(dl):
            if (time.time()-t0)/60>cfg.max_train_minutes:
                print(f"\n  TIME LIMIT"); save(model,tok,e2i,vlabels,cfg,odir,ep+1); return
            ids=batch["input_ids"].to(dev,non_blocking=True)
            mask=batch["attention_mask"].to(dev,non_blocking=True)
            lab=batch["labels"].to(dev,non_blocking=True)
            if amp_on:
                with torch.amp.autocast('cuda',dtype=amp_dt):
                    loss=model(input_ids=ids,attention_mask=mask,labels=lab).loss/cfg.gradient_accumulation_steps
            else: loss=model(input_ids=ids,attention_mask=mask,labels=lab).loss/cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (bi+1)%cfg.gradient_accumulation_steps==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            eloss+=loss.item()*cfg.gradient_accumulation_steps; nb+=1
            if (bi+1)%300==0:
                print(f"  E{ep+1}|B{bi+1}/{spe}|L:{eloss/nb:.4f}|T:{(time.time()-t0)/60:.1f}m")
        print(f"Epoch {ep+1}|L:{eloss/nb:.4f}|T:{(time.time()-ep0)/60:.1f}m|Tot:{(time.time()-t0)/60:.1f}m")
        save(model,tok,e2i,vlabels,cfg,odir,ep+1)
        if dev.type=="cuda": torch.cuda.empty_cache()
    print(f"Done! {(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    train(Config(),sys.argv[1] if len(sys.argv)>1 else "output",sys.argv[2] if len(sys.argv)>2 else "..")