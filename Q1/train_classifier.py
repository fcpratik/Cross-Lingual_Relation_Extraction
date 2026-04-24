"""Task 1: RE with Classification Head"""
import os,sys,json,random,time,numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel,get_linear_schedule_with_warmup
from peft import LoraConfig,get_peft_model,TaskType
from collections import Counter

class Config:
    model_name="Qwen/Qwen2.5-1.5B"
    lora_r=32;lora_alpha=64;lora_dropout=0.05
    lora_target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
    batch_size=64;gradient_accumulation_steps=1
    learning_rate_lora=2e-4;learning_rate_head=1e-3
    num_epochs=5;max_seq_len=200
    warmup_ratio=0.06;weight_decay=0.01;max_grad_norm=1.0
    max_en_samples=60000
    max_train_minutes=135
    seed=42
    num_pool_layers=4
    rare_oversample=4
    rare_threshold=500

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def load_jsonl(fp):
    if not os.path.exists(fp):return[]
    with open(fp,"r",encoding="utf-8") as f:content=f.read().strip()
    if not content:return[]
    try:return[json.loads(l) for l in content.split("\n") if l.strip()]
    except json.JSONDecodeError:pass
    entries,buf,bc=[],"",0
    for line in content.split("\n"):
        s=line.strip()
        if not s:continue
        buf+=s+" ";bc+=s.count("{")-s.count("}")
        if bc==0 and buf.strip():
            try:entries.append(json.loads(buf.strip()))
            except:pass
            buf=""
    return entries

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

def build_labels(en_file,sft_dir):
    labels=set()
    for e in load_jsonl(en_file):
        for rm in e.get("relationMentions",[]):
            l=rm.get("label","")
            if l:labels.add(l)
    if os.path.isdir(sft_dir):
        for f in os.listdir(sft_dir):
            if f.endswith("_map.json"):
                with open(os.path.join(sft_dir,f),"r",encoding="utf-8") as fh:labels.update(json.load(fh).keys())
    labels.discard("NA");labels.discard("")
    sl=["NA"]+sorted(labels)
    return{l:i for i,l in enumerate(sl)},{i:l for i,l in enumerate(sl)}

def load_samples(en_file,sft_dir,label2id,i2e,max_en,rare_factor,rare_threshold):
    samples=[];en=[]
    for e in load_jsonl(en_file):
        s=e.get("sentText","")
        for rm in e.get("relationMentions",[]):
            l=rm.get("label","NA")
            if l not in label2id:l="NA"
            en.append({"sent":s,"em1":rm.get("em1Text",""),"em2":rm.get("em2Text",""),"label":l})
    if len(en)>max_en:
        by_label={}
        for s in en:by_label.setdefault(s["label"],[]).append(s)
        sub=[];per=max(20,max_en//len(by_label))
        for lbl,items in by_label.items():random.shuffle(items);sub.extend(items[:per])
        rem=max_en-len(sub)
        if rem>0:
            used=set(id(s) for s in sub);rest=[s for s in en if id(s) not in used]
            random.shuffle(rest);sub.extend(rest[:rem])
        en=sub[:max_en]
    label_counts=Counter(s["label"] for s in en)
    augmented=[]
    for s in en:
        augmented.append(s)
        if label_counts[s["label"]]<rare_threshold and s["label"]!="NA":
            for _ in range(rare_factor-1):augmented.append(s)
    en=augmented
    samples.extend(en);print(f"  English: {len(en)}")
    indic=[]
    for lang in ["hi","kn","or","tcy"]:
        for c in [f"{lang}_train.jsonl",f"{lang}_val.jsonl"]:
            fp=os.path.join(sft_dir,c)
            if os.path.exists(fp):
                for e in load_jsonl(fp):
                    s=e.get("sentText","")
                    for rm in e.get("relationMentions",[]):
                        il=rm.get("label","NA");el=i2e.get(il,il)
                        if el not in label2id:el="NA"
                        indic.append({"sent":s,"em1":rm.get("em1Text",""),"em2":rm.get("em2Text",""),"label":el})
                break
    if indic and len(en)>0:
        factor=min(15,max(3,len(en)//(5*max(len(indic),1))))
        indic=indic*factor;print(f"  Indic {factor}x: {len(indic)}")
    samples.extend(indic);random.shuffle(samples);print(f"  Total: {len(samples)}")
    return samples

class REDataset(Dataset):
    def __init__(self,samples,tok,label2id,maxlen):
        self.samples=samples;self.tok=tok;self.label2id=label2id;self.maxlen=maxlen
        lc=Counter(s["label"] for s in samples);t=len(samples)
        self.cw={label2id[l]:min(t/(len(lc)*c),10.0) for l,c in lc.items()}
    def __len__(self):return len(self.samples)
    @staticmethod
    def mark(sent,e1,e2):
        ents=[(e1,"[E1S]","[E1E]"),(e2,"[E2S]","[E2E]")]
        ents.sort(key=lambda x:len(x[0]),reverse=True)
        m=sent
        for et,sm,em in ents:
            if et in m:m=m.replace(et,f"{sm} {et} {em}",1)
            else:m=f"{m} {sm} {et} {em}"
        return m
    def __getitem__(self,i):
        s=self.samples[i];m=self.mark(s["sent"],s["em1"],s["em2"])
        enc=self.tok(m,max_length=self.maxlen,truncation=True,padding="max_length",return_tensors="pt")
        ids=enc["input_ids"].squeeze(0);mask=enc["attention_mask"].squeeze(0)
        e1id=self.tok.convert_tokens_to_ids("[E1S]");e2id=self.tok.convert_tokens_to_ids("[E2S]")
        e1p=(ids==e1id).nonzero(as_tuple=True)[0];e2p=(ids==e2id).nonzero(as_tuple=True)[0]
        last=mask.sum().item()-1
        return{"input_ids":ids,"attention_mask":mask,
               "e1_pos":torch.tensor(e1p[0].item() if len(e1p)>0 else last),
               "e2_pos":torch.tensor(e2p[0].item() if len(e2p)>0 else last),
               "label":torch.tensor(self.label2id[s["label"]])}

class REModel(nn.Module):
    def __init__(self,base,hs,nl,num_pool_layers=4,drop=0.1):
        super().__init__();self.base=base;self.num_pool_layers=num_pool_layers
        self.clf=nn.Sequential(
            nn.Linear(hs*4,hs*2),nn.GELU(),nn.Dropout(drop),
            nn.Linear(hs*2,hs),nn.GELU(),nn.Dropout(drop),
            nn.Linear(hs,nl))
    def forward(self,ids,mask,e1,e2):
        outputs=self.base(input_ids=ids,attention_mask=mask,output_hidden_states=True)
        h=torch.stack(outputs.hidden_states[-self.num_pool_layers:],dim=0).mean(dim=0)
        b=torch.arange(h.size(0),device=h.device)
        e1_h=h[b,e1];e2_h=h[b,e2]
        feat=torch.cat([e1_h,e2_h,torch.abs(e1_h-e2_h),e1_h*e2_h],dim=-1)
        return self.clf(feat)

def save_ckpt(model,peft,tok,l2i,i2l,e2i,cfg,odir,ep):
    os.makedirs(odir,exist_ok=True)
    peft.save_pretrained(os.path.join(odir,"lora_adapter"))
    tok.save_pretrained(os.path.join(odir,"tokenizer"))
    torch.save(model.clf.state_dict(),os.path.join(odir,"classifier_head.pt"))
    for name,data in[("label2id.json",l2i),("id2label.json",{str(k):v for k,v in i2l.items()}),("en_to_indic.json",e2i)]:
        with open(os.path.join(odir,name),"w",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2)
    with open(os.path.join(odir,"config.json"),"w") as f:
        json.dump({"model_name":cfg.model_name,"max_seq_len":cfg.max_seq_len,
                    "num_labels":len(l2i),"hidden_size":peft.config.hidden_size,
                    "num_pool_layers":cfg.num_pool_layers,"epoch":ep},f,indent=2)
    print(f"  [Saved epoch {ep}]")

def train(cfg,odir,root):
    set_seed(cfg.seed);t0=time.time()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    if dev.type=="cuda":print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    en_file=os.path.join(root,"en_sft_dataset","train.jsonl")
    sft_dir=os.path.join(root,"sft_dataset")
    if not os.path.exists(en_file):print(f"ERROR: {en_file}");sys.exit(1)
    i2e,e2i=load_label_maps(sft_dir);l2i,i2l=build_labels(en_file,sft_dir)
    nl=len(l2i);print(f"Labels: {nl}")
    tok=AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:tok.pad_token=tok.eos_token;tok.pad_token_id=tok.eos_token_id
    tok.add_special_tokens({"additional_special_tokens":["[E1S]","[E1E]","[E2S]","[E2E]"]})
    samples=load_samples(en_file,sft_dir,l2i,i2e,cfg.max_en_samples,cfg.rare_oversample,cfg.rare_threshold)
    ds=REDataset(samples,tok,l2i,cfg.max_seq_len)
    cw=torch.ones(nl)
    for i,w in ds.cw.items():cw[i]=w
    cw=cw.to(dev)
    dl=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    spe=len(dl);tot_steps=(spe//cfg.gradient_accumulation_steps)*cfg.num_epochs
    print(f"Batches/epoch: {spe} | Total steps: {tot_steps}")
    dt=torch.bfloat16 if dev.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base=AutoModel.from_pretrained(cfg.model_name,torch_dtype=dt)
    base.resize_token_embeddings(len(tok))
    lora=LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=cfg.lora_r,lora_alpha=cfg.lora_alpha,
                     lora_dropout=cfg.lora_dropout,target_modules=cfg.lora_target_modules,bias="none")
    peft_model=get_peft_model(base,lora)
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()
    peft_model.print_trainable_parameters()
    model=REModel(peft_model,peft_model.config.hidden_size,nl,cfg.num_pool_layers,cfg.lora_dropout).to(dev)
    model.clf.float()
    if dev.type=="cuda":print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    opt=torch.optim.AdamW([
        {"params":[p for n,p in model.base.named_parameters() if p.requires_grad],"lr":cfg.learning_rate_lora},
        {"params":model.clf.parameters(),"lr":cfg.learning_rate_head}
    ],weight_decay=cfg.weight_decay)
    sched=get_linear_schedule_with_warmup(opt,int(tot_steps*cfg.warmup_ratio),tot_steps)
    amp_on=dev.type=="cuda"
    amp_dt=torch.bfloat16 if(amp_on and torch.cuda.is_bf16_supported())else torch.float16
    print(f"\n{'='*50}\nTRAINING: {cfg.num_epochs} epochs, budget {cfg.max_train_minutes}m\n{'='*50}")
    model.train()
    for ep in range(cfg.num_epochs):
        ep0=time.time();eloss=0;nb=0;cor=0;tot=0;opt.zero_grad()
        for bi,batch in enumerate(dl):
            if(time.time()-t0)/60>cfg.max_train_minutes:
                print(f"\n  TIME LIMIT. Saving.");save_ckpt(model,peft_model,tok,l2i,i2l,e2i,cfg,odir,ep+1);return
            ids=batch["input_ids"].to(dev,non_blocking=True);mask=batch["attention_mask"].to(dev,non_blocking=True)
            e1=batch["e1_pos"].to(dev,non_blocking=True);e2=batch["e2_pos"].to(dev,non_blocking=True)
            lab=batch["label"].to(dev,non_blocking=True)
            if amp_on:
                with torch.amp.autocast('cuda',dtype=amp_dt):
                    logits=model(ids,mask,e1,e2).float()
                    loss=F.cross_entropy(logits,lab,weight=cw)/cfg.gradient_accumulation_steps
            else:
                logits=model(ids,mask,e1,e2).float()
                loss=F.cross_entropy(logits,lab,weight=cw)/cfg.gradient_accumulation_steps
            loss.backward()
            if(bi+1)%cfg.gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
                opt.step();sched.step();opt.zero_grad()
            eloss+=loss.item()*cfg.gradient_accumulation_steps;nb+=1
            cor+=(logits.argmax(-1)==lab).sum().item();tot+=lab.size(0)
            if(bi+1)%100==0:
                print(f"  E{ep+1}|B{bi+1}/{spe}|L:{eloss/nb:.4f}|A:{cor/tot:.4f}|T:{(time.time()-t0)/60:.1f}m")
        print(f"Epoch {ep+1}|L:{eloss/nb:.4f}|A:{cor/tot:.4f}|{(time.time()-ep0)/60:.1f}m|Tot:{(time.time()-t0)/60:.1f}m")
        save_ckpt(model,peft_model,tok,l2i,i2l,e2i,cfg,odir,ep+1)
        if dev.type=="cuda":torch.cuda.empty_cache()
    print(f"Done! {(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    train(Config(),sys.argv[1] if len(sys.argv)>1 else "output",sys.argv[2] if len(sys.argv)>2 else "..")
