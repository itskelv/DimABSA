import json
import random
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from scipy.stats import pearsonr
from tqdm import tqdm
import math
import re
import requests


def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "laptop" #change what domain you want to test

train_url = f"dataset/{lang}/{lang}_{domain}_train_alltasks.jsonl"
predict_url = f"dataset/{lang}/{lang}_{domain}_dev_{task}.jsonl"

# --- model / training config ---
model_name = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
lr = 2e-5             
sigma_lr = 5e-2       
weight_decay = 0.01  
epochs = 25          
patience = 3         
batch_size = 16       
accum_steps = 4        
warmup_ratio = 0.1     
clip_norm = 1.0        
dropout = 0.1
max_len = 256          
seeds = [21, 99, 42]   

use_aspect_marker = True
ASP_OPEN, ASP_CLOSE = "[ASP]", "[/ASP]"

train_raw = load_jsonl(train_url)
predict_raw = load_jsonl(predict_url)

def jsonl_to_df(data):
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    return df

train_df = jsonl_to_df(train_raw)
predict_df = jsonl_to_df(predict_raw)

# split 10% for dev
train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

def show(title, df):
    print(f"\n=== {lang}_{domain} {title} ===")
    print(df.head().to_string())

show("train_df", train_df)
show("dev_df", dev_df)
show("predict_df", predict_df)


class VADataset(Dataset):
    '''
    Valence-Arousal regression dataset. Returns raw strings + labels;
    tokenization happens in the collate_fn so padding is dynamic.
    '''
    def __init__(self, dataframe):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "text": self.sentences[idx],
            "aspect": self.aspects[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def mark_aspect(text, aspect):
    '''
    NEW (aspect marker, as in YangSteam). Returns (segment_1, segment_2_or_None).
    - If the aspect occurs in the text, wrap that occurrence in place:  "... [ASP] screen [/ASP] ..."
    - Otherwise (e.g. implicit aspect "NULL"), keep the text and append the aspect as a second segment:
      [CLS] text [SEP] [ASP] aspect [/ASP] [SEP]
    '''
    aspect = str(aspect)
    m = re.search(re.escape(aspect), text, flags=re.IGNORECASE) if aspect and aspect.upper() != "NULL" else None
    if m:
        s, e = m.span()
        return f"{text[:s]}{ASP_OPEN} {text[s:e]} {ASP_CLOSE}{text[e:]}", None
    return text, f"{ASP_OPEN} {aspect} {ASP_CLOSE}"


def make_collate(tokenizer, max_len=128, use_marker=False):
    def collate(items):
        if use_marker:
            # CHANGED: tokenize per example (some have a 2nd segment, some do not), then pad the batch
            encs = []
            for x in items:
                a, b = mark_aspect(x["text"], x["aspect"])
                enc = tokenizer(a, b, truncation=True, max_length=max_len) if b is not None \
                    else tokenizer(a, truncation=True, max_length=max_len)
                encs.append({k: v for k, v in enc.items()})
            enc = tokenizer.pad(encs, padding=True, return_tensors="pt")
        else:
            enc = tokenizer(
                [x["text"] for x in items],
                [x["aspect"] for x in items],   # second segment -> [CLS] text [SEP] aspect [SEP]
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
        batch = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        batch["labels"] = torch.stack([x["labels"] for x in items])
        return batch
    return collate


tokenizer = AutoTokenizer.from_pretrained(model_name)   # CHANGED: was "bert-base-multilingual-cased"
asp_id = None
if use_aspect_marker:
    # NEW: register the markers as special tokens (the model's embeddings are resized in LogSigmaRegressor)
    tokenizer.add_tokens([ASP_OPEN, ASP_CLOSE], special_tokens=True)
    asp_id = tokenizer.convert_tokens_to_ids(ASP_OPEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cpu":
    print("WARNING: no GPU found. A large encoder x 3 seeds will be extremely slow on CPU.")

collate = make_collate(tokenizer, max_len, use_marker=use_aspect_marker)   # CHANGED: passes the marker switch
train_dataset = VADataset(train_df)
dev_dataset = VADataset(dev_df)
pred_dataset = VADataset(predict_df)

dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=collate)    # CHANGED: shuffle=False
pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False, collate_fn=collate)  # CHANGED: moved here, shuffle=False

class LogSigmaRegressor(nn.Module):
    '''
    Encoder -> [CLS] -> dropout -> two separate linear heads (V, A).
    s = log(sigma^2) = [s_V, s_A] are learned jointly with the model.
    '''
    def __init__(self, model_name, dropout=0.1, vocab_size=None, asp_id=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if vocab_size is not None and vocab_size > self.backbone.get_input_embeddings().num_embeddings:
            self.backbone.resize_token_embeddings(vocab_size)
        self.asp_id = asp_id
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc_v = nn.Linear(hidden, 1)
        self.fc_a = nn.Linear(hidden, 1)
        for fc in (self.fc_v, self.fc_a):
            nn.init.constant_(fc.bias, 5.0)
        self.s = nn.Parameter(torch.log(torch.empty(2).uniform_(0.2, 1.0)))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and getattr(self.backbone.config, "type_vocab_size", 1) > 1:
            kw["token_type_ids"] = token_type_ids
        hs = self.backbone(**kw).last_hidden_state
        h = hs[:, 0]   # [CLS]
        if self.asp_id is not None:
            is_asp = (input_ids == self.asp_id)
            has_asp = is_asp.any(dim=1)
            first = is_asp.int().argmax(dim=1)   # position of the first [ASP]
            asp_h = hs[torch.arange(hs.size(0), device=hs.device), first]
            h = torch.where(has_asp.unsqueeze(1), asp_h, h)
        h = self.dropout(h)
        return torch.cat([self.fc_v(h), self.fc_a(h)], dim=1)   # shape [B, 2] = [V, A]

    def uncertainty_loss(self, outputs, labels):
        # Eq. 5: 0.5*exp(-s_V)*L_V + 0.5*exp(-s_A)*L_A + (s_V + s_A)/2
        lv = F.mse_loss(outputs[:, 0].float(), labels[:, 0])
        la = F.mse_loss(outputs[:, 1].float(), labels[:, 1])
        sv, sa = self.s[0], self.s[1]
        return 0.5 * torch.exp(-sv) * lv + 0.5 * torch.exp(-sa) * la + 0.5 * (sv + sa)

    def train_epoch(self, dataloader, optimizer, scheduler, scaler, device, accum_steps, clip_norm, use_amp):
        self.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = self(**batch)
            loss = self.uncertainty_loss(outputs, labels)
            scaler.scale(loss / accum_steps).backward()  
            if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()                                   
                optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(dataloader)


@torch.no_grad()
def predict(model, dataloader, device, use_amp):
    '''NEW: returns an [N, 2] array of (V, A) predictions, clipped to the 1-9 scale (clipping is my addition).'''
    model.eval()
    outs = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels", None)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch)
        outs.append(outputs.float().cpu().numpy())
    return np.clip(np.vstack(outs), 1.0, 9.0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def official_rmse(pred, gold):
    # identical formula to evaluate_predictions_task1(is_norm=False): sqrt(sum of V and A squared errors / N)
    return math.sqrt(((pred - gold) ** 2).sum() / len(gold))


use_amp = device.type == "cuda"
gold_dev = dev_df[["Valence", "Arousal"]].values.astype(float)   # aligned with dev_loader (shuffle=False)


def train_one_seed(seed):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=g)

    model = LogSigmaRegressor(model_name, dropout, vocab_size=len(tokenizer), asp_id=asp_id).to(device)   # CHANGED: marker args

    # two parameter groups: encoder/heads vs. the log-variances (2500x higher LR, no weight decay)
    main_params = [p for n, p in model.named_parameters() if n != "s"]
    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": lr, "weight_decay": weight_decay},
        {"params": [model.s], "lr": sigma_lr, "weight_decay": 0.0},
    ])
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"rmse": float("inf")}
    bad = 0
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer, scheduler, scaler, device,
                                       accum_steps, clip_norm, use_amp)
        dev_pred = predict(model, dev_loader, device, use_amp)
        rmse = official_rmse(dev_pred, gold_dev)
        s2 = torch.exp(model.s).detach().cpu().numpy()
        print(f"[seed {seed}] epoch {epoch+1}: train_loss={train_loss:.4f} dev_RMSE={rmse:.4f} "
              f"sigma2_V={s2[0]:.3f} sigma2_A={s2[1]:.3f}")
        if rmse < best["rmse"]:
            # keep the predictions of the best epoch (no need to save big checkpoints)
            best = {"rmse": rmse, "dev": dev_pred, "sigma2": s2,
                    "pred": predict(model, pred_loader, device, use_amp)}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[seed {seed}] early stop at epoch {epoch+1}")
                break

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


results = [train_one_seed(s) for s in seeds]
for s, r in zip(seeds, results):
    print(f"seed {s}: best dev RMSE {r['rmse']:.4f}, sigma2 (V, A) = {r['sigma2'].round(3)}")

# 3-seed ensemble = arithmetic mean of predictions (Eq. 6)
dev_ens = np.mean([r["dev"] for r in results], axis=0)
pred_ens = np.mean([r["pred"] for r in results], axis=0)

#==== step 4 use dev data to check your model's performance ====
def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v, is_norm = False):
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range.")
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]

    gold_va = list(gold_v) + list(gold_a)
    pred_va = list(pred_v) + list(pred_a)
    def rmse_norm(gold_va, pred_va, is_normalization = True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        if is_normalization:
            return math.sqrt(sum(result)/len(gold_v))/math.sqrt(128)
        return math.sqrt(sum(result)/len(gold_v))
    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }


pred_v, pred_a = dev_ens[:, 0], dev_ens[:, 1]
gold_v, gold_a = gold_dev[:, 0], gold_dev[:, 1]
eval_score = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
print(f"{model_name} (LogSigma, {len(seeds)}-seed ensemble) dev_eval: {eval_score}")

#==== step 5 save & submit your predict results ====
def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

def df_to_jsonl(df, out_path):
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

predict_df["Valence"] = pred_ens[:, 0] 
predict_df["Arousal"] = pred_ens[:, 1]

df_to_jsonl(predict_df, f"pred_{lang}_{domain}.jsonl")
