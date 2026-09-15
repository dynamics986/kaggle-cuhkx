"""Budgeted GRU/Transformer search on the original 64-step sensor streams.

This intentionally does not turn a clip into one tabular row: temporal order is
preserved for the recurrent and attention models.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cuhkx_har.features import cache_key
from cuhkx_har.splits import fold_partition
from cuhkx_sep12.common import submit

ROOT = Path(__file__).resolve().parents[2]
NCLASS = 40
WIDTHS = (68, 80, 13)


class Clips(Dataset):
    def __init__(self, frame, cache, split):
        self.frame, self.cache, self.split = frame.reset_index(drop=True), Path(cache), split
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        r = self.frame.iloc[index]
        with np.load(self.cache / cache_key(self.split, str(r.clip_id))) as z:
            values = [z[k].astype(np.float32) for k in ("skeleton", "imu", "radar")]
            mask = z["sensor_mask"].astype(np.float32)
        return (*map(torch.from_numpy, values), torch.from_numpy(mask), int(r.label) if "label" in r else -1, str(r.clip_id))


class SensorNet(nn.Module):
    def __init__(self, kind, dim, layers, dropout):
        super().__init__(); self.kind = kind
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(w), nn.Linear(w, dim), nn.GELU()) for w in WIDTHS])
        if kind == "gru":
            self.enc = nn.ModuleList([nn.GRU(dim, dim // 2, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0) for _ in WIDTHS])
        else:
            layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim*2, dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.ModuleList([nn.TransformerEncoder(layer, num_layers=layers) for _ in WIDTHS]); self.position = nn.Parameter(torch.zeros(1, 64, dim)); nn.init.normal_(self.position, std=.02)
        self.score = nn.ModuleList([nn.Linear(dim, 1) for _ in WIDTHS]); self.head = nn.Sequential(nn.LayerNorm(dim*3+3), nn.Linear(dim*3+3, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, NCLASS))
    def forward(self, *items):
        streams, mask = items[:3], items[3]; output=[]
        for i, x in enumerate(streams):
            x = self.proj[i](x)
            encoded = self.enc[i](x if self.kind == "gru" else x + self.position[:, :x.shape[1]])
            x = encoded[0] if self.kind == "gru" else encoded
            a = torch.softmax(self.score[i](x).squeeze(-1), 1); output.append((x*a.unsqueeze(-1)).sum(1))
        return self.head(torch.cat([*output, mask], 1))


def batches(loader, device):
    for a,b,c,m,y,ids in loader: yield (a.to(device),b.to(device),c.to(device),m.to(device)), y.to(device), ids

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ps=[]; ys=[]; ids=[]
    for x,y,names in batches(loader,device): ps.append(model(*x).softmax(1).cpu()); ys.append(y.cpu()); ids += list(names)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy(), ids


def one_fold(kind, fold, train, test, cache, out, seconds, device):
    train_df, valid_df = fold_partition(train, fold); out.mkdir(parents=True, exist_ok=True)
    seed=20250915+fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr=DataLoader(Clips(train_df,cache,"train"), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    va=DataLoader(Clips(valid_df,cache,"train"), batch_size=32, num_workers=4, pin_memory=True)
    te=DataLoader(Clips(test,cache,"test"), batch_size=32, num_workers=4, pin_memory=True)
    candidates=[(96,1,.10,1e-3),(128,1,.15,7e-4),(128,2,.10,5e-4),(192,2,.15,4e-4)]
    deadline=time.monotonic()+seconds; best=(-1,None,None); trials=[]
    for trial,(dim,layers,drop,lr) in enumerate(candidates,1):
        if time.monotonic() >= deadline: break
        model=SensorNet(kind,dim,layers,drop).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); patience=0; trial_best=-1
        for epoch in range(1,81):
            if time.monotonic() >= deadline: break
            model.train(); loss_sum=n=0
            for x,y,_ in batches(tr,device):
                opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(*x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); loss_sum+=loss.item()*len(y); n+=len(y)
            prob,y,_=evaluate(model,va,device); acc=float((prob.argmax(1)==y).mean()); print(f"fold={fold} {kind} trial={trial} epoch={epoch} loss={loss_sum/max(n,1):.4f} val_acc={acc:.4f}",flush=True)
            if acc>trial_best: trial_best=acc; patience=0; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            else: patience+=1
            if patience>=10: break
        trials.append({"trial":trial,"dim":dim,"layers":layers,"dropout":drop,"lr":lr,"accuracy":trial_best})
        if trial_best>best[0]: best=(trial_best,(dim,layers,drop,lr),state)
    if best[1] is None: raise RuntimeError("time budget expired before the first trial")
    dim,layers,drop,lr=best[1]; model=SensorNet(kind,dim,layers,drop).to(device); model.load_state_dict(best[2]); prob,y,ids=evaluate(model,va,device); pd.DataFrame({"clip_id":ids,"label":y,"prediction":prob.argmax(1)}).to_csv(out/"validation_predictions.csv",index=False)
    torch.save({"kind":kind,"params":best[1],"state_dict":best[2]},out/"model.pt")
    test_prob,_,test_ids=evaluate(model,te,device); np.save(out/"test_probabilities.npy",test_prob); (out/"test_ids.json").write_text(json.dumps(test_ids)); summary={"fold":fold,"validation_accuracy":best[0],"best_params":{"dim":dim,"layers":layers,"dropout":drop,"lr":lr},"trials":trials}; (out/"summary.json").write_text(json.dumps(summary,indent=2)); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--kind",choices=("gru","transformer"),required=True); p.add_argument("--output",required=True); p.add_argument("--cache",default="artifacts/sep12/serial_depth_align/sensors"); p.add_argument("--folds",nargs="+",type=int,default=[2,4]); p.add_argument("--seconds-per-fold",type=int,default=7200); p.add_argument("--device",default="cuda"); a=p.parse_args()
    train=pd.read_csv(ROOT/"manifests/cv5/train.csv"); test=pd.read_csv(ROOT/"manifests/cv5/test.csv"); out=Path(a.output); device=torch.device(a.device if torch.cuda.is_available() else "cpu"); summaries=[]; probs=[]
    for fold in a.folds:
        s=one_fold(a.kind,fold,train,test,a.cache,out/f"fold_{fold}",a.seconds_per_fold,device); summaries.append(s); probs.append(np.load(out/f"fold_{fold}/test_probabilities.npy"))
    ids=json.loads((out/f"fold_{a.folds[0]}/test_ids.json").read_text()); pred=np.mean(probs,0).argmax(1); mapping=dict(zip(ids,pred)); test_rows=pd.read_csv(ROOT/"manifests/cv5/test.csv"); ordered=np.asarray([mapping[x] for x in test_rows.clip_id]); submit(test_rows, np.eye(NCLASS, dtype=np.float32)[ordered], ROOT/"../Small-Model-Track/Testing/test_file/test.csv", out/"submission.csv"); (out/"summary.json").write_text(json.dumps({"kind":a.kind,"folds":summaries},indent=2))

if __name__=="__main__": main()
