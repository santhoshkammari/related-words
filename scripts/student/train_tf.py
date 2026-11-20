"""usage: python -I train_tf.py DATA INIT_STATIC OUT [--layers 2 --epochs 4 --lr 5e-4]"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from datasets import load_from_disk
from model2vec import StaticModel
from bgmodel import tok_ids
from tfmodel import TF
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("init"); ap.add_argument("out")
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--bs", type=int, default=512); ap.add_argument("--tau", type=float, default=0.05)
a = ap.parse_args(); dev = "cuda"
sm = StaticModel.from_pretrained(a.init, quantize_to="float32")
ds = load_from_disk(a.data); texts = ds["text"]; Y = torch.tensor(np.array(ds["embedding"], dtype=np.float32))
ids = torch.cat([tok_ids(sm, texts[s:s+50000]) for s in range(0, len(texts), 50000)]).short(); N = len(texts); print("tok", N, flush=True)
model = TF(torch.tensor(sm.embedding), layers=a.layers).to(dev); print(sum(p.numel() for p in model.parameters())/1e6, "M params", flush=True)
perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); val, tr = perm[:20000], perm[20000:]
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); spe = len(tr)//a.bs
sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs*spe, pct_start=0.05); t0 = time.time()
def fwd(ix):
    i = ids[ix].to(dev).long(); L = int((i >= 0).sum(1).max()); 
    with torch.autocast("cuda", dtype=torch.bfloat16): s = model(i[:, :max(L, 1)])
    return s.float()
for ep in range(a.epochs):
    model.train(); p = tr[torch.randperm(len(tr))]
    for k in range(spe):
        ix = p[k*a.bs:(k+1)*a.bs]; s = fwd(ix); t = Y[ix].to(dev)
        cos = 1-(s*t).sum(-1).mean(); kl = F.kl_div(F.log_softmax(s@s.T/a.tau,-1), F.softmax(t@t.T/a.tau,-1), reduction="batchmean")
        l = cos+kl; opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if k % 500 == 0: print(f"  ep{ep} step{k}/{spe} cos {cos.item():.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad(): v = np.mean([1-(fwd(val[j:j+a.bs])*Y[val[j:j+a.bs]].to(dev)).sum(-1).mean().item() for j in range(0,20000-a.bs+1,a.bs)])
    print(f"ep {ep} val_cos {v:.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save(dict(sd=model.state_dict(), base=a.init, layers=a.layers), a.out)
print("saved", flush=True)
