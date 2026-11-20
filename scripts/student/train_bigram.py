"""usage: python -I train_bigram.py DATA INIT_STATIC OUT [--buckets 200003 --epochs 6 --lr 3e-4]"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from datasets import load_from_disk
from model2vec import StaticModel
from bgmodel import encode_ids, tok_ids
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("init"); ap.add_argument("out")
ap.add_argument("--buckets", type=int, default=200003); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--bs", type=int, default=2048)
a = ap.parse_args(); dev = "cuda"; B = a.buckets
sm = StaticModel.from_pretrained(a.init, quantize_to="float32")
ds = load_from_disk(a.data); texts = ds["text"]; Y = torch.tensor(np.array(ds["embedding"], dtype=np.float32))
ids = torch.cat([tok_ids(sm, texts[s:s + 50000]) for s in range(0, len(texts), 50000)]).int()
N = len(texts); print("tokenized", N, flush=True)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g); val, tr = perm[:20000], perm[20000:]
E = torch.nn.Parameter(torch.tensor(sm.embedding, dtype=torch.float32, device=dev))
Bg = torch.nn.Parameter(torch.zeros(B, 256, device=dev)); torch.nn.init.normal_(Bg, std=0.01)
opt = torch.optim.Adam([E, Bg], lr=a.lr)
def lf(ix):
    s = encode_ids(E, Bg, ids[ix].to(dev).long(), B); t = Y[ix].to(dev)
    cos = 1 - (s * t).sum(-1).mean()
    kl = F.kl_div(F.log_softmax(s @ s.T / a.tau, -1), F.softmax(t @ t.T / a.tau, -1), reduction="batchmean")
    return cos + a.lam * kl, cos
steps = a.epochs * (len(tr) // a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
t0 = time.time()
for ep in range(a.epochs):
    p = tr[torch.randperm(len(tr))]
    for k in range(len(tr) // a.bs):
        l, c = lf(p[k * a.bs:(k + 1) * a.bs]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    with torch.no_grad():
        v = [lf(val[j:j + a.bs])[1].item() for j in range(0, 20000 - a.bs + 1, a.bs)]
    print(f"ep {ep} train_cos {c.item():.4f} val_cos_loss {np.mean(v):.4f} {time.time()-t0:.0f}s", flush=True)
torch.save(dict(E=E.detach().cpu(), Bg=Bg.detach().cpu(), B=B, base=a.init), a.out); print("saved", a.out, flush=True)
