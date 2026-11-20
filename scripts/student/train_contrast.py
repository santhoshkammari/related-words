"""usage: python -I train_contrast.py DATA INIT OUT [--epochs --lr --lc]  (unigram static, distill + InfoNCE on sense pairs)"""
import argparse, re, time, random, numpy as np, pandas as pd, torch, torch.nn.functional as F
from datasets import load_from_disk
from model2vec import StaticModel
from bgmodel import tok_ids
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("init"); ap.add_argument("out")
ap.add_argument("--root", default=".."); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lc", type=float, default=1.0); ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--bs", type=int, default=2048)
ap.add_argument("--pb", type=int, default=1024)
a = ap.parse_args(); dev = "cuda"
sm = StaticModel.from_pretrained(a.init, quantize_to="float32")
ds = load_from_disk(a.data); texts = ds["text"]; Y = torch.tensor(np.array(ds["embedding"], dtype=np.float32))
ids = torch.cat([tok_ids(sm, texts[s:s+50000]) for s in range(0, len(texts), 50000)]).int(); N = len(texts)
# sense pairs (queries -> doc), grouped by lemma
df = pd.read_parquet(f"{a.root}/data/senses.parquet"); df = df[(df.split=="train")&(df.zipf>=1.5)&(df.zipf<=5.0)]
groups = {}
for r in df.itertuples():
    q = [r.gloss_leaf]
    for e in r.examples:
        m = re.sub(r"\b"+re.escape(r.lemma[:max(3,len(r.lemma)-2)])+r"\w*", "___", e, flags=re.I)
        if "___" in m: q.append(m)
    groups.setdefault(r.lemma, []).append((f"{r.lemma} ({r.pos}): {r.gloss_leaf}", q))
G = [g for g in groups.values()]; print("lemmas", len(G), "senses", sum(map(len, G)), flush=True)
def pair_batch(n):
    ds_, qs_ = [], []
    for g in random.sample(G, n // 2):
        for d, q in g[:3]: ds_.append(d); qs_.append(random.choice(q))
    return ds_[:n], qs_[:n]
E = torch.nn.Parameter(torch.tensor(sm.embedding, dtype=torch.float32, device=dev))
def enc(i):
    m = (i >= 0)[..., None]; v = E[i.clamp(min=0)] * m
    return F.normalize(v.sum(1) / m.sum(1).clamp(min=1), dim=-1)
perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); val, tr = perm[:20000], perm[20000:]
opt = torch.optim.Adam([E], lr=a.lr); spe = len(tr)//a.bs
sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs*spe, pct_start=0.05); t0 = time.time(); random.seed(0)
for ep in range(a.epochs):
    p = tr[torch.randperm(len(tr))]
    for k in range(spe):
        ix = p[k*a.bs:(k+1)*a.bs]; s = enc(ids[ix].to(dev).long()); t = Y[ix].to(dev)
        cos = 1-(s*t).sum(-1).mean()
        kl = F.kl_div(F.log_softmax(s@s.T/a.tau,-1), F.softmax(t@t.T/a.tau,-1), reduction="batchmean")
        d, q = pair_batch(a.pb); di = tok_ids(sm, d).to(dev); qi = tok_ids(sm, q).to(dev)
        sd, sq = enc(di), enc(qi); lg = sq@sd.T/a.tau; lab = torch.arange(len(d), device=dev)
        nce = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))/2
        l = cos + kl + a.lc*nce; opt.zero_grad(); l.backward(); opt.step(); sch.step()
    with torch.no_grad():
        v = np.mean([1-(enc(ids[val[j:j+a.bs]].to(dev).long())*Y[val[j:j+a.bs]].to(dev)).sum(-1).mean().item() for j in range(0,20000-a.bs+1,a.bs)])
    print(f"ep {ep} cos {cos.item():.4f} nce {nce.item():.3f} val_cos {v:.4f} {time.time()-t0:.0f}s", flush=True)
    sm.embedding = E.detach().cpu().numpy(); sm.save_pretrained(a.out)
print("saved", a.out, flush=True)
