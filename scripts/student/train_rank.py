"""Fine-tune a static student with cosine + in-batch similarity-structure distillation (retrieval-oriented).
usage: python -I train_rank.py DATA INIT OUT [--epochs 6 --lr 3e-4 --tau 0.05 --lam 1.0 --bs 2048]"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from datasets import load_from_disk
from model2vec import StaticModel
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("init"); ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--bs", type=int, default=2048); ap.add_argument("--maxlen", type=int, default=64)
a = ap.parse_args(); dev = "cuda"
m = StaticModel.from_pretrained(a.init, quantize_to="float32"); assert m.weights is None
ds = load_from_disk(a.data); texts = ds["text"]; Y = torch.tensor(np.array(ds["embedding"], dtype=np.float32))
mp = m.token_mapping if m.token_mapping is not None else np.arange(len(m.embedding))
t0 = time.time(); N = len(texts); L = a.maxlen
ids = np.full((N, L), -1, np.int32)
for s in range(0, N, 50000):
    for i, l in enumerate(m.tokenize(texts[s:s + 50000])):
        l = mp[l[:L]] if len(l) else []
        ids[s + i, :len(l)] = l
print(f"tokenized {N:,} in {time.time()-t0:.0f}s", flush=True)
ids = torch.from_numpy(ids); perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
val, tr = perm[:20000], perm[20000:]
E = torch.nn.Parameter(torch.tensor(m.embedding, dtype=torch.float32, device=dev))
opt = torch.optim.Adam([E], lr=a.lr)
def enc(ix):
    b = ids[ix].to(dev).long(); mask = (b >= 0); e = E[b.clamp(min=0)] * mask[..., None]
    return F.normalize(e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1), dim=-1)
def loss_fn(ix):
    s = enc(ix); t = Y[ix].to(dev)
    cos = 1 - (s * t).sum(-1).mean()
    ps = F.log_softmax(s @ s.T / a.tau, -1); pt = F.softmax(t @ t.T / a.tau, -1)
    return cos + a.lam * F.kl_div(ps, pt, reduction="batchmean"), cos
steps = a.epochs * (len(tr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
best = 9; st = 0
for ep in range(a.epochs):
    p = tr[torch.randperm(len(tr))]
    for k in range(len(tr) // a.bs):
        l, c = loss_fn(p[k * a.bs:(k + 1) * a.bs]); opt.zero_grad(); l.backward(); opt.step(); sched.step(); st += 1
    with torch.no_grad():
        vl = [loss_fn(val[j:j + a.bs]) for j in range(0, len(val) - a.bs + 1, a.bs)]
        print(f"ep {ep} train {l.item():.4f} val_total {np.mean([x[0].item() for x in vl]):.4f} val_cos_loss {np.mean([x[1].item() for x in vl]):.4f} {time.time()-t0:.0f}s", flush=True)
m2 = StaticModel(vectors=E.detach().cpu().numpy(), tokenizer=m.tokenizer, config=m.config, token_mapping=m.token_mapping, normalize=True)
m2.save_pretrained(a.out); print("saved", a.out, flush=True)
