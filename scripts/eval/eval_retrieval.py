"""Recall@k / MRR for situation -> lemma retrieval over the sense table.
Eval A: 64 hand-written situations (eval/situations.jsonl), any acceptable target counts.
Eval B: held-out dictionary example sentences (test-split lemmas), target word masked, truth = lemma.
usage: python -I eval_retrieval.py MODEL_SPEC [--tag name] [--index leaf|full] [--nB 2000]
MODEL_SPEC: st:<hf-name>[|q=<query prefix>|d=<doc prefix>]   or   m2v:<hf-name-or-path>"""
import sys, re, json, time, argparse, numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("spec"); ap.add_argument("--tag"); ap.add_argument("--index", default="leaf")
ap.add_argument("--nB", type=int, default=2000); ap.add_argument("--pool_lo", type=float, default=1.5)
ap.add_argument("--pool_hi", type=float, default=5.0); ap.add_argument("--root", default=".")
ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="runs/eval")
a = ap.parse_args()

parts = a.spec.split("|"); kind, name = parts[0].split(":", 1)
opts = dict(p.split("=", 1) for p in parts[1:])
qp, dp = opts.get("q", ""), opts.get("d", "")

df = pd.read_parquet(f"{a.root}/data/senses.parquet")
pool = df[(df.zipf >= a.pool_lo) & (df.zipf <= a.pool_hi)].reset_index(drop=True)
lemmas = pool.lemma.unique().tolist(); lid = {l: i for i, l in enumerate(lemmas)}
sid = torch.tensor(pool.lemma.map(lid).values)
gl = pool.gloss_leaf if a.index in ("leaf", "syn") else pool.gloss.str.replace(" > ", ": ")
if a.index == "syn":
    gl = gl + pool.synonyms.map(lambda s: (" | synonyms: " + ", ".join(s[:5])) if len(s) else "")
docs = (dp + pool.lemma + " (" + pool.pos + "): " + gl).tolist()

if kind == "st":
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device=a.device)
    dim = int(opts["dim"]) if "dim" in opts else None
    def enc(t, bs=32, pn=None):
        pn = pn if pn is not None else (opts.get("dn") if enc.is_doc else opts.get("qn"))
        x = m.encode(t, batch_size=bs, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False, prompt_name=pn).float().cpu()
        if dim: x = torch.nn.functional.normalize(x[:, :dim], dim=1)
        return x
    enc.is_doc = True
elif kind == "m2v":
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(name)
    def enc(t, bs=256):
        x = torch.from_numpy(np.asarray(m.encode(t), dtype=np.float32)); return torch.nn.functional.normalize(x, dim=1)

elif kind == "tf":
    import sys; sys.path.insert(0, f"{a.root}/src")
    from bgmodel import tok_ids
    from tfmodel import TF
    from model2vec import StaticModel
    ck = torch.load(name); sm = StaticModel.from_pretrained(ck["base"], quantize_to="float32")
    net = TF(torch.tensor(sm.embedding), layers=ck["layers"]).cuda().eval(); net.load_state_dict(ck["sd"])
    def enc(t, bs=512):
        out = []
        for i in range(0, len(t), bs):
            ii = tok_ids(sm, list(t[i:i+bs])).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): out.append(net(ii[:, :max(int((ii>=0).sum(1).max()),1)]).float().cpu())
        return torch.cat(out)
elif kind == "bg":
    import sys; sys.path.insert(0, f"{a.root}/src")
    from bgmodel import encode_ids, tok_ids
    from model2vec import StaticModel
    ck = torch.load(name); sm = StaticModel.from_pretrained(ck["base"]); Ec, Bc = ck["E"].cuda(), ck["Bg"].cuda()
    def enc(t, bs=1024):
        out = []
        for i in range(0, len(t), bs):
            with torch.no_grad(): out.append(encode_ids(Ec, Bc, tok_ids(sm, list(t[i:i+bs])).cuda(), ck["B"]).cpu())
        return torch.cat(out)

t0 = time.time(); D = enc(docs); idx_s = time.time() - t0
if kind == "st": enc.is_doc = False
dev = a.device if torch.cuda.is_available() else "cpu"
D = D.to(dev); sid_d = sid.to(dev); L = len(lemmas)

def lemma_scores(Q):                      # (q, L): max over a lemma's senses
    out = []
    for i in range(0, len(Q), 256):
        s = Q[i:i + 256].to(dev) @ D.T
        o = torch.full((s.shape[0], L), -1e9, device=dev).scatter_reduce_(1, sid_d.expand(s.shape[0], -1), s, "amax")
        out.append(o.cpu())
    return torch.cat(out)

def metrics(sc, truth):                   # truth: list of sets of lemma ids
    ranks = []
    for r, tr in zip(sc, truth):
        tr = list(tr)
        if not tr: continue
        best = r[tr].max(); ranks.append(int((r > best).sum()) + 1)
    ranks = np.array(ranks)
    return dict(n=len(ranks), r1=float((ranks <= 1).mean()), r5=float((ranks <= 5).mean()), r10=float((ranks <= 10).mean()),
                r50=float((ranks <= 50).mean()), mrr=float((1 / ranks).mean()), med_rank=float(np.median(ranks)))

# Eval A
ev = [json.loads(l) for l in open(f"{a.root}/eval/situations.jsonl")]
QA = enc([qp + e["situation"] for e in ev], 64)
trA = [{lid[t] for t in e["targets"] if t in lid} for e in ev]
resA = metrics(lemma_scores(QA), trA)
resA_primary = metrics(lemma_scores(QA), [{lid[e["targets"][0]]} if e["targets"][0] in lid else set() for e in ev])

# Eval C: WordNet definitions -> synset lemmas
evc = [json.loads(l) for l in open(f"{a.root}/eval/wn_defs.jsonl")]
resC = metrics(lemma_scores(enc([qp + e["query"] for e in evc])), [{lid[t] for t in e["targets"] if t in lid} for e in evc])

# Eval B: held-out examples, target masked
rng = np.random.default_rng(0)
tb = df[(df.split == "test") & (df.zipf >= 2.5) & (df.zipf <= 4.5) & (df.examples.str.len() > 0) & df.lemma.isin(lid)]
cand = [(r.lemma, e) for r in tb.itertuples() for e in r.examples]
sel = rng.choice(len(cand), size=min(a.nB, len(cand)), replace=False)
qB, trB = [], []
for k in sel:
    lem, ex = cand[k]
    stem = re.escape(lem[:max(3, len(lem) - 2)])
    masked = re.sub(r"\b" + stem + r"\w*", "___", ex, flags=re.I)
    if "___" not in masked: continue
    qB.append(qp + masked); trB.append({lid[lem]})
resB = metrics(lemma_scores(enc(qB)), trB)

# single-query CPU latency
lat = None
if kind == "st":
    m.to("cpu"); torch.set_num_threads(8); m.encode(["warm"]); t = time.time()
    for q in qB[:20]: m.encode([q])
    lat = (time.time() - t) / 20 * 1000
elif kind in ("bg", "tf"):
    lat = float("nan")
else:
    m.encode(["warm"]); t = time.time()
    for q in qB[:200]: m.encode([q])
    lat = (time.time() - t) / 200 * 1000

res = dict(spec=a.spec, index=a.index, pool=[a.pool_lo, a.pool_hi], n_docs=len(docs), n_lemmas=L, index_sec=round(idx_s, 1),
           cpu_query_ms=round(lat, 2), A=resA, A_primary=resA_primary, B=resB, C=resC)
tag = a.tag or re.sub(r"[^A-Za-z0-9_.-]", "_", name)
import os; os.makedirs(f"{a.root}/{a.out}", exist_ok=True)
json.dump(res, open(f"{a.root}/{a.out}/{tag}.json", "w"), indent=1)
f = lambda r: f"R@1 {r['r1']:.2f} R@5 {r['r5']:.2f} R@10 {r['r10']:.2f} R@50 {r['r50']:.2f} MRR {r['mrr']:.3f} med {r['med_rank']:.0f}"
print(f"{tag:34} A(n={resA['n']}): {f(resA)} | B(n={resB['n']}): {f(resB)} | C(n={resC['n']}): {f(resC)} | {lat:.1f} ms/q, index {idx_s:.0f}s")
