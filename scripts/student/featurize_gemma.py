"""Training data for the potion student: (text, Gemma-256 embedding). No Sage/synthetic-LLM text.
Sources (all public / our train split): dictionary entries (doc prompt), dictionary usage examples (query prompt),
C4 paragraphs and Wikipedia sentences (query prompt). usage: python -I featurize_gemma.py OUT [--n_c4 N --n_wiki N]"""
import re, os, itertools, argparse, numpy as np, pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--root", default=".")
ap.add_argument("--n_c4", type=int, default=150000); ap.add_argument("--n_wiki", type=int, default=150000)
ap.add_argument("--dim", type=int, default=256); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--teacher", default="google/embeddinggemma-300m")
a = ap.parse_args()
DOC, QRY = "title: none | text: ", "task: search result | query: "
df = pd.read_parquet(f"{a.root}/data/senses.parquet")
df = df[(df.split == "train") & (df.zipf >= 1.5) & (df.zipf <= 5.0)]
items = []                                   # (text, prompt)
for r in df.itertuples():
    items.append((f"{r.lemma} ({r.pos}): {r.gloss_leaf}", DOC))
    stem = re.escape(r.lemma[:max(3, len(r.lemma) - 2)])
    for e in r.examples:
        items.append((e, QRY))
        m = re.sub(r"\b" + stem + r"\w*", "___", e, flags=re.I)
        if "___" in m: items.append((m, QRY))
n_dict = len(items)
c4 = []
for x in itertools.islice(load_dataset("allenai/c4", name="en", split="train", streaming=True), a.n_c4 * 2):
    p = x["text"].split("\n")[0].strip()
    if 40 <= len(p) <= 400: c4.append(p)
    if len(c4) >= a.n_c4: break
wk = []
ws = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train", streaming=True)
for x in itertools.islice(ws, a.n_wiki * 2):
    s = (x.get("sentence") or x.get("text") or "").strip()
    if 30 <= len(s) <= 300: wk.append(s)
    if len(wk) >= a.n_wiki: break
items += [(t, QRY) for t in c4 + wk]
seen = {}; [seen.setdefault(t, p) for t, p in items]; items = list(seen.items())
print(f"dict+examples {n_dict:,} | c4 {len(c4):,} | wiki {len(wk):,} -> {len(items):,} unique", flush=True)
m = SentenceTransformer(a.teacher, device="cuda"); m.max_seq_length = 128
m.to(__import__("torch").bfloat16)
order = np.argsort([len(t) for t, _ in items]); emb = np.zeros((len(items), a.dim), np.float32)
import time; t0 = time.time()
for prompt in (DOC, QRY):                     # one prompt per pass (sentence-transformers batches share a prompt)
    idx = [i for i in order if items[i][1] == prompt]
    for s in range(0, len(idx), a.bs * 16):
        b = idx[s:s + a.bs * 16]
        e = m.encode([items[j][0] for j in b], prompt=prompt, batch_size=a.bs, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[:, :a.dim]
        emb[b] = e / np.linalg.norm(e, axis=1, keepdims=True)
        if (s // (a.bs * 16)) % 20 == 0: print(f"{prompt[:6]} {s}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
assert np.isfinite(emb).all()
Dataset.from_dict({"text": [t for t, _ in items], "embedding": emb.tolist()}).save_to_disk(a.out)
print("saved", a.out, f"{time.time()-t0:.0f}s", flush=True); os._exit(0)
