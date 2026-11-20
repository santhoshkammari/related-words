"""Eval C: WordNet definition -> all lemmas of the synset (that exist in our pool). python -P make_wn_eval.py ROOT N"""
import sys, re, json, numpy as np, pandas as pd
import nltk
nltk.data.path.insert(0, sys.argv[1] + "/data/nltk")
from nltk.corpus import wordnet as wn
root, N = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(f"{root}/data/senses.parquet", columns=["lemma", "zipf", "split"]).drop_duplicates("lemma")
pool = set(df[(df.zipf >= 1.5) & (df.zipf <= 5.0)].lemma)
test = set(df[(df.split == "test") & (df.zipf >= 2.5) & (df.zipf <= 4.5)].lemma)
items = []
for s in wn.all_synsets():
    lem = [l.name().lower() for l in s.lemmas()]
    lem = [l for l in lem if re.fullmatch(r"[a-z][a-z\-]*", l)]
    inpool = [l for l in dict.fromkeys(lem) if l in pool]
    if not (set(inpool) & test): continue
    d = s.definition().strip()
    if len(d) < 20: continue
    leak = any(re.search(r"\b" + re.escape(l[:max(4, len(l) - 2)]), d, re.I) for l in inpool)
    if leak: continue
    items.append(dict(synset=s.name(), query=d, targets=inpool))
rng = np.random.default_rng(0)
sel = rng.choice(len(items), size=min(N, len(items)), replace=False)
out = [items[i] for i in sorted(sel)]
with open(f"{root}/eval/wn_defs.jsonl", "w") as f:
    for i, o in enumerate(out): f.write(json.dumps(dict(id=i, **o)) + "\n")
print(len(items), "candidate synsets ->", len(out), "written; mean #targets", round(float(np.mean([len(o['targets']) for o in out])), 2))
for o in out[:3]: print(o["query"], "->", o["targets"])
