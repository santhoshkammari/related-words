"""Build (text, embedding) HF dataset for tokenlearn.train using the teacher's REAL sentence embeddings
(not stock mean-of-tokens). Corpus: train-split glosses/examples (raw+masked) + general C4 sentences.
usage: python -I featurize_teacher.py TEACHER OUT [--dim 0] [--prompt ''] [--n_general 120000] [--extra file.jsonl]"""
import re, os, sys, argparse, itertools, numpy as np, pandas as pd, torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("teacher"); ap.add_argument("out"); ap.add_argument("--root", default=".")
ap.add_argument("--dim", type=int, default=0); ap.add_argument("--prompt", default="")
ap.add_argument("--n_general", type=int, default=120000); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--max_len", type=int, default=128); ap.add_argument("--extra", default="")
a = ap.parse_args()

df = pd.read_parquet(f"{a.root}/data/senses.parquet")
df = df[(df.split == "train") & (df.zipf >= 1.5) & (df.zipf <= 5.0)]
texts = []
for r in df.itertuples():
    texts.append(f"{r.lemma} ({r.pos}): {r.gloss_leaf}")
    stem = re.escape(r.lemma[:max(3, len(r.lemma) - 2)])
    for e in r.examples:
        texts.append(e)
        m = re.sub(r"\b" + stem + r"\w*", "___", e, flags=re.I)
        if "___" in m: texts.append(m)
n_dict = len(texts)
if a.extra:  # e.g. synthetic situations: jsonl with a "text" field
    import json
    texts += [json.loads(l)["text"] for l in open(a.extra)]
n_extra = len(texts) - n_dict

ds = load_dataset("allenai/c4", name="en", split="train", streaming=True)
gen = []
for x in itertools.islice(ds, a.n_general * 2):
    para = x["text"].split("\n")[0].strip()
    if 40 <= len(para) <= 400: gen.append(para)
    if len(gen) >= a.n_general: break
texts += gen
texts = list(dict.fromkeys(texts))
print(f"corpus: dict {n_dict:,} + extra {n_extra:,} + general {len(gen):,} -> {len(texts):,} unique", flush=True)

m = SentenceTransformer(a.teacher, device="cuda"); m.max_seq_length = a.max_len
order = np.argsort([len(t) for t in texts])           # length-sorted batches -> fast
emb = np.zeros((len(texts), a.dim or m.get_sentence_embedding_dimension()), dtype=np.float32)
import time; t0 = time.time()
for i in range(0, len(texts), a.bs * 8):
    idx = order[i:i + a.bs * 8]
    e = m.encode([a.prompt + texts[j] for j in idx], batch_size=a.bs, convert_to_numpy=True, show_progress_bar=False)
    if a.dim:
        e = e[:, :a.dim]; e = e / np.linalg.norm(e, axis=1, keepdims=True)
    emb[idx] = e
print(f"encoded {len(texts):,} texts in {time.time() - t0:.0f}s -> {len(texts) / (time.time() - t0):.0f} sent/s | dim {emb.shape[1]}", flush=True)
Dataset.from_dict({"text": texts, "embedding": emb.tolist()}).save_to_disk(a.out)
print("saved", a.out, flush=True)
os._exit(0)
