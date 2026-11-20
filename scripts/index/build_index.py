"""Embed word-pool senses with Gemma; save data/index/{emb.npy (fp16, 768-d), meta.parquet}."""
import sys, os, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
root = os.environ.get("WSR_ROOT", "."); lo, hi = 1.5, 5.0
df = pd.read_parquet(f"{root}/data/senses.parquet")
pool = df[(df.zipf >= lo) & (df.zipf <= hi)].reset_index(drop=True)
docs = (pool.lemma + " (" + pool.pos + "): " + pool.gloss_leaf).tolist()
print("senses", len(pool), "lemmas", pool.lemma.nunique(), flush=True)
m = SentenceTransformer("google/embeddinggemma-300m", device="cuda"); m.to(torch.bfloat16)
order = np.argsort([len(t) for t in docs])           # length-sorted batches: much faster
E = np.zeros((len(docs), 768), np.float16)
for i in range(0, len(docs), 4096):
    ix = order[i:i + 4096]
    E[ix] = m.encode([docs[j] for j in ix], batch_size=64, prompt_name="document", normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True).float().cpu().numpy().astype(np.float16)
    print(i + len(ix), flush=True)
assert np.isfinite(E.astype(np.float32)).all(), "NaN in embeddings"
os.makedirs(f"{root}/data/index", exist_ok=True)
np.save(f"{root}/data/index/emb.npy", E)
pool[["sense_id", "lemma", "pos", "gloss_leaf", "zipf"]].to_parquet(f"{root}/data/index/meta.parquet")
print("saved", E.shape, flush=True)
