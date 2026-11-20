"""python suggest.py "text" ... | reads stdin lines if no args. Prints top words (lemma = best sense)."""
import sys, os, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
root = os.environ.get("WSR_ROOT", "."); K = int(os.environ.get("K", 8)); dev = os.environ.get("DEV", "cpu")
E = torch.from_numpy(np.load(f"{root}/data/index/emb.npy")).float(); M = pd.read_parquet(f"{root}/data/index/meta.parquet")
m = SentenceTransformer(os.environ.get("GEMMA", "models/embeddinggemma-300m"), device=dev)
texts = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
Q = m.encode(texts, prompt_name="query", normalize_embeddings=True, convert_to_tensor=True).float().cpu()
S = Q @ E.T
for t, s in zip(texts, S):
    top = s.topk(300).indices.tolist(); seen = []; print(f"\n> {t}")
    for i in top:
        r = M.iloc[i]
        if r.lemma in seen: continue
        seen.append(r.lemma); print(f"  {s[i]:.3f}  {r.lemma} ({r.pos}, zipf {r.zipf:.1f}): {r.gloss_leaf[:90]}")
        if len(seen) == K: break
